import os

import gmsh
from dolfinx import fem, io
import numpy as np
import ufl
from fenicsxprecice import Adapter, DiscreteLinearProblem
from mpi4py import MPI

# --------- #
# CONSTANTS #
# --------- #

MPI_COMM = MPI.COMM_WORLD
CURRENT_FOLDER = os.path.dirname(__file__)
PARTICIPANT_CONFIG = os.path.join(CURRENT_FOLDER, "precice-config.json")
RESULTS_DIR = os.path.join(CURRENT_FOLDER, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.chdir(CURRENT_FOLDER)

WRITER = io.VTKFile(MPI_COMM, f"{RESULTS_DIR}/result.pvd", "w")

R = 0.005
L = 0.05

E = 300000.0
nu = 0.3
rho = 1200.0

BETA_ = 0.25
GAMMA_ = 0.5


# ------- #
# MESHING #
# ------- #
INLET = 1000
OUTLET = 2000
INTERFACE = 3000
VOLUME_TAG = 4000
ELEMENTS_ORDER = 2
VECTOR_SPACE_DEGREE = 2


def gmsh_outer(R: float, L: float) -> gmsh.model:
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    model_name = "Tower"
    gmsh.model.add(model_name)

    model = gmsh.model()
    model.setCurrent(model_name)

    # Recombine tetrahedra to hexahedra
    gmsh.option.setNumber("Mesh.RecombinationAlgorithm", 2)
    gmsh.option.setNumber("Mesh.RecombineAll", 1)
    gmsh.option.setNumber("Mesh.CharacteristicLengthFactor", 0.1)

    outer_circle = model.occ.addDisk(0, 0, 0, R + 0.001, R + 0.001)
    inner_circle = model.occ.addDisk(0, 0, 0, R, R)

    outer = model.occ.extrude(
        [(2, outer_circle)], 0, 0, L, numElements=[40], recombine=True
    )
    inner = model.occ.extrude(
        [(2, inner_circle)], 0, 0, L, numElements=[40], recombine=True
    )

    outer_volume = outer[1][1]
    inner_volume = inner[1][1]
    vol = model.occ.cut([(3, outer_volume)], [(3, inner_volume)])

    model.occ.synchronize()

    inlet_sf = model.addPhysicalGroup(2, [7], tag=INLET)
    outlet_sf = model.addPhysicalGroup(2, [8], tag=OUTLET)
    interface_sf = model.addPhysicalGroup(2, [5], tag=INTERFACE)
    vol = model.addPhysicalGroup(3, [1], tag=VOLUME_TAG)

    model.setPhysicalName(2, inlet_sf , "INLET")
    model.setPhysicalName(2, outlet_sf, "OUTLET")
    model.setPhysicalName(2, interface_sf, "INTERFACE")
    model.setPhysicalName(3, vol, "VOLUME")
    model.occ.synchronize()

    model.mesh.generate(3)
    model.mesh.setOrder(ELEMENTS_ORDER)
    model.mesh.optimize()
    # gmsh.write("solid.msh")
    gmsh.write("solid.vtk")
    return model

def gmsh_inner(R: float, L: float) -> gmsh.model:
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    model_name = "Tower"
    gmsh.model.add(model_name)

    model = gmsh.model()
    model.setCurrent(model_name)

    # Recombine tetrahedra to hexahedra
    gmsh.option.setNumber("Mesh.CharacteristicLengthFactor", 0.1)

    circle = model.occ.addDisk(0, 0, 0, R, R)

    inner = model.occ.extrude(
        [(2, circle)], 0, 0, L, numElements=[40], recombine=True
    )

    model.occ.synchronize()

    inlet_sf = model.addPhysicalGroup(2, [1], tag=INLET)
    outlet_sf = model.addPhysicalGroup(2, [3], tag=OUTLET)
    interface_sf = model.addPhysicalGroup(2, [2], tag=INTERFACE)
    vol = model.addPhysicalGroup(3, [1], tag=VOLUME_TAG)

    model.setPhysicalName(2, inlet_sf , "INLET")
    model.setPhysicalName(2, outlet_sf, "OUTLET")
    model.setPhysicalName(2, interface_sf, "INTERFACE")
    model.setPhysicalName(3, vol, "VOLUME")

    model.mesh.setOrder(ELEMENTS_ORDER)
    model.mesh.generate(3)
    model.mesh.optimize()
    gmsh.write("fluid.msh")
    gmsh.write("fluid.vtk")
    return model


model = gmsh_outer(R, L)
# gmsh.fltk.run()
# model = gmsh_inner(R, L)
domain, cell_markers, facet_markers = io.gmshio.model_to_mesh(model, MPI_COMM, rank=0)
domain.topology.create_connectivity(2, 3)

WRITER.write_mesh(domain)

# -------------- #
# Function Space #
# -------------- #
dim = domain.geometry.dim
V = fem.functionspace(domain, ("Lagrange", VECTOR_SPACE_DEGREE, (dim,)))
u = fem.Function(V, name="Displacement")
u_delta = fem.Function(V, name="DisplacementDelta")
f = fem.Function(V, name="Force")

bs = V.dofmap.index_map_bs
num_dofs_local = V.dofmap.index_map.size_local
print(f"Rank {domain.comm.rank}, Block size {bs} Num local dofs {num_dofs_local*bs}")

# --------------------#
# Boundary conditions #
# --------------------#
# Fixed BC
inlet_facets = facet_markers.find(INLET)
outlet_facets = facet_markers.find(OUTLET)
inlet_surface_dofs = fem.locate_dofs_topological(V, dim - 1, inlet_facets)
outlet_surface_dofs = fem.locate_dofs_topological(V, dim - 1, outlet_facets)

coupling_boundary_markers = facet_markers.find(INTERFACE)

u_bc = fem.Function(V)
bcs = [
    fem.dirichletbc(u_bc, inlet_surface_dofs),
    fem.dirichletbc(u_bc, outlet_surface_dofs),
]

# ------------ #
# PRECICE INIT #
# ------------ #
participant = Adapter(MPI_COMM, PARTICIPANT_CONFIG, domain)
participant.initialize(V, coupling_boundary_markers)
np.savetxt("dofs.txt", participant.interface_coordinates, delimiter=",", header="x,y,z")
dt = participant.dt

# ------------------------ #
# linear elastic equations #
# ------------------------ #

E = fem.Constant(domain, E)
nu = fem.Constant(domain, nu)
rho = fem.Constant(domain, rho)

lmbda = E * nu / (1 + nu) / (1 - 2 * nu)
mu = E / 2 / (1 + nu)


def epsilon(v):
    return ufl.sym(ufl.grad(v))


def sigma(v):
    return lmbda * ufl.tr(epsilon(v)) * ufl.Identity(dim) + 2 * mu * epsilon(v)


# ------------------- #
# Time discretization #
# ------------------- #
# prev time step
u_old = fem.Function(V)
v_old = fem.Function(V)
a_old = fem.Function(V)

# current time step
a_new = fem.Function(V)
v_new = fem.Function(V)

beta = fem.Constant(domain, BETA_)
gamma = fem.Constant(domain, GAMMA_)

dx = ufl.Measure("dx", domain=domain)

a = (1 / (beta * dt**2)) * (u - u_old - dt * v_old) - (
    (1 - 2 * beta) / (2 * beta)
) * a_old
a_expr = fem.Expression(a, V.element.interpolation_points())

v = v_old + dt * ((1 - gamma) * a_old + gamma * a)
v_expr = fem.Expression(v, V.element.interpolation_points())

# ------------------ #
# mass, a stiffness  #
# ------------------ #
u_ = ufl.TestFunction(V)
du = ufl.TrialFunction(V)


def mass(u, u_):
    return rho * ufl.dot(u, u_) * dx


def stiffness(u, u_):
    return ufl.inner(sigma(u), epsilon(u_)) * dx


Residual = mass(a, u_) + stiffness(u, u_)
Residual_du = ufl.replace(Residual, {u: du})
a_form = ufl.lhs(Residual_du)
L_form = ufl.rhs(Residual_du)

problem = DiscreteLinearProblem(
    a=a_form, L=L_form, u=u, bcs=bcs, point_dofs=participant.interface_dof
)

# parameters for Time-Stepping
t = 0.0

while participant.is_coupling_ongoing():
    if participant.requires_writing_checkpoint():  # write checkpoint
        participant.store_checkpoint(u_old, v_old, a_old, t)

    read_data = participant.read_data(dt)

    u = problem.solve(read_data)
    u_delta.vector[:] = u.vector[:] - u_old.vector[:]
    u_delta.vector.assemble()
    # Write new displacements to preCICE
    participant.write_data(u_delta)

    # Call to advance coupling, also returns the optimum time step value
    participant.advance(dt)

    # Either revert to old step if timestep has not converged or move to next timestep
    if participant.requires_reading_checkpoint():
        (
            u_cp,
            v_cp,
            a_cp,
            t_cp,
        ) = participant.retrieve_checkpoint()
        u_old.vector.copy(u_cp.vector)
        v_old.vector.copy(v_cp.vector)
        a_old.vector.copy(a_cp.vector)
        t = t_cp
    else:
        v_new.interpolate(v_expr)
        a_new.interpolate(a_expr)
        u.vector.copy(u_old.vector)
        v_new.vector.copy(v_old.vector)
        a_new.vector.copy(a_old.vector)
        t += dt

    if participant.is_time_window_complete():
        f.vector[participant.interface_dof] = read_data
        f.vector.assemble()

        WRITER.write_function(u, t)
        WRITER.write_function(f, t)
        WRITER.close()

WRITER.close()
participant.finalize()
