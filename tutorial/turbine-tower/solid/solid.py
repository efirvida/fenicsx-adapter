import os
from typing import NamedTuple

# enforce 1 thread per core
nthreads = 1
os.environ["OMP_NUM_THREADS"] = str(nthreads)
os.environ["OPENBLAS_NUM_THREADS"] = str(nthreads)
os.environ["MKL_NUM_THREADS"] = str(nthreads)

import gmsh
from dolfinx import fem, io, default_scalar_type
import numpy as np
import ufl
from fenicsxprecice import Adapter, DiscreteLinearProblem
from mpi4py import MPI
import numpy as np


# --------- #
# CONSTANTS #
# --------- #

MPI_COMM = MPI.COMM_WORLD
CURRENT_FOLDER = os.path.dirname(__file__)
RESULTS_DIR = os.path.join(CURRENT_FOLDER, "results")
PARTICIPANT_CONFIG = os.path.join(CURRENT_FOLDER, "precice-config.json")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.chdir(CURRENT_FOLDER)

WRITER = io.VTKFile(MPI_COMM, f"{RESULTS_DIR}/result.pvd", "w")


# -------------------#
# Problem properties #
# -------------------#
# https://www.nrel.gov/docs/fy20osti/75698.pdf
class Material(NamedTuple):
    name: str
    E: default_scalar_type
    nu: default_scalar_type
    rho: default_scalar_type


material = Material("Aluminum 6061", 200e11, 0.4, 785e3)

# beam size
height = 150  # m
diameter = 6.5  # m
thickness = 0.055 # 55mm

BETA_ = 0.25
GAMMA_ = 0.5

# ------- #
# MESHING #
# ------- #
FIXED_TAG = 1000
BEAM_SURFACE_TAG = 2000
TIP_SURFACE_TAG = 3000
VOLUME_TAG = 4000
ELEMENTS_ORDER = 1
VECTOR_SPACE_DEGREE = 2


def gmsh_tower(h: float, d: float) -> gmsh.model:
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    model_name = "Tower"
    gmsh.model.add(model_name)

    model = gmsh.model()
    model.setCurrent(model_name)

    # Recombine tetrahedra to hexahedra
    gmsh.option.setNumber("Mesh.RecombinationAlgorithm", 2)
    gmsh.option.setNumber("Mesh.RecombineAll", 2)
    gmsh.option.setNumber("Mesh.CharacteristicLengthFactor", 0.02)

    outer_circle = model.occ.addDisk(0, 0, 0, d / 2, d / 2)
    inner_circle = model.occ.addDisk(0, 0, 0, d / 2 - thickness, d / 2 - thickness)

    outer = model.occ.extrude([(2, outer_circle)], 0, 0, h, numElements=[1.5*h], recombine=True)
    inner = model.occ.extrude([(2, inner_circle)], 0, 0, h, numElements=[1.5*h], recombine=True)

    outer_volume = outer[1][1]
    inner_volume = inner[1][1]
    model.occ.cut([(3, outer_volume)], [(3, inner_volume)])
    
    model.occ.synchronize()

    fixed_sf = model.addPhysicalGroup(2, [7], tag=FIXED_TAG)
    tip_sf = model.addPhysicalGroup(2, [6,8], tag=BEAM_SURFACE_TAG)
    vol = model.addPhysicalGroup(3, [1], tag=VOLUME_TAG)

    model.setPhysicalName(2, fixed_sf, "FIXED SURFACE")
    model.setPhysicalName(2, tip_sf, "LOAD SURFACE")
    model.setPhysicalName(3, vol, "Mesh volume")

    model.mesh.setOrder(ELEMENTS_ORDER)
    model.mesh.generate(3)
    model.mesh.optimize()
    gmsh.write("mesh.vtk")
    return model

model = gmsh_tower(h=height, d=diameter)
domain, cell_markers, facet_markers = io.gmshio.model_to_mesh(model, MPI_COMM, rank=0)

# -------------- #
# Function Space #
# -------------- #
dim = domain.geometry.dim
V = fem.functionspace(domain, ("Lagrange", VECTOR_SPACE_DEGREE, (dim,)))
u = fem.Function(V, name="Displacement")

bs = V.dofmap.index_map_bs
num_dofs_local = V.dofmap.index_map.size_local
print(f"Rank {domain.comm.rank}, Block size {bs} Num local dofs {num_dofs_local*bs}")

# --------------------#
# Boundary conditions #
# --------------------#
# Fixed BC
fixed_facets = facet_markers.find(FIXED_TAG)
fixed_surface_dofs = fem.locate_dofs_topological(V, dim - 1, fixed_facets)

coupling_boundary_markers = facet_markers.find(BEAM_SURFACE_TAG)

u_bc = fem.Function(V)
bcs = [fem.dirichletbc(u_bc, fixed_surface_dofs)]

# ------------ #
# PRECICE INIT #
# ------------ #
participant = Adapter(MPI_COMM, PARTICIPANT_CONFIG, domain)
participant.initialize(V, coupling_boundary_markers)
dt = participant.dt
np.savetxt(
    "coupling_points.csv",
    participant.interface_coordinates,
    delimiter=",",
    header="X,Y,Z",
)

# ------------------------ #
# linear elastic equations #
# ------------------------ #
E = fem.Constant(domain, material.E)
nu = fem.Constant(domain, material.nu)
rho = fem.Constant(domain, material.rho)

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
n = 0

while participant.is_coupling_ongoing():
    if participant.requires_writing_checkpoint():  # write checkpoint
        participant.store_checkpoint((u_old, v_old, a_old, t))

    read_data = participant.read_data(dt)

    u = problem.solve(read_data)

    # Write new displacements to preCICE
    participant.write_data(u)

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
        if n % 10 == 0:
            WRITER.write_function(u, t)
            WRITER.close()

WRITER.close()
participant.finalize()
