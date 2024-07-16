import os
import glob

import numpy as np
import ufl
from mpi4py import MPI

import dolfinx as dfx
from dolfinx.mesh import create_rectangle, CellType
from petsc4py import PETSc
from dolfinx.fem.petsc import apply_lifting, assemble_matrix, set_bc

from interface import Adapter


class Material(NamedTuple):
    name: str
    E: PETSc.ScalarType
    nu: PETSc.ScalarType
    rho: PETSc.ScalarType


# --------- #
# CONSTANTS #
# --------- #

MPI_COMM = MPI.COMM_WORLD
CURRENT_FOLDER = os.path.dirname(__file__)
PARTICIPANT_CONFIG = os.path.join(CURRENT_FOLDER, "precice-config.json")
RESULTS_DIR = os.path.join(CURRENT_FOLDER, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.chdir(CURRENT_FOLDER)

WRITER = dfx.io.VTKFile(MPI_COMM, f"{RESULTS_DIR}/result.pvd", "w")


WIDTH, HEIGHT = 0.1, 1
NX, NY = 4, 26

flap_material = Material("Solid properties", E=4e6, nu=0.3, rho=3000.0)

BETA_ = 0.25
GAMMA_ = 0.5

# ------- #
# MESHING #
# ------- #

domain = create_rectangle(
    MPI_COMM,
    [np.array([-WIDTH / 2, 0]), np.array([WIDTH / 2, HEIGHT])],
    [NX, NY],
    cell_type=CellType.quadrilateral,
)
dim = domain.topology.dim
domain.topology.create_connectivity(dim - 1, dim)
WRITER.write_mesh(domain)

# -------------- #
# Function Space #
# -------------- #
degree = 2
shape = (dim,)
V = dfx.fem.functionspace(domain, ("P", degree, shape))
u = dfx.fem.Function(V, name="Displacement")

# ------------------- #
# Boundary conditions #
# ------------------- #


def clamped_boundary(x):
    return np.isclose(x[1], 0.0)


def neumann_boundary(x):
    return np.logical_or(np.isclose(x[1], HEIGHT), np.isclose(np.abs(x[0]), WIDTH / 2))


fixed_boundary = dfx.fem.locate_dofs_geometrical(V, clamped_boundary)
coupling_boundary = dfx.mesh.locate_entities_boundary(domain, dim - 1, neumann_boundary)
coupling_boundary_tags = dfx.mesh.meshtags(domain, dim - 1, np.sort(coupling_boundary), 1)

with open(f"{RESULTS_DIR}/fixed_boundary.csv", "w") as p_file:
    p_file.write("X,Y,Z\n")
    np.savetxt(p_file, V.tabulate_dof_coordinates()[fixed_boundary], delimiter=",")

bc = dfx.fem.dirichletbc(np.zeros((dim,)), fixed_boundary, V)

# ------------ #
# PRECICE INIT #
# ------------ #
participant = Adapter(MPI_COMM, PARTICIPANT_CONFIG, domain, flap_material)
participant.initialize(V, coupling_boundary)
with open(f"{RESULTS_DIR}/interpolation_points.csv", "w") as p_file:
    p_file.write("X,Y\n")
    np.savetxt(p_file, participant.interface_coordinates, delimiter=",")

dt = dfx.fem.Constant(domain, PETSc.ScalarType(participant.dt))

# ------------------------ #
# linear elastic equations #
# ------------------------ #

E = dfx.fem.Constant(domain, flap_material.E)
nu = dfx.fem.Constant(domain, flap_material.nu)
rho = dfx.fem.Constant(domain, flap_material.rho)

lmbda = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
mu = E / (2.0 * (1.0 + nu))


def epsilon(v):
    return 0.5 * (ufl.grad(v) + ufl.grad(v).T)


def sigma(v):
    return lmbda * ufl.div(v) * ufl.Identity(dim) + 2 * mu * epsilon(v)


# ------------------- #
# Time discretization #
# ------------------- #
# prev time step
u_old = dfx.fem.Function(V)
v_old = dfx.fem.Function(V)
a_old = dfx.fem.Function(V)

# current time step
a_new = dfx.fem.Function(V)
v_new = dfx.fem.Function(V)

beta = dfx.fem.Constant(domain, BETA_)
gamma = dfx.fem.Constant(domain, GAMMA_)

dx = ufl.Measure("dx", domain=domain)


a = 1 / beta / dt**2 * (u - u_old - dt * v_old) + a_old * (1 - 1 / 2 / beta)
a_expr = dfx.fem.Expression(a, V.element.interpolation_points())

v = v_old + dt * ((1 - gamma) * a_old + gamma * a)
v_expr = dfx.fem.Expression(v, V.element.interpolation_points())

# ------------------ #
# mass, a stiffness  #
# ------------------ #
u_ = ufl.TestFunction(V)
du = ufl.TrialFunction(V)


def mass(u, u_):
    return rho * ufl.dot(u, u_) * ufl.dx


def stiffness(u, u_):
    return ufl.inner(sigma(u), epsilon(u_)) * ufl.dx


Residual = mass(a, u_) + stiffness(u, u_)
Residual_du = ufl.replace(Residual, {u: du})


# ------------------ #
# assemble and solve #
# ------------------ #
a = dfx.fem.form(ufl.lhs(Residual_du))-



A = assemble_matrix(a, bcs=[bc])
A.assemble()

b_func = dfx.fem.Function(V, name="Load")
b_func.x.array[:] = 0
b = b_func.vector
with b.localForm() as b_local:
    b_local.set(0)

solver = PETSc.KSP().create(domain.comm)  # type: ignore
solver.setFromOptions()
solver.setType(PETSc.KSP.Type.CG)
pc = solver.getPC()
pc.setType(PETSc.PC.Type.GAMG)
solver.setTolerances(rtol=1e-5, atol=1e-11, max_it=1000)
solver.setOperators(A)

t = 0.0
uh = dfx.fem.Function(V, name="U")

while participant.is_coupling_ongoing():
    if participant.requires_writing_checkpoint():
        participant.store_checkpoint(u_old, v_old, a_old, t)

    read_data = participant.read_data(participant.dt)

    A = assemble_matrix(a, bcs=[bc])
    A.assemble()

    # apply incoming loads to the Function dof
    b[participant.interface_dof] = read_data

    b.assemble()
    apply_lifting(b, [a], bcs=[[bc]])
    b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)  # type: ignore
    set_bc(b, [bc])

    solver.solve(b, uh.vector)
    uh.x.scatter_forward()

    # Write new displacements to preCICE
    participant.write_data(uh)

    # Call to advance coupling, also returns the optimum time step value
    participant.advance(participant.dt)

    if participant.requires_reading_checkpoint():
        u_cp, v_cp, a_cp, t_cp = participant.retrieve_checkpoint()
        u_cp.vector.copy(u_old.vector)
        v_cp.vector.copy(v_old.vector)
        a_cp.vector.copy(a_old.vector)
        t = t_cp
    else:
        v_new.interpolate(v_expr)
        a_new.interpolate(a_expr)
        u.vector.copy(u_old.vector)
        v_new.vector.copy(v_old.vector)
        a_new.vector.copy(a_old.vector)
        t += dt.value

    if participant.is_time_window_complete():
        WRITER.write_function(uh, t)
        WRITER.close()


WRITER.close()
participant.finalize()
