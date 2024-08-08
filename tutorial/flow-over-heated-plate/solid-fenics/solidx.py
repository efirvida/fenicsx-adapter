import ufl
import numpy as np
import os
import shutil

from petsc4py import PETSc
from mpi4py import MPI

from dolfinx import fem, io
from dolfinx.mesh import CellType, create_box, locate_entities_boundary
from dolfinx.fem.petsc import LinearProblem
from fenicsxprecice import Adapter, DiscreteLinearProblem


MPI_COMM = MPI.COMM_WORLD
CURRENT_FOLDER = os.path.dirname(__file__)
PARTICIPANT_CONFIG = os.path.join(CURRENT_FOLDER, "precice-adapter-config.json")
RESULTS_DIR = os.path.join(CURRENT_FOLDER, "results")

if os.path.isdir(RESULTS_DIR):
    shutil.rmtree(RESULTS_DIR)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.chdir(CURRENT_FOLDER)

WRITER = io.VTKFile(MPI_COMM, f"{RESULTS_DIR}/result.pvd", "w")

alpha = 1  # m^2/s, https://en.wikipedia.org/wiki/Thermal_diffusivity
k = 100  # kg * m / s^3 / K, https://en.wikipedia.org/wiki/Thermal_conductivity
T_hot = 310
dt_out = 0.2  # same as fluid

# ------- #
# MESHING #
# ------- #

nx = 100
ny = 25
nz = 1

y_top = 0
y_bottom = y_top - 0.25
x_left = 0
x_right = x_left + 1

domain = create_box(
    MPI_COMM,
    [np.array([x_left, y_bottom, 0]), np.array([x_right, y_top, 0.4])],
    [nx, ny, nz],
    cell_type=CellType.hexahedron,
)
dim = domain.topology.dim
V = fem.functionspace(domain, ("Lagrange", 2))
V_g = fem.functionspace(domain, ("Lagrange", 2, (3,)))
WRITER.write_mesh(domain)


def bottom_boundary_fn(x):
    return np.isclose(x[1], y_bottom)


def neumann_boundary(x):
    return np.isclose(x[1], y_top)


def determine_heat_flux(V_g, u_g):
    w = ufl.TrialFunction(V_g)
    v = ufl.TestFunction(V_g)

    a = ufl.inner(w, v) * ufl.dx
    L = ufl.inner(-k * ufl.grad(u_g), v) * ufl.dx
    problem = LinearProblem(a, L)
    return problem.solve()


bottom_boundary = fem.locate_dofs_geometrical(V, bottom_boundary_fn)
coupling_boundary = locate_entities_boundary(domain, dim - 1, neumann_boundary)

bcs = [fem.dirichletbc(PETSc.ScalarType(T_hot), bottom_boundary, V)]

# ------------ #
# PRECICE INIT #
# ------------ #
participant = Adapter(PARTICIPANT_CONFIG, domain)
participant.initialize(V, coupling_boundary)
dt = participant.dt

# Define solution variable, and interpolate initial solution for visualization in Paraview
uh = fem.Function(V)
u_n = fem.Function(V, name="T")

u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)
F = u * v / dt * ufl.dx + alpha * ufl.dot(ufl.grad(u), ufl.grad(v)) * ufl.dx - u_n * v / dt * ufl.dx
a, L = ufl.lhs(F), ufl.rhs(F)

heat_problem = DiscreteLinearProblem(a=a, L=L, u=uh, bcs=bcs, point_dofs=participant.interface_dof)

t = 0.0
while participant.is_coupling_ongoing():
    if participant.requires_writing_checkpoint():  # write checkpoint
        participant.store_checkpoint((u_n, t))

    read_data = participant.read_data(dt)
    heat_problem.solve()
    u_flux = determine_heat_flux(V_g, uh)
    u_flux.name = "Heat Flux"
    uh.x.scatter_forward()
    u_flux.x.scatter_forward()

    # Only exchange y component of flux
    participant.write_data(u_flux.sub(1))

    # Call to advance coupling, also returns the optimum time step value
    participant.advance(dt)

    if participant.requires_reading_checkpoint():
        u_cp, t_cp = participant.retrieve_checkpoint()
        u_n.x.array[:] = u_cp.x.array
        t = t_cp
    else:
        u_n.x.array[:] = uh.x.array
        t += dt

    if participant.is_time_window_complete():
        tol = 10e-5  # we need some tolerance, since otherwise output might be skipped.
        if abs((t + tol) % dt_out) < 2 * tol:
            WRITER.write_function([u_n, u_flux], t)
            WRITER.close()

WRITER.close()
participant.finalize()
