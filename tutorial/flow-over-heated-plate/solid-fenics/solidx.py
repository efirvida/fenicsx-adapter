import matplotlib as mpl
import ufl
import numpy as np
import os

from petsc4py import PETSc
from mpi4py import MPI

from dolfinx import fem, io
from dolfinx.mesh import CellType, create_box,locate_entities_boundary
from dolfinx.fem.petsc import assemble_vector, assemble_matrix, create_vector, apply_lifting, set_bc
from fenicsxprecice import Adapter, DiscreteLinearProblem


MPI_COMM = MPI.COMM_WORLD
CURRENT_FOLDER = os.path.dirname(__file__)
PARTICIPANT_CONFIG = os.path.join(CURRENT_FOLDER, "precice-adapter-config.json")
RESULTS_DIR = os.path.join(CURRENT_FOLDER, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.chdir(CURRENT_FOLDER)

WRITER = io.VTKFile(MPI_COMM, f"{RESULTS_DIR}/result.pvd", "w")


alpha = 1  # m^2/s, https://en.wikipedia.org/wiki/Thermal_diffusivity
k     = 100  # kg * m / s^3 / K, https://en.wikipedia.org/wiki/Thermal_conductivity
u_D   = 310
# ------- #
# MESHING #
# ------- #

nx = 100
ny = 25
nz = 1

y_top = 0
y_bottom = y_top - .25
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
V_g = fem.functionspace(domain, ("Lagrange", 2))
V_flux_y = V_g.sub(1)

u_n = fem.Function(V, name="T")
u_D_Function = fem.Function(V) 

# ------------------- #
# Boundary conditions #
# ------------------- #
u_D = fem.Constant(domain, u_D)
k = fem.Constant(domain, k)
alpha = fem.Constant(domain, alpha)
u_n.interpolate(u_D)


tol = 1e-14


def bottom_boundary_fn(x):
    return np.isclose(x[1], y_bottom)


def neumann_boundary(x):
    return np.isclose(x[1], y_top)



bottom_boundary = fem.locate_dofs_geometrical(V, bottom_boundary_fn)
coupling_boundary = locate_entities_boundary(domain, dim - 1, neumann_boundary)

bcs = [fem.dirichletbc(PETSc.ScalarType(0), bottom_boundary, V)]

# ------------ #
# PRECICE INIT #
# ------------ #
participant = Adapter(MPI_COMM, PARTICIPANT_CONFIG, domain)
participant.initialize(V, coupling_boundary)
np.savetxt("dofs.txt", participant.interface_coordinates, delimiter=",", header="x,y,z")
dt = participant.dt
WRITER.write_mesh(domain)

# -------------------------- #
# Define variational problem #
# -------------------------- #
u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
f = fem.Constant(domain, PETSc.ScalarType(0))
F = u * v / dt * ufl.dx + alpha * ufl.dot(ufl.grad(u), ufl.grad(v)) * ufl.dx - u_n * v / dt * ufl.dx
a, L = ufl.lhs(F), ufl.rhs(F)

t = 0

for i in range(num_steps):
    t += dt

    # Update the right hand side reusing the initial vector
    with b.localForm() as loc_b:
        loc_b.set(0)
    assemble_vector(b, linear_form)

    # Apply Dirichlet boundary condition to the vector
    apply_lifting(b, [bilinear_form], [[bc]])
    b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
    set_bc(b, [bc])

    # Solve linear problem
    solver.solve(b, uh.vector)
    uh.x.scatter_forward()

    # Update solution at previous time step (u_n)
    u_n.x.array[:] = uh.x.array

    # Write solution to file
    WRITER.write_function(uh, t)
    # Update plot
WRITER.close()
