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


# Define temporal parameters
t = 0  # Start time
T = 1.0  # Final time
num_steps = 50
dt = T / num_steps  # time step size

# Define mesh
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

u_n = fem.Function(V)
u_n.name = "u_n"


# ------------------- #
# Boundary conditions #
# ------------------- #
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






# Create boundary condition
fdim = domain.topology.dim - 1
boundary_facets = locate_entities_boundary(
    domain, fdim, lambda x: np.full(x.shape[1], True, dtype=bool))
bc = fem.dirichletbc(PETSc.ScalarType(0), fem.locate_dofs_topological(V, fdim, boundary_facets), V)

WRITER.write_mesh(domain)

# Define solution variable, and interpolate initial solution for visualization in Paraview
uh = fem.Function(V)
uh.name = "uh"
uh.interpolate(initial_condition)
WRITER.write_function(uh, t)

u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
f = fem.Constant(domain, PETSc.ScalarType(0))
a = u * v * ufl.dx + dt * ufl.dot(ufl.grad(u), ufl.grad(v)) * ufl.dx
L = (u_n + dt * f) * v * ufl.dx

bilinear_form = fem.form(a)
linear_form = fem.form(L)

A = assemble_matrix(bilinear_form, bcs=[bc])
A.assemble()
b = create_vector(linear_form)

solver = PETSc.KSP().create(domain.comm)
solver.setOperators(A)
solver.setType(PETSc.KSP.Type.PREONLY)
solver.getPC().setType(PETSc.PC.Type.LU)

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
