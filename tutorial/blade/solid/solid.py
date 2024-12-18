import os
from typing import NamedTuple
import time

# enforce 1 thread per core
nthreads = 1
os.environ["OMP_NUM_THREADS"] = str(nthreads)
os.environ["OPENBLAS_NUM_THREADS"] = str(nthreads)
os.environ["MKL_NUM_THREADS"] = str(nthreads)

import gmsh
from dolfinx import fem, io, default_scalar_type, mesh
import ufl
from mpi4py import MPI
from petsc4py import PETSc
import numpy as np
from dolfinx.graph import partitioner_scotch
from dolfinx.fem.petsc import LinearProblem
from fenicsxprecice import Adapter, DiscreteLinearProblem

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


material = Material("Fiberglass", 4.00e10, 0.3, 2.00e3)

dt = 0.01

BETA_ = 0.25
GAMMA_ = 0.5

# ------- #
# MESHING #
# ------- #
FIXED_TAG = 1003
BLADE_SURFACE_TAG = 1004
VOLUME_TAG = 1005
ELEMENTS_ORDER = 1
VECTOR_SPACE_DEGREE = 1


def gmsh_tower(msh_file: str):
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 1)
    model = gmsh.model()
    gmsh.merge(msh_file)
    model.mesh.setOrder(ELEMENTS_ORDER)
    return model


model = gmsh_tower(os.path.join(CURRENT_FOLDER, "solid.msh"))
partitioner = mesh.create_cell_partitioner(partitioner_scotch())

domain, cell_markers, facet_markers = io.gmshio.model_to_mesh(
    model,
    MPI_COMM,
    rank=0,
    partitioner=partitioner,
)

# -------------- #
# Function Space #
# -------------- #
dim = domain.geometry.dim
V = fem.functionspace(domain, ("Lagrange", VECTOR_SPACE_DEGREE, (dim,)))
u = fem.Function(V, name="Displacement")
f = fem.Function(V, name="Force")

bs = V.dofmap.index_map_bs
num_dofs_local = V.dofmap.index_map.size_local
print(f"Rank {domain.comm.rank}, Block size {bs} Num local dofs {num_dofs_local*bs}")

# --------------------#
# Boundary conditions #
# --------------------#
# Fixed BC
fixed_facets = facet_markers.find(FIXED_TAG)
fixed_surface_dofs = fem.locate_dofs_topological(V, dim - 1, fixed_facets)
coupling_boundary_markers = facet_markers.find(BLADE_SURFACE_TAG)

u_bc = fem.Function(V)
bcs = [fem.dirichletbc(u_bc, fixed_surface_dofs)]
# ------------ #
# PRECICE INIT #
# ------------ #
participant = Adapter(PARTICIPANT_CONFIG, domain)
participant.initialize(V, coupling_boundary_markers)
dt = participant.dt
np.savetxt("dofs_coords.csv", participant.interface_coordinates, delimiter=",", header="x,y,z")
np.savetxt("dofs_ids.csv", participant.interface_dof, delimiter=",", header="x,y,z", fmt="%d")

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

a = (1 / (beta * dt**2)) * (u - u_old - dt * v_old) - ((1 - 2 * beta) / (2 * beta)) * a_old
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
    a=a_form,
    L=L_form,
    u=u,
    bcs=bcs,
    point_dofs=participant.interface_dof,
    petsc_options={
        "ksp_type": "cg",  # Tipo de solver
        "pc_type": "hypre",  # Precondicionador
        "ksp_rtol": 1e-8,  # Tolerancia de convergencia
    },
)


# parameters for Time-Stepping
t = 0.0
dt_out = 0.02
time_window = 0
it = 0

iteration_times = []
total_start_time = time.time()
while participant.is_coupling_ongoing():
    print(f"=========================================================")
    print(f"Starting iteration {it} of time window {time_window}, time: {t:.2f} [s]")
    iteration_start_time = time.time()

    if participant.requires_writing_checkpoint():  # write checkpoint
        participant.store_checkpoint((u_old, v_old, a_old, t))

    read_data = participant.read_data(dt)
    f.vector[participant.interface_dof] = read_data
    f.vector.assemble()

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
        # np.savetxt(f"dofs_load_{t}.csv", read_data, delimiter=",", header="fx,fy,fz")
        t += dt

    if participant.is_time_window_complete():
        tol = 10e-5
        time_window += 1
        it = 0
        if abs((t + tol) % dt_out) < 2 * tol:
            WRITER.write_function((u, f), t)
            WRITER.close()
    else:
        it += 1

    iteration_end_time = time.time()
    iteration_times.append(iteration_end_time - iteration_start_time)
    average_time_per_iteration = sum(iteration_times) / len(iteration_times)
    print(f"Iteración elapsed time: {iteration_times[-1]:.2f} [s]")
    print(f"Average iteración elapsed time: {average_time_per_iteration:.2f} [s]")


total_end_time = time.time()

WRITER.close()

total_time = total_end_time - total_start_time
print(f"=========================================================")
print(f"Total simulation time: {total_time:.2f} [s]")
