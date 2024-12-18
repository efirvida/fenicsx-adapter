import os
from typing import NamedTuple
import time

nthreads = 1
os.environ["OMP_NUM_THREADS"] = str(nthreads)
os.environ["OPENBLAS_NUM_THREADS"] = str(nthreads)
os.environ["MKL_NUM_THREADS"] = str(nthreads)

import gmsh
import numpy as np
import ufl
from petsc4py import PETSc
from dolfinx import default_scalar_type, fem, io, mesh
from dolfinx.fem.petsc import LinearProblem
from dolfinx.graph import partitioner_scotch
from mpi4py import MPI


# --------- #
# CONSTANTS #
# --------- #

MPI_COMM = MPI.COMM_WORLD
CURRENT_FOLDER = os.path.dirname(__file__)
RESULTS_DIR = os.path.join(CURRENT_FOLDER, "results2")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.chdir(CURRENT_FOLDER)

WRITER = io.VTKFile(MPI_COMM, f"{RESULTS_DIR}/result.pvd", "w")


# -------------------#
# Problem properties #
# -------------------#
class Material(NamedTuple):
    name: str
    E: default_scalar_type
    nu: default_scalar_type


material = Material("Fiberglass", 4.00e10, 0.3)
f = ufl.as_vector([10, 0, 0])

height = 150  # m
diameter = 6.5  # m
thickness = 0.055  # 55mm

# ------- #
# MESHING #
# ------- #
FIXED_TAG = 1000
BEAM_SURFACE_TAG = 2000
VOLUME_TAG = 4000
ELEMENTS_ORDER = 1
VECTOR_SPACE_DEGREE = 1


def gmsh_tower(model, h: float, d: float, thickness: float) -> gmsh.model:
    model_name = "Tower"
    gmsh.model.add(model_name)

    model.setCurrent(model_name)

    # Recombine tetrahedra to hexahedra
    gmsh.option.setNumber("Mesh.RecombinationAlgorithm", 2)
    gmsh.option.setNumber("Mesh.RecombineAll", 2)
    gmsh.option.setNumber("Mesh.CharacteristicLengthFactor", 0.02)

    outer_circle = model.occ.addDisk(0, 0, 0, d / 2, d / 2)
    inner_circle = model.occ.addDisk(0, 0, 0, d / 2 - thickness, d / 2 - thickness)

    outer = model.occ.extrude([(2, outer_circle)], 0, 0, h, numElements=[1.5 * h], recombine=True)
    inner = model.occ.extrude([(2, inner_circle)], 0, 0, h, numElements=[1.5 * h], recombine=True)

    outer_volume = outer[1][1]
    inner_volume = inner[1][1]
    model.occ.cut([(3, outer_volume)], [(3, inner_volume)])

    model.occ.synchronize()

    fixed_sf = model.addPhysicalGroup(2, [7], tag=FIXED_TAG)
    tip_sf = model.addPhysicalGroup(2, [6, 8], tag=BEAM_SURFACE_TAG)
    vol = model.addPhysicalGroup(3, [1], tag=VOLUME_TAG)

    model.setPhysicalName(2, fixed_sf, "FIXED SURFACE")
    model.setPhysicalName(2, tip_sf, "LOAD SURFACE")
    model.setPhysicalName(3, vol, "Mesh volume")

    model.mesh.generate(3)
    model.mesh.optimize()
    gmsh.write("mesh.vtk")
    return model


gmsh.initialize()
gmsh.option.setNumber("General.Terminal", 0)
model = gmsh.model()

if MPI_COMM.rank == 0:
    model = gmsh_tower(model, h=height, d=diameter, thickness=thickness)

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

# --------------------#
# Boundary conditions #
# --------------------#
# Fixed BC
fixed_facets = facet_markers.find(FIXED_TAG)
fixed_surface_dofs = fem.locate_dofs_topological(V, dim - 1, fixed_facets)
load_surface_markers = facet_markers.find(BEAM_SURFACE_TAG)

u_bc = fem.Function(V)
bcs = [fem.dirichletbc(u_bc, fixed_surface_dofs)]


def load_dofs_in_vector_space(V, tags):
    bs = V.dofmap.bs
    fs_coords = V.tabulate_dof_coordinates()
    boundary_dofs = fem.locate_dofs_topological(V, V.mesh.geometry.dim - 1, tags)
    unrolled_dofs = np.empty(len(boundary_dofs) * bs, dtype=np.int32)

    index_map = V.dofmap.index_map
    size_local = index_map.size_local
    num_ghosts = index_map.num_ghosts

    print(f"Rank {MPI_COMM.rank}: boundary_dofs {boundary_dofs.shape}")
    print(f"Rank {MPI_COMM.rank}: size_local {size_local}")
    print(f"Rank {MPI_COMM.rank}: num_ghosts {num_ghosts}")
    owned_dofs = np.arange(size_local, dtype=np.int32)

    boundary_dofs = np.array([dof for dof in boundary_dofs if dof in owned_dofs])
    print(f"Rank {MPI_COMM.rank}: boundary_dofs_filtered {boundary_dofs.shape}")

    for i, dof in enumerate(boundary_dofs):
        for b in range(bs):
            unrolled_dofs[i * bs + b] = dof * bs + b

    return unrolled_dofs, fs_coords[boundary_dofs]


surface_dofs, surface_coords = load_dofs_in_vector_space(V, load_surface_markers)
np.savetxt(f"Rank_{MPI_COMM.rank}_dofs_coords.csv", surface_coords, delimiter=",", header="x,y,z")

# ------------------------ #
# linear elastic equations #
# ------------------------ #
E = fem.Constant(domain, material.E)
nu = fem.Constant(domain, material.nu)

lmbda = E * nu / (1 + nu) / (1 - 2 * nu)
mu = E / 2 / (1 + nu)


def epsilon(v):
    return ufl.sym(ufl.grad(v))


def sigma(v):
    return lmbda * ufl.tr(epsilon(v)) * ufl.Identity(dim) + 2 * mu * epsilon(v)


u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)
a = ufl.inner(sigma(u), epsilon(v)) * ufl.dx
L = ufl.dot(f, v) * ufl.dx


# ---------------- #
# PROBLEM SOLUTION #
# ---------------- #
print(f"RANK: {MPI_COMM.rank}: dofs: {surface_dofs.shape}")

start = time.time()
problem = LinearProblem(
    a=a,
    L=L,
    bcs=bcs,
    petsc_options={
        "ksp_type": "cg",
        "pc_type": "hypre",
        "ksp_rtol": 1e-8,
    },
)
uh = problem.solve()

WRITER.write_function(uh, 0)
with uh.vector.localForm() as b_loc:
    array = b_loc.array
print(f"Rank: {MPI_COMM.rank}, Max surface dofs: {surface_dofs.max()}")
WRITER.close()
end = time.time()
print(f"Rank: {MPI_COMM.rank}, Solve Time: {end-start} [s]")
