import os
from typing import NamedTuple

import gmsh
import numpy as np
import ufl
from dolfinx import fem, io, mesh
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx.fem.petsc import apply_lifting, assemble_matrix, assemble_vector, set_bc

from turbine.solvers.post import compute_reaction

# enforce 1 thread per core
nthreads = 1
os.environ["OMP_NUM_THREADS"] = str(nthreads)
os.environ["OPENBLAS_NUM_THREADS"] = str(nthreads)
os.environ["MKL_NUM_THREADS"] = str(nthreads)

comm = MPI.COMM_WORLD
CURRENT_FOLDER = os.path.dirname(__file__)
RESULTS_DIR = os.path.join(CURRENT_FOLDER, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.chdir(CURRENT_FOLDER)


# -------------------#
# Problem properties #
# -------------------#
class Material(NamedTuple):
    name: str
    E: PETSc.ScalarType
    nu: PETSc.ScalarType
    rho: PETSc.ScalarType


material = Material("Aluminum 6061", 68.9e9, 0.33, 2700)

# beam size
height = 150  # m
diameter = 6.5  # m

# loads
# nearest mesh dof point of load face center
LOAD_POINT = (4.9490529e-02, -1.4846031e-01, 1.5000000e02)
load = np.array([10.0, 0, 0])
# --------------------#


# ------- #
# MESHING #
# ------- #
FIXED_TAG = 1000
BEAM_SURFACE_TAG = 2000
TIP_SURFACE_TAG = 3000
VOLUME_TAG = 4000
ELEMENTS_ORDER = 2
VECTOR_SPACE_DEGREE = 2


def gmsh_tower(model: gmsh.model, name: str, h: float, d: float) -> gmsh.model:
    gmsh.clear()
    model.add(name)
    model.setCurrent(name)

    gmsh.option.setNumber("Mesh.CharacteristicLengthFactor", 0.05)
    gmsh.option.setNumber("Mesh.MshFileVersion", 2)

    circle = model.occ.addDisk(0, 0, 0, d / 2, d / 2)
    surface = model.occ.addPlaneSurface([circle])
    model.occ.synchronize()

    model.occ.extrude([(2, circle)], 0, 0, h, numElements=[h], recombine=True)
    model.occ.synchronize()

    model.addPhysicalGroup(2, [surface], tag=FIXED_TAG, name="FIXED_SURFACE")

    subdivision = [h]
    extrusion = model.occ.extrude([(2, surface)], 0, 0, h, subdivision)
    model.occ.synchronize()

    model.addPhysicalGroup(3, [extrusion[1][1]], tag=VOLUME_TAG, name="VOLUME")
    model.addPhysicalGroup(2, [extrusion[2][1]], tag=BEAM_SURFACE_TAG, name="BEAM_SURFACE")
    model.addPhysicalGroup(2, [extrusion[0][1]], tag=TIP_SURFACE_TAG, name="TIP_SURFACE")

    model.mesh.generate(1)
    model.mesh.generate(2)
    model.mesh.generate(3)
    model.mesh.setOrder(ELEMENTS_ORDER)
    model.mesh.optimize()
    return model


def create_mesh(comm: MPI.Comm, model: gmsh.model, name: str, filename: str, mode: str):
    msh, ct, ft = io.gmshio.model_to_mesh(model, comm, rank=0)
    msh.name = name
    ct.name = f"{msh.name}_cells"
    ft.name = f"{msh.name}_facets"
    with io.XDMFFile(msh.comm, filename, mode) as file:
        msh.topology.create_connectivity(2, 3)
        file.write_mesh(msh)
        file.write_meshtags(
            ct, msh.geometry, geometry_xpath=f"/Xdmf/Domain/Grid[@Name='{msh.name}']/Geometry"
        )
        file.write_meshtags(
            ft, msh.geometry, geometry_xpath=f"/Xdmf/Domain/Grid[@Name='{msh.name}']/Geometry"
        )
    return (msh, ct, ft)


gmsh.initialize()
gmsh.option.setNumber("General.Terminal", 0)

# Create model
model_name = "Tower"
model = gmsh.model()
model = gmsh_tower(model=model, name=model_name, h=height, d=diameter)
model.setCurrent(model_name)
mesh_file = f"mesh_rank_{MPI.COMM_WORLD.rank}.xdmf"
domain, cell_markers, facet_markers = create_mesh(
    comm=MPI.COMM_SELF,
    model=model,
    name=model_name,
    filename=mesh_file,
    mode="w",
)

# -------------- #
# Function Space #
# -------------- #
dim = domain.geometry.dim
V = fem.functionspace(domain, ("Lagrange", VECTOR_SPACE_DEGREE, (dim,)))

load_facets = facet_markers.find(TIP_SURFACE_TAG)
mt = mesh.meshtags(domain, 2, load_facets, 1)
metadata = {"quadrature_degree": 4}
ds = ufl.Measure("ds", subdomain_data=mt, subdomain_id=1, metadata=metadata)

E = fem.Constant(domain, float(material.E))
nu = fem.Constant(domain, float(material.nu))
rho = fem.Constant(domain, float(material.rho))

# --------------------#
# Boundary conditions #
# --------------------#
# Fixed BC
fixed_facets = facet_markers.find(FIXED_TAG)
fixed_surface_dofs = fem.locate_dofs_topological(V, 2, fixed_facets)
u_bc = fem.Function(V)
fixed_bc = fem.dirichletbc(u_bc, fixed_surface_dofs)

# -------------------------#
# linear elastic equations #
# -------------------------#

lmbda = E * nu / (1 + nu) / (1 - 2 * nu)
mu = E / 2 / (1 + nu)


def epsilon(u):
    return ufl.sym(ufl.grad(u))


def sigma(u):
    return lmbda * ufl.nabla_div(u) * ufl.Identity(len(u)) + 2 * mu * epsilon(u)


def point(x):
    return (
        np.isclose(x[0], LOAD_POINT[0])
        & np.isclose(x[1], LOAD_POINT[1])
        & np.isclose(x[2], LOAD_POINT[2])
    )


point_dof = fem.locate_dofs_geometrical(V, point)[0]
point_dofs = np.arange(point_dof * dim, (point_dof + 1) * dim)


du = ufl.TrialFunction(V)
u_ = ufl.TestFunction(V)
a = fem.form(ufl.inner(sigma(du), epsilon(u_)) * ufl.dx)

u = fem.Function(V, name="Displacement")

bcs = [fixed_bc]

A = assemble_matrix(a, bcs=bcs)
A.assemble()

b_func = fem.Function(V)
b_func.x.array[:] = 0
b = b_func.vector
with b.localForm() as b_local:
    b_local.set(0)

for i, j in enumerate(point_dofs.tolist()):
    b[j] = load[i]

b.assemble()
apply_lifting(b, [a], bcs=[bcs])
b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)  # type: ignore
set_bc(b, bcs)

solver = PETSc.KSP().create(domain.comm)  # type: ignore
solver.setFromOptions()
solver.setOperators(A)
solver.setType(PETSc.KSP.Type.CG)
pc = solver.getPC()
pc.setType(PETSc.PC.Type.GAMG)
solver.setTolerances(rtol=1e-5, atol=1e-11, max_it=300)

uh = fem.Function(V)

# Set a monitor, solve linear system, and display the solver
# configuration
solver.setConvergenceTest(lambda ksp, n, rnorm: print(f"Iteración {n}, Residuo {rnorm}"))
solver.solve(b, uh.vector)

# Scatter forward the solution vector to update ghost values
uh.x.scatter_forward()

R = diameter / 2
L = height
A = np.pi * R**2
Fx, Fy, Fz = load

print("====================================================================")
print("                           DEFORMATION                              ")
print("====================================================================")
Ix = np.pi / 4 * R**4
print(f"         Analytic: {(Fx * L**3) / (3 * material.E * Ix)}")
print(f"         Computed: {uh.x.array[point_dofs]}")
