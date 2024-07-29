import os
import gmsh
from dolfinx import fem, io
from mpi4py import MPI


# --------- #
# CONSTANTS #
# --------- #

MPI_COMM = MPI.COMM_WORLD
CURRENT_FOLDER = os.path.dirname(__file__)
os.chdir(CURRENT_FOLDER)

height = 150  # m
diameter = 6.5  # m
thickness = 0.055 # 55mm

# ------- #
# MESHING #
# ------- #
FIXED_TAG = 1000
BEAM_SURFACE_TAG = 2000
VOLUME_TAG = 4000
ELEMENTS_ORDER = 2
VECTOR_SPACE_DEGREE = 2


def gmsh_tower(h: float, d: float) -> gmsh.model:
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    model = gmsh.model()

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
    gmsh.write("mesh.msh")
    return model

model = gmsh_tower(h=height, d=diameter)

domain, cell_markers, facet_markers = io.gmshio.model_to_mesh(model, MPI_COMM, rank=0)
#domain, cell_markers, facet_markers = io.gmshio.read_from_msh("mesh.msh", MPI_COMM, 0, gdim=3)

# -------------- #
# Function Space #
# -------------- #
dim = domain.geometry.dim
V = fem.functionspace(domain, ("Lagrange", VECTOR_SPACE_DEGREE, (dim,)))
u = fem.Function(V, name="Displacement")

bs = V.dofmap.index_map_bs
num_dofs_local = V.dofmap.index_map.size_local
print(f"Rank {domain.comm.rank}, Block size {bs} Num local dofs {num_dofs_local*bs}")
