import os

import gmsh


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

    outer = model.occ.extrude([(2, outer_circle)], 0, 0, L, numElements=[40], recombine=True)
    inner = model.occ.extrude([(2, inner_circle)], 0, 0, L, numElements=[40], recombine=True)

    outer_volume = outer[1][1]
    inner_volume = inner[1][1]
    vol = model.occ.cut([(3, outer_volume)], [(3, inner_volume)])

    model.occ.synchronize()

    inlet_sf = model.addPhysicalGroup(2, [7], tag=INLET)
    outlet_sf = model.addPhysicalGroup(2, [8], tag=OUTLET)
    interface_sf = model.addPhysicalGroup(2, [5], tag=INTERFACE)
    vol = model.addPhysicalGroup(3, [1], tag=VOLUME_TAG)

    model.setPhysicalName(2, inlet_sf, "INLET")
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

    inner = model.occ.extrude([(2, circle)], 0, 0, L, numElements=[40], recombine=False)

    model.occ.synchronize()

    inlet_sf = model.addPhysicalGroup(2, [1], tag=INLET)
    outlet_sf = model.addPhysicalGroup(2, [3], tag=OUTLET)
    interface_sf = model.addPhysicalGroup(2, [2], tag=INTERFACE)
    vol = model.addPhysicalGroup(3, [1], tag=VOLUME_TAG)

    model.setPhysicalName(2, inlet_sf, "INLET")
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
