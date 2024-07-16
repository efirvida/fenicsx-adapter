import precice
import os
import json
import sys
import numpy as np
import dolfinx as dfx
from petsc4py import PETSc
import logging
from turbine.materials import Material
from typing import NamedTuple


logger = logging.getLogger("precice")


class Dimensions(NamedTuple):
    interface: int
    mesh: str


class Config:
    """
    Handles reading of config. parameters of the fenicsxadapter based on JSON
    configuration file. Initializer calls read_json() method. Instance attributes
    can be accessed by provided getter functions.
    """

    def __init__(self, config_path: str):
        self._config_file = config_path
        self._precice_config = None
        self._participant = None
        self._mesh = None
        self._read_data = None
        self._write_data = None
        self._patch_tags = []

        self.read_json(config_path)

    def read_json(self, config_path: str):
        """
        Reads JSON adapter configuration file and saves the data to the respective instance attributes.

        Parameters
        ----------
        config_path : string
            Name of the JSON configuration file
        """
        read_file = open(config_path, "r")
        data = json.load(read_file)
        folder = os.path.dirname(
            os.path.join(os.getcwd(), os.path.dirname(sys.argv[0]), config_path)
        )
        self._precice_config = os.path.join(folder, data["precice_config"])
        self._participant = data["participant"]
        self._mesh = data["interface"]["mesh"]

        try:
            self._write_data = data["interface"]["write_data"]
        except KeyError:
            # not required for one-way coupling, if this participant reads data
            self._write_data = None

        try:
            self._read_data = data["interface"]["read_data"]
        except KeyError:
            # not required for one-way coupling, if this participant writes data
            self._read_data = None

        read_file.close()

    @property
    def precice_config(self):
        return self._precice_config

    @property
    def participant(self):
        return self._participant

    @property
    def mesh(self):
        return self._mesh

    @property
    def patch_tags(self):
        return self._patch_tags

    @property
    def read_data(self):
        return self._read_data

    @property
    def write_data(self):
        return self._write_data


class FunctionData:
    def __init__(self, vector_space, interface_dof):
        self.V = vector_space
        self._dim = self.V.mesh.geometry.dim
        self._v_space_coords = self.V.tabulate_dof_coordinates()[:, : self._dim]
        self._interface_dof = interface_dof
        self._vector_size = len(self._v_space_coords.flatten())

        self._vector = PETSc.Vec().createMPI(self._vector_size, comm=PETSc.COMM_WORLD)
        self._vector.setFromOptions()
        self._vector.setUp()
        self.reset()

    def reset(self):
        self._vector.set(0.0)
        self._vector.assemble()

    def setValues(self, values):
        self.reset()
        for i, dof in enumerate(self._interface_dof):
            for j in range(len(dof)):
                self._vector[dof[j]] = values[i, j]
        self._vector.assemble()

    def __str__(self):
        if self._dim == 3:
            Fx, Fy, Fz = self.array.sum(axis=0)
            return f"Sum Fx: {Fx:.2e}, Sum Fy: {Fy:.2e}, Sum Fz:{Fz:.2e}"
        if self._dim == 2:
            Fx, Fy = self.array.sum(axis=0)
            return f"Sum Fx: {Fx:.2f}, Sum Fy: {Fy:.2f}"

    @property
    def array(self):
        return self._vector.getArray().reshape(-1, self._dim)

    @property
    def vector(self):
        self._vector.assemble()
        return self._vector


class SolverState:
    def __init__(self, u, v, a, t):
        self.u = u
        self.v = v
        self.a = a
        self.t = t

    def get_state(self):
        return self.u, self.v, self.a, self.t

    def __str__(self):
        u, v, a, t = self.get_state()
        return f"u={u}, v={v}, a={a}, t={t}"


class Adapter:
    def __init__(self, mpi_comm, config_path: str, mesh, material_properties: Material) -> None:
        self._config = Config(config_path)

        self._comm = mpi_comm
        self._domain = mesh
        self._topology = self._domain.topology
        self._checkpoint = None
        self._material_properties = material_properties

        self._function_space = None

        self._interface = precice.Participant(
            self._config.participant,
            self._config.precice_config,
            self._comm.Get_rank(),
            self._comm.Get_size(),
        )

    def interpolation_points_in_vector_space(self):
        V = self._function_space
        bs = V.dofmap.bs

        fs_coords = V.tabulate_dof_coordinates()
        fs_coords = fs_coords[:, : self.dimensions.mesh]
        boundary_dofs = dfx.fem.locate_dofs_topological(V, self.dimensions.mesh - 1, self._facesets)
        unrolled_dofs = np.empty(len(boundary_dofs) * bs, dtype=np.int32)

        for i, dof in enumerate(boundary_dofs):
            for b in range(bs):
                unrolled_dofs[i * bs + b] = dof * bs + b

        return unrolled_dofs.reshape(-1, bs), fs_coords[boundary_dofs]

    def initialize(self, function_space, facesets):
        self._function_space = function_space
        self._facesets = facesets
        self._facesets_tags = dfx.mesh.meshtags(
            self._domain, self.dimensions.mesh - 1, np.sort(facesets), 1
        )

        self._interface_dof, self._interface_dof_coords = (
            self.interpolation_points_in_vector_space()
        )

        self._precice_vertex_ids = self._interface.set_mesh_vertices(
            self._config.mesh, self._interface_dof_coords[:, : self.dimensions.interface]
        )

        self._function_data = FunctionData(self._function_space, self._interface_dof)

        if self._interface.requires_initial_data():
            self._interface.write_data(
                self._config.mesh,
                self._config.write_data,
                self._precice_vertex_ids,
                np.zeros(self._interface_dof_coords.shape),
            )
        self._interface.initialize()
        return self._function_data

    @property
    def interface_dof(self):
        return self._interface_dof

    @property
    def interface_coordinates(self):
        return self._interface_dof_coords

    @property
    def precice(self):
        return self._interface

    @property
    def dimension_mismatch(self):
        return self.dimensions.mesh != self.dimensions.interface

    @property
    def dimensions(self) -> Dimensions:
        return Dimensions(
            self._interface.get_mesh_dimensions(self._config.mesh), self._domain.geometry.dim
        )

    @property
    def dt(self):
        return self._interface.get_max_time_step_size()

    @property
    def critical_dt(self):
        num_cells = self._domain.topology.index_map(self.dimensions.mesh - 1).size_local
        min_edge_length = min(self._domain.h(self.dimensions.mesh - 1, np.arange(num_cells)))
        c = np.sqrt(
            self._material_properties.E
            * (1 - self._material_properties.nu)
            / (
                self._material_properties.rho
                * (1 + self._material_properties.nu)
                * (1 - 2 * self._material_properties.nu)
            )
        )
        return min_edge_length / c

    def is_coupling_ongoing(self):
        return self._interface.is_coupling_ongoing()

    def is_time_window_complete(self):
        return self._interface.is_time_window_complete()

    def requires_reading_checkpoint(self):
        return self._interface.requires_reading_checkpoint()

    def requires_writing_checkpoint(self):
        return self._interface.requires_writing_checkpoint()

    def advance(self, dt: float):
        return self._interface.advance(dt)

    def finalize(self):
        return self._interface.finalize()

    def read_data(self, dt):
        mesh_name = self._config.mesh
        data_name = self._config.read_data
        read_data = self._interface.read_data(mesh_name, data_name, self._precice_vertex_ids, dt)
        return read_data

    def write_data(self, write_function):
        mesh_name = self._config.mesh
        write_data_name = self._config.write_data
        write_data = []
        for dof in self.interface_dof:
            if self.dimensions.interface == 2:
                ux, uy = write_function.vector[dof[0]], write_function.vector[dof[1]]
                write_data.append([ux, uy])
            if self.dimensions.interface == 3:
                ux, uy, uz = (
                    write_function.vector[dof[0]],
                    write_function.vector[dof[1]],
                    write_function.vector[dof[2]],
                )
                write_data.append([ux, uy])

        self._interface.write_data(mesh_name, write_data_name, self._precice_vertex_ids, write_data)

    def store_checkpoint(self, u, v, a, t):
        self._checkpoint = SolverState(u.copy(), v.copy(), a.copy(), t)

    def retrieve_checkpoint(self):
        assert not self.is_time_window_complete()
        return self._checkpoint.get_state()
