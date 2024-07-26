import precice
import os
import json
import sys
import numpy as np
import dolfinx as dfx
import logging
from turbine.materials import Material
from typing import NamedTuple

logger = logging.getLogger("precice")


class Dimensions(NamedTuple):
    interface: int
    mesh: int


class Config:
    """Handles reading of config. parameters of the fenicsxadapter based on JSON
    configuration file. Initializer calls read_json() method. Instance attributes
    can be accessed by provided getter functions.
    """

    def __init__(self, config_path: str):
        """Initialize the Config object.

        Parameters
        ----------
        config_path : str
            Path to the JSON configuration file.
        """
        self._config_file = config_path
        self._precice_config = None
        self._participant = None
        self._mesh = None
        self._read_data = None
        self._write_data = None
        self._patch_tags = []

        self.read_json(config_path)

    def read_json(self, config_path: str):
        """Reads JSON adapter configuration file and saves the data to the respective instance attributes.

        Parameters
        ----------
        config_path : str
            Name of the JSON configuration file.
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
        """Returns the preCICE configuration file path."""
        return self._precice_config

    @property
    def participant(self):
        """Returns the participant name."""
        return self._participant

    @property
    def mesh(self):
        """Returns the mesh name."""
        return self._mesh

    @property
    def patch_tags(self):
        """Returns the patch tags."""
        return self._patch_tags

    @property
    def read_data(self):
        """Returns the data to be read."""
        return self._read_data

    @property
    def write_data(self):
        """Returns the data to be written."""
        return self._write_data


class SolverState:
    """Stores the state of the solver, including displacement, velocity, acceleration, and time."""

    def __init__(self, u, v, a, t):
        """Initialize the SolverState object.

        Parameters
        ----------
        u : dfx.fem.Function
            Displacement function.
        v : dfx.fem.Function
            Velocity function.
        a : dfx.fem.Function
            Acceleration function.
        t : float
            Time.
        """
        self.u = u
        self.v = v
        self.a = a
        self.t = t

    def get_state(self):
        """Returns the state of the solver.

        Returns
        -------
        tuple
            A tuple containing displacement, velocity, acceleration, and time.
        """
        return self.u, self.v, self.a, self.t


class Adapter:
    """Adapter for coupling FEniCS with preCICE."""

    def __init__(self, mpi_comm, config_path: str, mesh) -> None:
        """Initialize the Adapter object.

        Parameters
        ----------
        mpi_comm : MPI.Comm
            MPI communicator.
        config_path : str
            Path to the configuration file.
        mesh : dfx.mesh.Mesh
            Computational mesh.
        """
        self._config = Config(config_path)

        self._comm = mpi_comm
        self._domain = mesh
        self._topology = self._domain.topology
        self._checkpoint = None

        self._function_space = None

        self._interface = precice.Participant(
            self._config.participant,
            self._config.precice_config,
            self._comm.Get_rank(),
            self._comm.Get_size(),
        )

    def interpolation_points_in_vector_space(self):
        """Determine interpolation points in the vector space.

        Returns
        -------
        tuple
            Unrolled degrees of freedom and their coordinates.
        """
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
        """Initialize the Adapter with function space and facesets.

        Parameters
        ----------
        function_space : dfx.fem.FunctionSpace
            Function space for the problem.
        facesets : array_like
            Array of faceset tags.
        """
        self._function_space = function_space
        self._facesets = facesets

        self._interface_dof, self._interface_dof_coords = (
            self.interpolation_points_in_vector_space()
        )

        self._precice_vertex_ids = self._interface.set_mesh_vertices(
            self._config.mesh, self._interface_dof_coords[:, : self.dimensions.interface]
        )

        if self._interface.requires_initial_data():
            self._interface.write_data(
                self._config.mesh,
                self._config.write_data,
                self._precice_vertex_ids,
                np.zeros(self._interface_dof_coords.shape),
            )
        self._interface.initialize()

    @property
    def interface_dof(self):
        """Returns the interface degrees of freedom."""
        return self._interface_dof

    @property
    def interface_coordinates(self):
        """Returns the interface coordinates."""
        return self._interface_dof_coords

    @property
    def precice(self):
        """Returns the preCICE interface object."""
        return self._interface

    @property
    def dimension_mismatch(self):
        """Checks if there is a dimension mismatch between the mesh and the interface."""
        return self.dimensions.mesh != self.dimensions.interface

    @property
    def dimensions(self) -> Dimensions:
        """Returns the dimensions of the mesh and the interface."""
        return Dimensions(
            self._interface.get_mesh_dimensions(self._config.mesh), self._domain.geometry.dim
        )

    @property
    def dt(self):
        """Returns the maximum time step size allowed by preCICE."""
        return self._interface.get_max_time_step_size()

    def is_coupling_ongoing(self):
        """Checks if the coupling is ongoing.

        Returns
        -------
        bool
            True if the coupling is ongoing, False otherwise.
        """
        return self._interface.is_coupling_ongoing()

    def is_time_window_complete(self):
        """Checks if the time window is complete.

        Returns
        -------
        bool
            True if the time window is complete, False otherwise.
        """
        return self._interface.is_time_window_complete()

    def requires_reading_checkpoint(self):
        """Checks if reading a checkpoint is required.

        Returns
        -------
        bool
            True if reading a checkpoint is required, False otherwise.
        """
        return self._interface.requires_reading_checkpoint()

    def requires_writing_checkpoint(self):
        """Checks if writing a checkpoint is required.

        Returns
        -------
        bool
            True if writing a checkpoint is required, False otherwise.
        """
        return self._interface.requires_writing_checkpoint()

    def advance(self, dt: float):
        """Advances the simulation by a given time step.

        Parameters
        ----------
        dt : float
            Time step to advance.

        Returns
        -------
        float
            The new simulation time.
        """
        return self._interface.advance(dt)

    def finalize(self):
        """Finalizes the preCICE coupling.

        Returns
        -------
        None
        """
        return self._interface.finalize()

    def read_data(self, dt):
        """Reads data from preCICE.

        Parameters
        ----------
        dt : float
            Time step size.

        Returns
        -------
        np.ndarray
            Data read from preCICE.
        """
        mesh_name = self._config.mesh
        data_name = self._config.read_data
        read_data = self._interface.read_data(mesh_name, data_name, self._precice_vertex_ids, dt)
        return read_data

    def write_data(self, write_function):
        """Writes data to preCICE.

        Parameters
        ----------
        write_function : dfx.fem.Function
            Function containing the data to write.
        """
        mesh_name = self._config.mesh
        write_data_name = self._config.write_data
        write_data = write_function.vector[self.interface_dof]
        self._interface.write_data(mesh_name, write_data_name, self._precice_vertex_ids, write_data)

    def store_checkpoint(self, u, v, a, t):
        """Stores the current state as a checkpoint.

        Parameters
        ----------
        u : dfx.fem.Function
            Displacement function.
        v : dfx.fem.Function
            Velocity function.
        a : dfx.fem.Function
            Acceleration function.
        t : float
            Time.
        """
        self._checkpoint = SolverState(u.copy(), v.copy(), a.copy(), t)

    def retrieve_checkpoint(self):
        """Retrieves the stored checkpoint state.

        Returns
        -------
        tuple
            The stored checkpoint state (u, v, a, t).
        """
        assert not self.is_time_window_complete()
        return self._checkpoint.get_state()
