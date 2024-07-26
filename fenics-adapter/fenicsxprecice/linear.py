from __future__ import annotations

import typing

from petsc4py import PETSc

import numpy as np

import ufl
from dolfinx.fem.petsc import (
    assemble_vector,
    assemble_matrix_mat,
    set_bc,
    apply_lifting,
    LinearProblem,
)
from dolfinx.fem.bcs import DirichletBC
from dolfinx.fem.function import Function as _Function
from numpy.typing import NDArray


Dofs = NDArray[np.int32]
DofsValues = NDArray[np.float64]


class DiscreteLinearProblem(LinearProblem):
    def __init__(
        self,
        a: ufl.Form,
        L: ufl.Form,
        bcs: list[DirichletBC] = [],
        u: typing.Optional[_Function] = None,
        point_dofs: typing.Optional[dict] = None,
        petsc_options: typing.Optional[dict] = None,
        form_compiler_options: typing.Optional[dict] = None,
        jit_options: typing.Optional[dict] = None,
    ):
        if point_dofs is not None:
            if not isinstance(type(point_dofs), np.ndarray):
                point_dofs = np.array([point_dofs], dtype="int32")
        self._point_dofs = point_dofs

        super().__init__(
            a, L, bcs, u, petsc_options, form_compiler_options, jit_options
        )

    def solve(self, values: typing.Optional[DofsValues] = None):
        """Solve the problem."""

        # Assemble lhs
        self._A.zeroEntries()
        assemble_matrix_mat(self._A, self._a, bcs=self.bcs)
        self._A.assemble()

        # Assemble rhs
        with self._b.localForm() as b_loc:
            b_loc.set(0)

        if self._point_dofs is not None and values is not None:
            # apply load in dofs
            self._b[self._point_dofs] = values
            self._b.assemble()

        assemble_vector(self._b, self._L)

        # Apply boundary conditions to the rhs
        apply_lifting(self._b, [self._a], bcs=[self.bcs])
        self._b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        set_bc(self._b, self.bcs)

        # Solve linear system and update ghost values in the solution
        self._solver.solve(self._b, self._x)
        self.u.x.scatter_forward()

        return self.u
