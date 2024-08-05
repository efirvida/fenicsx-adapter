import copy
from collections.abc import Sequence

import dolfinx as dfx


class SolverState:
    def __init__(self, states: Sequence):
        """Store a list of function as states for those
        iteration that not converge
        """
        states_cp = []
        for state in states:
            if isinstance(state, dfx.fem.Function):
                states_cp.append(state.copy())
            else:
                states_cp.append(copy.deepcopy(state))
        self.__state = states_cp

    def get_state(self):
        """Returns the state of the solver."""
        return self.__state
