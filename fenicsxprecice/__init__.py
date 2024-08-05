import warnings

try:
    from dolfinx import *
except ModuleNotFoundError:
    warnings.warn("No FEniCSx installation found on system. Please check whether it is found correctly. "
                  "The FEniCSx adapter might not work as expected.\n\n")

from . import _version
from .fenicsxprecice import Adapter

__version__ = _version.get_versions()['version']
