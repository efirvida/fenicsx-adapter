#!/bin/bash
module load OpenFOAM
cd "${0%/*}" || exit                                # Run from this directory
. ${WM_PROJECT_DIR:?}/bin/tools/CleanFunctions      # Tutorial clean functions
#------------------------------------------------------------------------------

cleanCase0

#------------------------------------------------------------------------------
