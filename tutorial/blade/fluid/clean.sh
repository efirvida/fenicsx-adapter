#!/bin/bash
module load OpenFOAM/v2312-foss-2023a
. $FOAM_BASH
cd "${0%/*}" || exit                                # Run from this directory
. ${WM_PROJECT_DIR:?}/bin/tools/CleanFunctions      # Tutorial clean functions
#------------------------------------------------------------------------------

cleanCase0
rm -rf *.log precice-profiling

#------------------------------------------------------------------------------
