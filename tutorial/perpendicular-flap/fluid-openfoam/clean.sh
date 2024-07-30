#!/bin/bash
module load OpenFOAM
. $FOAM_BASH
cd "${0%/*}" || exit                                # Run from this directory
. ${WM_PROJECT_DIR:?}/bin/tools/CleanFunctions      # Tutorial clean functions
#------------------------------------------------------------------------------

rm -rf precice* *.log
cleanCase0

#------------------------------------------------------------------------------
