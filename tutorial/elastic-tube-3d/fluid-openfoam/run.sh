#!/bin/bash
module load preCICE_OpenFOAM
. $FOAM_BASH
. $WM_PROJECT_DIR/bin/tools/RunFunctions

./clean.sh
touch fluid.foam

#-------------#
#   Meshing   #
#-------------#
runApplication -o gmshToFoam   fluid.msh
runApplication -o renumberMesh -overwrite
runApplication -o checkMesh    -constant


# ------------#
#   Running   #
# ------------#
restore0Dir
runApplication -o $(getApplication)
find . -type d -regex '.*/[0-9]+\.[0-9]+' ! -exec test -e "{}/cellDisplacement" \; -exec rm -rf {} +