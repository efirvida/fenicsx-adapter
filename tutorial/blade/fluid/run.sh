#!/bin/bash
module load preCICE_OpenFOAM
. $FOAM_BASH
. $WM_PROJECT_DIR/bin/tools/RunFunctions

./clean.sh
touch fluid.foam

#-------------#
#   Meshing   #
#-------------#
runApplication -o blockMesh
runApplication -o surfaceFeatureExtract
runApplication -o decomposePar
runParallel    -o snappyHexMesh -overwrite
runParallel    -o renumberMesh -noFields -overwrite
runParallel    -o checkMesh

# ------------#
#   Running   #
# ------------#

restore0Dir    -processor
runParallel -o $(getApplication)
find . -type d -regex '.*/[0-9]+\.[0-9]+' ! -exec test -e "{}/cellDisplacement" \; -exec rm -rf {} +