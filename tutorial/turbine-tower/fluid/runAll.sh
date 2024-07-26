#!/bin/bash
#$ -S /bin/sh -cwd
#$ -q gold
#$ -pe mpi.fillup 160
#$ -N Tower
###################################
module load preCICE_OpenFOAM
. $FOAM_BASH
. $WM_PROJECT_DIR/bin/tools/RunFunctions


# ./clean.sh
# touch fluid.foam

#-------------#
#   Meshing   #
#-------------#
# runApplication blockMesh
# runApplication surfaceFeatureExtract
# runApplication snappyHexMesh -overwrite
# runApplication renumberMesh -overwrite
# runApplication checkMesh -constant

# -------------#
#   Running   #
# -------------#

runApplication -o decomposePar
restore0Dir -processor
runParallel -o setFields
runParallel -o $(getApplication)
runApplication -o reconstructPar -newTimes