#!/bin/bash
#$ -S /bin/sh -cwd
#$ -q gold
#$ -pe mpi.fillup 240
#$ -N Tower
###################################
module load OpenFOAM/v2312-foss-2023a
. $FOAM_BASH
. $WM_PROJECT_DIR/bin/tools/RunFunctions


#./clean.sh
#touch fluid.foam

#-------------#
#   Meshing   #
#-------------#
#runApplication blockMesh
#runApplication surfaceFeatureExtract
#runApplication snappyHexMesh -overwrite
#runApplication renumberMesh -overwrite
#runApplication createPatch -overwrite
#runApplication checkMesh -constant

# -------------#
#   Running   #
# -------------#

runApplication -o decomposePar
restore0Dir    -processor
#runParallel    -o setFields
runParallel    -o $(getApplication)
#runApplication    -o setFields
#runApplication    -o $(getApplication)
runApplication -o reconstructPar -newTimes

