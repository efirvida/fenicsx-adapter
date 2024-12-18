#!/bin/sh
#$ -S /bin/sh -cwd
#$ -q silver
#$ -pe mpi.fillup 120
#$ -N Rigid-Blade

module load OpenFOAM/10-foss-2023a 

. $FOAM_BASH

runParallel $(getApplication)
runApplication reconstructPar
rm -rf processor*

. ../../tools/openfoam-remove-empty-dirs.sh && openfoam_remove_empty_dirs
