#!/bin/bash
module load preCICE_CalculiX
# export OMP_NUM_THREADS=4

./clean.sh
ccx_preCICE -i flap -precice-participant Solid
