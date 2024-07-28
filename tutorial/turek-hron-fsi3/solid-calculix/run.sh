#!/bin/bash
module load preCICE_CalculiX
./clean.sh
ccx_preCICE -i flap -precice-participant Solid
