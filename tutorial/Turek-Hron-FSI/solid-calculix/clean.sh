#!/bin/sh
set -e -u
rm -fv ./*.cvg ./*.dat ./*.frd ./*.sta ./*.12d ./*.rout spooles.out dummy
rm -fv WarnNodeMissMultiStage.nam
rm -fv ./*.eig
rm -fv ./*.vtk
rm -fv *.log log.* *.json
rm -rf precice-exports precice-profiling