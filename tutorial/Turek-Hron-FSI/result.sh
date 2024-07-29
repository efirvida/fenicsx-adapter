#! /bin/bash
module load gnuplot
rm -rf *.png
gnuplot -p << EOF
set grid
set title 'Displacement of the Flap Tip'
set xlabel 'Time [s]'
set ylabel 'Y-Displacement [m]'
set term pngcairo enhanced size 1920,1080 lw 5
set output "y-displacement.png"
plot "solid-calculix/precice-Solid-watchpoint-Flap-Tip.log" using 1:5 with lines title "OpenFOAM-CalculiX" lc rgb "red" lw 1,\
     "solid-fenicsx/precice-Solid-watchpoint-Flap-Tip.log" using 1:5 with lines title "OpenFOAM-DOLFINx" lc rgb "blue" lw 1
     #"referenceY-disp" title "Turek-Hron FSI benchmark FSI2" lc rgb "black"
EOF

gnuplot -p << EOF
set grid
set title 'Displacement of the Flap Tip'
set xlabel 'Time [s]'
set ylabel 'Y-Forces [m]'
set term pngcairo enhanced size 1920,1080 lw 5
set output "y-forces.png"
plot "solid-calculix/precice-Solid-watchpoint-Flap-Tip.log" using 1:7 with lines title "OpenFOAM-CalculiX" lc rgb "red" lw 1,\
     "solid-fenicsx/precice-Solid-watchpoint-Flap-Tip.log" using 1:7 with lines title "OpenFOAM-DOLFINx" lc rgb "blue" lw 1
EOF

