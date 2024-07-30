#!/usr/bin/env sh
module load gnuplot
 
gnuplot -p << EOF                                                               
	set grid                                                                        
	set title 'displacement at the middle of the tube'                                        
	set xlabel 'time [s]'                                                           
	set ylabel 'displacement [m]'
	set term pngcairo enhanced size 1920,1080 lw 5
	set output "displacement.png"
	plot "solid-calculix/precice-Solid-watchpoint-Tube-Midpoint.log" using 1:5 with lines title "OpenFOAM-CalculiX circumferential", \
	     "solid-calculix/precice-Solid-watchpoint-Tube-Midpoint.log" using 1:6 with lines title "OpenFOAM-CalculiX radial", \
	     "solid-calculix/precice-Solid-watchpoint-Tube-Midpoint.log" using 1:7 with lines title "OpenFOAM-CalculiX axial", \
	     "solid-fenicsx/precice-Solid-watchpoint-Tube-Midpoint.log" using 1:5 with lines title "OpenFOAM-FEniCS circumferential", \
	     "solid-fenicsx/precice-Solid-watchpoint-Tube-Midpoint.log" using 1:6 with lines title "OpenFOAM-FEniCS radial", \
	     "solid-fenicsx/precice-Solid-watchpoint-Tube-Midpoint.log" using 1:7 with lines title "OpenFOAM-FEniCS axial
EOF
