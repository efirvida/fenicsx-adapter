#!/bin/bash
module load gnuplot
rm -rf *.png
gnuplot -p << EOF
  set grid
  set title 'x-displacement of the flap tip'
  set xlabel 'time [s]'
  set ylabel 'x-displacement [m]'
  set term pngcairo enhanced size 1920,1080 lw 5
  set output "flap-x-displacement-tip.png"
  plot  "solid-fenicsx/precice-Solid-watchpoint-Flap-Tip.log" using 1:4 with lines title "OpenFOAM-FenicsX", \
        "solid-calculix/precice-Solid-watchpoint-Flap-Tip.log" using 1:4 with lines title "OpenFOAM-CalculiX"
EOF

gnuplot -p << EOF
  set grid
  set title 'y-displacement of the flap tip'
  set xlabel 'time [s]'
  set ylabel 'y-displacement [m]'
  set term pngcairo enhanced size 1920,1080 lw 5
  set output "flap-y-displacement-tip.png"
  plot  "solid-fenicsx/precice-Solid-watchpoint-Flap-Tip.log" using 1:5 with lines title "OpenFOAM-FenicsX", \
        "solid-calculix/precice-Solid-watchpoint-Flap-Tip.log" using 1:5 with lines title "OpenFOAM-CalculiX"
EOF

gnuplot -p << EOF
  set grid
  set title 'x-forces of the flap tip'
  set xlabel 'time [s]'
  set ylabel 'x-Force [N]'
  set term pngcairo enhanced size 1920,1080 lw 5
  set output "flap-x-forces-tip.png"
  plot  "solid-fenicsx/precice-Solid-watchpoint-Flap-Tip.log" using 1:6 with lines title "OpenFOAM-FenicsX", \
        "solid-calculix/precice-Solid-watchpoint-Flap-Tip.log" using 1:6 with lines title "OpenFOAM-CalculiX"
EOF

gnuplot -p << EOF
  set grid
  set title 'y-forces of the flap tip'
  set xlabel 'time [s]'
  set ylabel 'y-Force [N]'
  set term pngcairo enhanced size 1920,1080 lw 5
  set output "flap-y-forces-tip.png"
  plot  "solid-fenicsx/precice-Solid-watchpoint-Flap-Tip.log" using 1:7 with lines title "OpenFOAM-FenicsX", \
        "solid-calculix/precice-Solid-watchpoint-Flap-Tip.log" using 1:7 with lines title "OpenFOAM-CalculiX"
EOF
