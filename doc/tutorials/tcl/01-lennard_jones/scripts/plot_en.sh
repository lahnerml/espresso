#! /bin/sh
echo "set term pngcairo size 800,400 enhanced font 'FreeSans,10'" > plot_en.gp
echo "set output '$2'" >> plot_en.gp
echo "set key outside bottom center horizontal" >> plot_en.gp
echo "plot '$1' u 1:2 t 'P' w line, '$1' u 1:3 t 'Ekin' w line, '$1' u 1:4 t 'Epot' w line, '$1' u 1:5 t 'T' w line, '$1' u 1:6 t 'Etot' w line" >> plot_en.gp
echo "set term wxt size 800,400" >> plot_en.gp
echo "replot" >> plot_en.gp
echo "pause -1" >> plot_en.gp
gnuplot plot_en.gp
