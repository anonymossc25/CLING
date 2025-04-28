set terminal pdfcairo enhanced color font "Arial,12"
set output 'gcn_speedup.pdf'

set style data histogram
set style histogram cluster gap 1
set style fill solid border -1
set boxwidth 0.9

set title "GCN Speedup Across Datasets"
set ylabel "Speedup"
#set xtics rotate by -15
set grid ytics

set key outside top center horizontal
set key autotitle columnhead

plot 'varying_fanout.dat' using 2:xtic(1) title "F=10", \
     '' using 3 title "F=15", \
     '' using 4 title "F=20"


