# Set terminal to EPS
set terminal postscript eps enhanced color font 'Helvetica,12'
set output 'accuracy_plot.eps'

# Labels and title
set title "Accuracy over Epochs"
set xlabel "Epoch"
set ylabel "Accuracy"
set grid

# Set X range (epoch 0 to 99)
set xrange [0:99]
set yrange [0.0:1.0]

# Plot using row number as X (starting from 0)
plot 'accuracy_line_product.txt' using 0:1 with linespoints title 'CLING' lt 1 lc rgb 'blue' pt 1, \
     '' using 0:2 with linespoints title 'GraphSAGE' lt 0 lc rgb 'red' pt 1

