# Set terminal to EPS and output file name
set terminal postscript eps enhanced color solid font 'Helvetica,25' size 8,6.0
set output 'SM_throughput_product.eps'
#set boxwidth 0.9 absolute
set style fill solid 1.00 border lt -1
# Set the legend (key) font
set key inside top right vertical  font "Arial,35"  # Adjust the font size as needed
# Set the labels (no title)
set xlabel "Batch" font "Arial,45 italic bold"
set bmargin 4 
set ylabel "Compute Throught[%] (SM)" font "Arial,45 italic bold" offset -1
set lmargin 10
# Define bar style and width
set style data linespoints
#set style histogram cluster gap 1
set style fill solid 1.0 border -1
#set boxwidth 0.9

# Set y-axis grid line
#set grid ytics

# Set xtics (x-axis labels) from the first column (no rotation)
#set xtics nomirror
set xtics font "Arial,45 italic bold"
set ytics font "Arial,45 italic bold"

# Remove extra space between Y-axis and first bar by setting x range
#set xrange [-0.1:3.7]  # Adjust this range to remove extra space on the left side
#set xrange [-0.6:2.7]  # Adjust this range to remove extra space on the left side
#set xrange [0:100]
set yrange [10:60]
set ytics 10
# Plot the data from a text file
# Skip the first 2 header lines
plot 'SM_throughput_product.dat' using 0:1 every ::2 with linespoints title 'CLING Layer 1' lt 1 lw 2 lc rgb 'blue' pt 2, \
     '' using 0:2 every ::2 with linespoints title 'CLING Layer 2' lt 1 lw 2 lc rgb 'blue' pt 5, \
     '' using 0:3 every ::2 with linespoints title 'CLING Layer 3' lt 1 lw 2 lc rgb 'blue' pt 1, \
     '' using 0:4 every ::2 with linespoints title 'GraphSAGE Layer 1' lt 1 lw 2 lc rgb 'purple' pt 2, \
     '' using 0:5 every ::2 with linespoints title 'GraphSAGE Layer 2' lt 1 lw 2 lc rgb 'purple' pt 5, \
     '' using 0:6 every ::2 with linespoints title 'GraphSAGE Layer 3' lt 1 lw 2 lc rgb 'purple' pt 1

#plot 'accuracy_line_product.txt' using 0:1 with linespoints title 'CLING' lw 4 lc rgb 'blue' pt 12, \
#     '' using 0:2 with linespoints title 'GraphSAGE' lw 4 lc rgb 'red' pt 12
# Close the output file
set output


