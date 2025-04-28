# Set terminal to EPS and output file name
set terminal postscript eps enhanced color solid font 'Helvetica,25' size 8,6.0
set output 'GMemory_Accesses_product.eps'
#set boxwidth 0.9 absolute
set style fill solid 1.00 border lt -1
# Set the legend (key) font
set key inside center right vertical font "Arial,45"  # Adjust the font size as needed
# Set the labels (no title)
set tmargin 2
set xlabel "Batch" font "Arial,45 italic bold"
set bmargin 4 
set ylabel "Global Memory Acessess" font "Arial,45 italic bold" offset -3
set lmargin 15
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
#set xrange [0:200]
set xtics 20
#set yrange [7000000:25000000]
#set ytics 90000000
#set logscale y
# Plot the data from a text file
plot 'GMemory_Accesses_product.dat' using 0:1 with linespoints title 'CLING' lw 4 lc rgb 'blue' pt 12, \
     '' using 0:2 with linespoints title 'GraphSAGE' lw 4 lc rgb 'purple' pt 12
# Close the output file
set output


