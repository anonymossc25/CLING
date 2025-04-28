
# Set terminal to EPS and output file name
set terminal postscript eps enhanced color solid font 'Helvetica,25' size 8,6.0
set output 'accuracy_plot_reddit.eps'
#set boxwidth 0.9 absolute
set style fill solid 1.00 border lt -1
# Set the legend (key) font
set key inside bottom right vertical  font "Arial,45"  # Adjust the font size as needed
# Set the labels (no title)
set xlabel "Epoch" font "Arial,45 italic bold"
set bmargin 4 
set ylabel "Accuracy" font "Arial,45 italic bold"
#set lmargin 15
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
set xrange [0:100]
set yrange [0.8:1.0]
set ytics 0.04
# Plot the data from a text file
plot 'accuracy_line_reddit.dat' using 0:1 every ::2 with linespoints title 'CLING' lw 1 pt 12, \
     '' using 0:2 every ::2 with linespoints title 'GraphSAGE' lw 1 pt 12, \
     '' using 0:3 every ::2 with linespoints title 'gSampler' lw 3 pt 12, \
     '' using 0:4 every ::2 with linespoints title 'LADIES' lw 1 pt 12
# Close the output file
set output


