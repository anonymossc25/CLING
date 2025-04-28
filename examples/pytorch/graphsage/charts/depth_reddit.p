# Set terminal to EPS and output file name
set terminal postscript eps enhanced color solid font 'Helvetica,25' size 9.0,6.0
#set terminal postscript eps enhanced color font 'Helvetica,10'
set output 'depth_reddit.eps'
set boxwidth 0.9 absolute
set style fill solid 1.00 border lt -1
# Set the legend (key) font
set key inside top left vertical font "Arial,60"  # Adjust the font size as needed
# Set the labels (no title)
set tmargin 2
#set xlabel "Batch size" font "Arial,55 italic bold" offset 0,-3
set bmargin 4
set ylabel "Total time (Seconds)" font "Arial,55 italic bold" offset -5
set lmargin 16

# Define bar style and width
set style data histograms
set style histogram cluster gap 1
set style fill solid 1.0 border -1
set boxwidth 0.9

# Set y-axis grid line
set grid ytics

# Set xtics (x-axis labels) from the first column (no rotation)
set xtics nomirror
set xtics font "Arial,60 italic bold"
set ytics font "Arial,60 italic bold"
#set xtics rotate by -45
# Remove extra space between Y-axis and first bar by setting x range
#set xrange [-0.9:3.5]  # Adjust this range to remove extra space on the left side
#set xrange [-0.6:2.7]  # Adjust this range to remove extra space on the left side

# Plot the data from a text file
plot 'depth_reddit.txt' using 2:xtic(1) title 'GraphSAGE' lt rgb "skyblue", \
     '' using 3 title 'CLING' lt rgb "orange"
     #'' using 4 title 'F:20' lt rgb "forest-green"
# Close the output file
set output

