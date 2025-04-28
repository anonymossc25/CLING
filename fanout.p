#set terminal pdfcairo size 10cm,7cm enhanced font 'Arial,12.5'
#set output 'products_fanout.pdf'
# Set terminal to EPS and output file name
set terminal postscript eps enhanced color solid font 'Helvetica,25' size 8,6.0
#set terminal postscript eps enhanced color solid font 'Helvetica,25'
#set output "fanout.eps"
set output outfile
#set boxwidth 0.9 absolute
set style fill solid 1.00 border lt -1
# Set the legend (key) font
set key inside top left vertical  font "Arial,65"  # Adjust the font size as needed
# Set the labels (no title)
set xlabel "Fanout" font "Arial,70 italic bold" offset 0,-2
set bmargin 7 
set ylabel "Total time (Seconds)" font "Arial,65 italic bold" offset -6
set lmargin 15
# Define bar style and width
set style data histograms
set style histogram cluster gap 1
set style fill solid 1.0 border -1
#set boxwidth 0.9

# Set y-axis grid line
set grid ytics

# Set xtics (x-axis labels) from the first column (no rotation)
#set xtics nomirror
set xtics font "Arial,65 italic bold" offset 0, -1
set ytics font "Arial,65 italic bold"

# Remove extra space between Y-axis and first bar by setting x range
#set xrange [-0.1:3.7]  # Adjust this range to remove extra space on the left side
#set xrange [-0.6:2.7]  # Adjust this range to remove extra space on the left side
set yrange [0:]
set ytics 100
# Plot the data from a text file
plot datafile using 2:xtic(1) title 'GraphSAGE' lt rgb "skyblue", \
     '' using 3 title 'CLING' lt rgb "orange"
     #'' using 4 title 'F:20' lt rgb "forest-green"
# Close the output file
set output

