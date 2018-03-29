filenames="LH_paths HL_paths HZ_paths FF_paths FB_paths lateral_paths"
for file in $filenames
do
        latex ${file}.tex
        dvipdf ${file}.dvi
        pdftops -eps ${file}.pdf
	rm ${file}.aux
	rm ${file}.log
	rm ${file}.dvi
	rm ${file}.pdf
done
