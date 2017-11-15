# Form to set directories
form Directories
	comment Enter directory with TextGrids:
	sentence directory /Users/Laura/Desktop/Hmong_Grids-old/
#	sentence directory /Users/Laura/Desktop/Hmong-test/
endform

clearinfo

#-------------------------------------------------------------------------#

# Find number of text grids in the specified directory
Create Strings as file list... grids 'directory$'*.TextGrid
numberoffiles = Get number of strings

# Print initial report
echo 'numberoffiles' TextGrid files found in directory 'directory$'.'newline$'

#-------------------------------------------------------------------------#

# Go through each file
for file to numberoffiles
	select Strings grids
	gridfile$ = Get string... file
	Read from file... 'directory$''gridfile$'
	gridname$ = selected$ ("TextGrid", 1)

	# Rename first tier from "phonation" to "vowel"
	# Duplicate first tier, put in second position with label "phonation"
	Set tier name... 1 vowel
	Duplicate tier... 1 2 phonation

	select TextGrid 'gridname$'
	Save as text file... 'directory$''gridfile$'

	Remove
endfor

select Strings grids
Remove
echo Done.