# Stop on stressed vowels
# Extract phonation label
# Write to text file

# form to enter dirs
form Directories
	comment Enter directory with TextGrids:
	sentence Textgrid_directory /Users/Laura/Desktop/Dissertation/interrater-reliability/
	
	comment Enter name/loc of results file:
	sentence Results_file /Users/Laura/Desktop/Dissertation/interrater-reliability/results.txt
endform

# set tier numbers
phone_tier = 1
word_tier = 2
transcription_tier = 3
phonation_tier = 4

# initialize results file
results_header$ = "gridfile,vowel_label,word_label,phonation1'newline$'"

if fileReadable (results_file$)
	beginPause ("Warning")
		comment ("The file 'results_file$' already exists.")
	results_choice = endPause ("Append", "Overwrite", 1)
	if results_choice = 2
		filedelete 'results_file$'
		fileappend "'results_file$'" 'results_header$'
	endif
else
		fileappend "'results_file$'" 'results_header$'
endif

# list text grids
Create Strings as file list... gridlist 'textgrid_directory$'*.TextGrid
numberoffiles = Get number of strings

initial_report$ = "Beginning analysis of 'numberoffiles' TextGrids..."
echo 'initial_report$'

for ifile to numberoffiles
	select Strings gridlist
	gridfile$ = Get string... ifile
		# gridfile$ includes file extension, eg NWF222-6B-coarse.TextGrid
	Read from file... 'textgrid_directory$''gridfile$'
	gridname$ = selected$()
		# gridname$ is TextGrid filename excluding extension, eg NWF222-6B-coarse
	
	phone_intervals = Get number of intervals... phone_tier

	for phone_interval to phone_intervals
		select 'gridname$'
		vowel_label$ = Get label of interval... phone_tier phone_interval
		
		if right$(vowel_label$) = "1" or right$(vowel_label$) = "2"
			vowel_start = Get start point... phone_tier phone_interval
			vowel_end = Get end point... phone_tier phone_interval
			vowel_dur = (vowel_end - vowel_start)
			vowel_midpoint = vowel_start + (vowel_dur / 2)
			
			word_interval = Get interval at time... word_tier vowel_midpoint
			word_label$ = Get label of interval... word_tier word_interval

			phonation_interval = Get interval at time... phonation_tier vowel_midpoint
			phonation_label$ = Get label of interval... phonation_tier phonation_interval

			#phonation2_interval = Get interval at time... phonation2_tier vowel_midpoint
			#phonation2_label$ = Get label of interval... phonation2_tier phonation2_interval

			# make blank things NA
			if word_label$ = ""
				word_label$ = "NA"
			endif
			if phonation_label$ = ""
				phonation_label$ = "NA"
			endif
			
			# output
			results_line$ = "'gridfile$','vowel_label$','word_label$','phonation_label$''newline$'"
			fileappend "'results_file$'" 'results_line$'
		endif
	endfor
		select 'gridname$'
		Remove
endfor
	
# remove objects	
select Strings gridlist
Remove

echo Done. 
