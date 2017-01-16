# Form to set directories
form Directories
	comment Enter directory with TextGrids:
	sentence Textgrid_directory /Users/Laura/Desktop/Dissertation/test-data/
	
	comment Enter name and location of results file:
	sentence Results_file /Users/Laura/Desktop/Dissertation/test-data/results.txt
endform

# Set tier numbers
phone_tier = 1
word_tier = 2
transcription_tier = 3
phonation_tier = 4

# Initialize results file
results_header$ = "gridfile	vowel_start	vowel_end	vowel_dur	vowel_label	word_label	phonation	jitter_ddp	jitter_loc	jitter_loc_abs	jitter_rap	jitter_ppq5	shimmer_loc	shimmer_local_dB	shimmer_apq3	shimmer_apq5	shimmer_apq11	shimmer_dda	hnr_mean	f1	f2	'newline$'"


# Check if the results file already exists
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

# Find number of text grids in the specified directory
Create Strings as file list... gridlist 'textgrid_directory$'*.TextGrid
numberoffiles = Get number of strings

# Print initial report
initial_report$ = "Beginning analysis of 'numberoffiles' TextGrids..."
echo 'initial_report$'

# Go through each file
for ifile to numberoffiles
	select Strings gridlist
	gridfile$ = Get string... ifile
		# gridfile$ includes file extension, eg NWF222-6B-coarse.TextGrid
	Read from file... 'textgrid_directory$''gridfile$'
	gridname$ = selected$()
		# gridname$ is TextGrid filename excluding extension, eg NWF222-6B-coarse
	#filename$ = gridname$ - "TextGrid"
	soundname$ = replace$ (gridname$, "TextGrid ", "", 0)
	filename$ = soundname$ + ".wav"
		# filename$ is the sound file including extension, eg NWF222.wav

	phone_intervals = Get number of intervals... phone_tier
	Read from file... 'textgrid_directory$''filename$'

	# Create Pitch, Point Process, and Harmonicity objects
	select Sound 'soundname$'
	To Pitch... 0 75 600
	select Sound 'soundname$'
	plus Pitch 'soundname$'
	To PointProcess (cc)
	select Sound 'soundname$'
	To Harmonicity (cc)... 0.01 75.0 0.1 1.0
	select Sound 'soundname$'
	To Formant (burg)... 0.0 5.0 5500.0 0.025 50.0
	# CHANGE THE ABOVE SETTINGS BY GENDER!

	for phone_interval to phone_intervals
		select 'gridname$'

		# Get vowel annotation
		vowel_label$ = Get label of interval... phone_tier phone_interval 
		
		# Only look at stressed vowels (primary = 1; secondary = 2)
		if right$(vowel_label$) = "1" or right$(vowel_label$) = "2"
			vowel_start = Get start point... phone_tier phone_interval
			vowel_end = Get end point... phone_tier phone_interval
			vowel_dur = (vowel_end - vowel_start)
			vowel_midpoint = vowel_start + (vowel_dur / 2)

			# Get word annotation
			word_interval = Get interval at time... word_tier vowel_midpoint
			word_label$ = Get label of interval... word_tier word_interval

			# Get phonation annotation
			phonation_interval = Get interval at time... phonation_tier vowel_midpoint
			phonation_label$ = Get label of interval... phonation_tier phonation_interval
			
			# Get jitter
			procedure Jitter
				select PointProcess 'soundname$'_'soundname$'
				jitter_ddp = Get jitter (ddp)... vowel_start vowel_end 0.0001 0.02 1.3
				jitter_loc = Get jitter (local)... vowel_start vowel_end 0.0001 0.02 1.3
				jitter_loc_abs = Get jitter (local, absolute)... vowel_start vowel_end 0.0001 0.02 1.3
				jitter_rap = Get jitter (rap)... vowel_start vowel_end 0.0001 0.02 1.3
				jitter_ppq5 = Get jitter (ppq5)... vowel_start vowel_end 0.0001 0.02 1.3
			endproc
			call Jitter

			# Get shimmer
			procedure Shimmer
			select Sound 'soundname$'
			plus PointProcess 'soundname$'_'soundname$'
			shimmer_loc = Get shimmer (local)... vowel_start vowel_end 0.0001 0.02 1.3 1.6
			shimmer_loc_dB = Get shimmer (local_dB)... vowel_start vowel_end 0.0001 0.02 1.3 1.6
			shimmer_apq3 = Get shimmer (apq3)... vowel_start vowel_end 0.0001 0.02 1.3 1.6
			shimmer_apq5 = Get shimmer (apq5)... vowel_start vowel_end 0.0001 0.02 1.3 1.6
			shimmer_apq11 = Get shimmer (apq11)... vowel_start vowel_end 0.0001 0.02 1.3 1.6
			shimmer_dda = Get shimmer (dda)... vowel_start vowel_end 0.0001 0.02 1.3 1.6
			endproc
			call Shimmer

			# Get harmonicity
			procedure Harmonicity
			select Harmonicity 'soundname$'
			hnr_mean = Get mean... vowel_start vowel_end
			endproc
			call Harmonicity

			# Get F1 and F2
			procedure Formants
			select Formant 'soundname$'
			f1 = Get mean... 1 vowel_start vowel_end Hertz
			f2 = Get mean... 2 vowel_start vowel_end Hertz
			endproc
			call Formants

			# Make blank things NA
			if word_label$ = ""
				word_label$ = "NA"
			endif
			if phonation_label$ = ""
				phonation_label$ = "NA"
			endif
			
			# Output
			results_line$ = "'gridfile$'	'vowel_start'	'vowel_end'	'vowel_dur'	'vowel_label$'	'word_label$'	'phonation_label$'	'jitter_ddp'	'jitter_loc'	'jitter_loc_abs'	'jitter_rap'	'jitter_ppq5'	'shimmer_loc'	'shimmer_loc_dB'	'shimmer_apq3'	'shimmer_apq5'	'shimmer_apq11'	'shimmer_dda'	'hnr_mean'	'f1'	'f2'	'newline$'"
			fileappend "'results_file$'" 'results_line$'
		endif
	endfor
		select 'gridname$'
		Remove
endfor
	
# Remove objects	
select Strings gridlist
Remove

echo Done. 