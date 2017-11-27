#########################################################################################
# Author: Laura Panfili (lpanfili@uw.edu)												
#																						
# DESCRIPTION:																			
# This script extracts vowel duration, prosodic position, surrounding phones, jitter, 	
# and shimmer. It is intended for use with ATAROS files and text grids. These measures 	
# are extracted only for stressed vowels (ending in 1 or 2), and their phonation type 	
# (indicated in the grid) is also extracted. Output is printed to a tab-delimited file.	
#																						
# EXTRACTS:																				
# duration 																				
# Preceding phone 																		
# Following phone 																		
# Percent of way through word 															
# Percent of way through spurt 															
# ms from end of word 																	
# ms from end of spurt 																	
#																						
# COMPUTES AS MEANS OVER ENTIRE VOWEL AND THIRDS OF THE VOWEL:							
# Local Jitter 																			
# Local Abs Jitter 																		
# RAP Jitter 																			
# PPQ5 Jitter 																			
# Local Shimmer 																		
# Local Shimmer dB 																		
# APQ3 Shimmer 																			
# APQ5 Shimmer 																			
# APQ11 Shimmer 																		
#########################################################################################

# Form to set directories
form Directories
	comment Enter directory with TextGrids and wav files:
	sentence Textgrid_directory /home2/lpanfili/dissertation/ATAROS
	
	comment Enter name and location of results file:
	sentence Results_name results.txt
endform

clearinfo

#-------------------------------------------------------------------------#
# Create full path for results file
# First ensure that path has final slash; add it if it doesn't
   if right$(textgrid_directory$) = "\" or right$(textgrid_directory$) = "/"
	# Nothing
   else
	# Add path division
	textgrid_directory$ = "'textgrid_directory$'/"
   endif

results_file$ = "'textgrid_directory$'" + "'results_name$'" 
print find results at 'results_file$''newline$'

#-------------------------------------------------------------------------#
# Initialize results file
results_header$ = "gridfile	vowel_start	vowel_end	vowel_dur	word_label	phonation	jitter_loc_mean	jitter_loc_1	jitter_loc_2	jitter_loc_3	jitter_loc_abs_mean	jitter_loc_abs_1	jitter_loc_abs_2	jitter_loc_abs_3	jitter_rap_mean	jitter_rap_1	jitter_rap_2	jitter_rap_3	jitter_ppq5_mean	jitter_ppq5_1	jitter_ppq5_2	jitter_ppq5_3	shimmer_loc_mean	shimmer_loc_1	shimmer_loc_2	shimmer_loc_3	shimmer_local_dB_mean	shimmer_loc_db_1	shimmer_loc_db_2	shimmer_loc_db_3	shimmer_apq3_mean	shimmer_apq3_1	shimmer_apq3_2	shimmer_apq3_3	shimmer_apq5_mean	shimmer_apq5_1	shimmer_apq5_2	shimmer_apq5_3	shimmer_apq11_mean	shimmer_apq11_1	shimmer_apq11_2	shimmer_apq11_3	preceding_phone	following_phone	word_per	utt_per	ms_from_word_end	ms_from_utt_end	vowel_label	'newline$'"

#-------------------------------------------------------------------------#
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
print 'initial_report$'

#-------------------------------------------------------------------------#
# Go through each file
for ifile to numberoffiles
	select Strings gridlist
	gridfile$ = Get string... ifile
		# gridfile$ includes file extension, eg NWF222-6B-coarse.TextGrid
	Read from file... 'textgrid_directory$''gridfile$'
	gridname$ = selected$()
		# gridname$ is TextGrid filename excluding extension, eg NWF222-6B-coarse
	soundname$ = replace$ (gridname$, "TextGrid ", "", 0)
	filename$ = soundname$ + ".wav"
		# filename$ is the sound file including extension, eg NWF222.wav

# Set all default arguments for functions (procedure at bottom of script)
   call set_parameters

	phone_intervals = Get number of intervals... phone_tier
	Read from file... 'textgrid_directory$''filename$'

	# Create Pitch, Point Process, Spectrum and Harmonicity objects
	select Sound 'soundname$'
	#To Pitch... 0 minpitch maxpitch
	To Pitch (cc)... 0.0 75 15 no 0.03 0.45 0.01 0.35 0.14 600
	select Sound 'soundname$'
	plus Pitch 'soundname$'
	To PointProcess (cc)
	select Sound 'soundname$'
	To Harmonicity (cc)... harmonicity_timestep minpitch silence_thresh periods_per_window
	select Sound 'soundname$'
	To Formant (burg)... 0.0 numformants maxformant 0.025 50.0

	for phone_interval to phone_intervals
		select 'gridname$'

		# Get vowel annotation
		vowel_label$ = Get label of interval... phone_tier phone_interval 
		
		# Only look at stressed vowels (primary = 1; secondary = 2)
		if right$(vowel_label$) = "1" or right$(vowel_label$) = "2"

			# Calculate various time points
			start = Get start point... phone_tier phone_interval
			end = Get end point... phone_tier phone_interval
			duration = (end - start)
			midpoint = start + (duration / 2)
			third = (duration / 3)
			third_1 = (start + third)
			third_2 = (end - third)

			# Get word annotation
			word_interval = Get interval at time... word_tier midpoint
			word_label$ = Get label of interval... word_tier word_interval

			# Get phonation annotation
			phonation_interval = Get interval at time... phonation_tier midpoint
			phonation_label$ = Get label of interval... phonation_tier phonation_interval

			# Get preceding phone
			if phone_interval > 1
				preceding_phone$ = Get label of interval... phone_tier (phone_interval - 1)
			else
				preceding_phone$ = "xxx"
			endif

			# Get following phone
			if phone_interval <> phone_intervals
				following_phone$ = Get label of interval... phone_tier (phone_interval + 1)
			else
				# If there is no following phone, use 'xxx'
				following_phone$ = "xxx"
			endif

			# Get word-level info
			word_interval = Get interval at time... word_tier midpoint
			word_start = Get start point... word_tier word_interval
			word_end = Get end point... word_tier word_interval
			word_dur = word_end - word_start
			word_ms_fromstart = midpoint - word_start
			word_per = word_ms_fromstart / word_dur
			ms_word = word_end - midpoint

			# Get spurt-level info
			utt_interval = Get interval at time... transcription_tier midpoint
			utt_start = Get start point... transcription_tier utt_interval
			utt_end = Get end point... transcription_tier utt_interval
			utt_dur = utt_end - utt_start
			utt_ms_fromstart = midpoint - utt_start
			utt_per = utt_ms_fromstart / utt_dur
			ms_utt = utt_end - midpoint
			
			# Get jitter
			   call Jitter

			# Get shimmer
			   call Shimmer

			# Make blank things NA
			if word_label$ = ""
				word_label$ = "NA"
			endif
			if phonation_label$ = ""
				phonation_label$ = "NA"
			endif
			
			# Output

			results_line$ = "'gridfile$'	'start:6'	'end:6'	'duration:3'	'vowel_label$'	'phonation_label$'	'jitter_loc_mean:6'	'jitter_loc_1:6'	'jitter_loc_2:6'	'jitter_loc_3:6'	'jitter_loc_abs_mean:6'	'jitter_loc_abs_1:6'	'jitter_loc_abs_2:6'	'jitter_loc_abs_3:6'	'jitter_rap_mean:6'	'jitter_rap_1:6'	'jitter_rap_2:6'	'jitter_rap_3:6'	'jitter_ppq5_mean:6'	'jitter_ppq5_1:6'	'jitter_ppq5_2:6'	'jitter_ppq5_3:6'	'shimmer_loc_mean:6'	'shimmer_loc_1:6'	'shimmer_loc_2:6'	'shimmer_loc_3:6'	'shimmer_loc_db_mean:6'	'shimmer_loc_db_1:6'	'shimmer_loc_db_2:6'	'shimmer_loc_db_3:6'	'shimmer_apq3_mean:6'	'shimmer_apq3_1:6'	'shimmer_apq3_2:6'	'shimmer_apq3_3:6'	'shimmer_apq5_mean:6'	'shimmer_apq5_1:6'	'shimmer_apq5_2:6'	'shimmer_apq5_3:6'	'shimmer_apq11_mean:6'	'shimmer_apq11_1:6'	'shimmer_apq11_2:6'	'shimmer_apq11_3:6'	'preceding_phone$'	'following_phone$'	'word_per'	'utt_per'	'ms_word'	'ms_utt'	'word_label$''newline$'"
			fileappend "'results_file$'" 'results_line$'
		endif
	endfor
		select 'gridname$'
		Remove
endfor
	
# Remove objects	
select all
Remove

print Done. 

#-------------------------------------------------------------------------#
# PROCEDURES 

procedure Jitter
	select PointProcess 'soundname$'_'soundname$'

	# loc
	jitter_loc_mean = Get jitter (local)... start end 0.0001 0.02 1.3
	jitter_loc_1 = Get jitter (local)... start third_1 0.0001 0.02 1.3
	jitter_loc_2 = Get jitter (local)... third_1 third_2 0.0001 0.02 1.3
	jitter_loc_3 = Get jitter (local)... third_2 end 0.0001 0.02 1.3

	# loc_abs
	jitter_loc_abs_mean = Get jitter (local, absolute)... start end 0.0001 0.02 1.3
	jitter_loc_abs_1 = Get jitter (local, absolute)... start third_1 0.0001 0.02 1.3
	jitter_loc_abs_2 = Get jitter (local, absolute)... third_1 third_2 0.0001 0.02 1.3
	jitter_loc_abs_3 = Get jitter (local, absolute)... third_2 end 0.0001 0.02 1.3

	# rap
	jitter_rap_mean = Get jitter (rap)... start end 0.0001 0.02 1.3
	jitter_rap_1 = Get jitter (rap)... start third_1 0.0001 0.02 1.3
	jitter_rap_2 = Get jitter (rap)... third_1 third_2 0.0001 0.02 1.3
	jitter_rap_3 = Get jitter (rap)... third_2 end 0.0001 0.02 1.3

	# ppq5
	jitter_ppq5_mean = Get jitter (ppq5)... start end 0.0001 0.02 1.3
	jitter_ppq5_1 = Get jitter (ppq5)... start third_1 0.0001 0.02 1.3
	jitter_ppq5_2 = Get jitter (ppq5)... third_1 third_2 0.0001 0.02 1.3
	jitter_ppq5_3 = Get jitter (ppq5)... third_2 end 0.0001 0.02 1.3

endproc
			
			
procedure Shimmer
	select Sound 'soundname$'
	plus PointProcess 'soundname$'_'soundname$'

	# loc
	shimmer_loc_mean = Get shimmer (local)... start end 0.0001 0.02 1.3 1.6
	shimmer_loc_1 = Get shimmer (local)... start third_1 0.0001 0.02 1.3 1.6
	shimmer_loc_2 = Get shimmer (local)... third_1 third_2 0.0001 0.02 1.3 1.6
	shimmer_loc_3 = Get shimmer (local)... third_2 end 0.0001 0.02 1.3 1.6

	# loc db
	shimmer_loc_db_mean = Get shimmer (local_dB)... start end 0.0001 0.02 1.3 1.6
	shimmer_loc_db_1 = Get shimmer (local_dB)... start third_1 0.0001 0.02 1.3 1.6
	shimmer_loc_db_2 = Get shimmer (local_dB)... third_1 third_2 0.0001 0.02 1.3 1.6
	shimmer_loc_db_3 = Get shimmer (local_dB)... third_2 end 0.0001 0.02 1.3 1.6

	# apq3
	shimmer_apq3_mean = Get shimmer (apq3)... start end 0.0001 0.02 1.3 1.6
	shimmer_apq3_1 = Get shimmer (apq3)... start third_1 0.0001 0.02 1.3 1.6
	shimmer_apq3_2 = Get shimmer (apq3)... third_1 third_2 0.0001 0.02 1.3 1.6
	shimmer_apq3_3 = Get shimmer (apq3)... third_2 end 0.0001 0.02 1.3 1.6

	# apq5
	shimmer_apq5_mean = Get shimmer (apq5)... start end 0.0001 0.02 1.3 1.6
	shimmer_apq5_1 = Get shimmer (apq5)... start third_1 0.0001 0.02 1.3 1.6
	shimmer_apq5_2 = Get shimmer (apq5)... third_1 third_2 0.0001 0.02 1.3 1.6
	shimmer_apq5_3 = Get shimmer (apq5)... third_2 end 0.0001 0.02 1.3 1.6

	# apq11
	shimmer_apq11_mean = Get shimmer (apq11)... start end 0.0001 0.02 1.3 1.6
	shimmer_apq11_1 = Get shimmer (apq11)... start third_1 0.0001 0.02 1.3 1.6
	shimmer_apq11_2 = Get shimmer (apq11)... third_1 third_2 0.0001 0.02 1.3 1.6
	shimmer_apq11_3 = Get shimmer (apq11)... third_2 end 0.0001 0.02 1.3 1.6

endproc

procedure set_parameters

	minpitch = 75
	maxpitch = 600
  	maxformant = 5500
  
  	harmonicity_timestep = 0.01
  	silence_thresh  = 0.1
 	periods_per_window = 1
  
  	numformants = 5

  	# Set tier numbers
  	phone_tier = 1
 	word_tier = 2
  	transcription_tier = 3
  	phonation_tier = 4
  
endproc
