# Form to set directories
form Directories
	comment Enter directory with TextGrids:
	sentence Textgrid_directory /Users/Laura/Desktop/Dissertation/test-data2
	
	comment Enter name and location of results file:
	sentence Results_name results.txt
endform

clearinfo

#-------------------------------------------------------------------------#
# create full path for results file
# first ensure that path has final slash; add it if it doesn't
   if right$(textgrid_directory$) = "\" or right$(textgrid_directory$) = "/"
	# nothing
   else
	# add path division
	textgrid_directory$ = "'textgrid_directory$'/"
   endif

results_file$ = "'textgrid_directory$'" + "'results_name$'" 
print find results at 'results_file$''newline$'


#-------------------------------------------------------------------------#
# set all default arguments for functions (procedure at bottom of script)
   call set_parameters

#-------------------------------------------------------------------------#
# Initialize results file
results_header$ = "gridfile,vowel_start,vowel_end,vowel_dur,vowel_label,word_label,phonation,jitter_ddp,jitter_loc,jitter_loc_abs,jitter_rap,jitter_ppq5,jitter_ddp_1,jitter_loc_1,jitter_loc_abs_1,jitter_rap_1,jitter_ppq5_1,jitter_ddp_2,jitter_loc_2,jitter_loc_abs_2,jitter_rap_2,jitter_ppq5_2,jitter_ddp_3,jitter_loc_3,jitter_loc_abs_3,jitter_rap_3,jitter_ppq5_3,shimmer_loc,shimmer_local_dB,shimmer_apq3,shimmer_apq5,shimmer_apq11,shimmer_loc_1,shimmer_loc_dB_1,shimmer_apq3_1,shimmer_apq5_1,shimmer_apq11_1,shimmer_loc_2,shimmer_loc_dB_2,shimmer_apq3_2,shimmer_apq5_2,shimmer_apq11_2,shimmer_loc_3,shimmer_loc_dB_3,shimmer_apq3_3,shimmer_apq5_3,shimmer_apq11_3,hnr,hnr_1,hnr_2,hnr_3,f1,f1_1,f1_2,f1_3,f2,f2_1,f2_2,f2_3,F0,F0_1,F0_2,F0_3,energy,energy_1,energy_2,energy_3'newline$'"

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
	#filename$ = gridname$ - "TextGrid"
	soundname$ = replace$ (gridname$, "TextGrid ", "", 0)
	filename$ = soundname$ + ".wav"
		# filename$ is the sound file including extension, eg NWF222.wav

	phone_intervals = Get number of intervals... phone_tier
	Read from file... 'textgrid_directory$''filename$'

	# Create Pitch, Point Process, and Harmonicity objects
	select Sound 'soundname$'
	To Pitch... 0 minpitch maxpitch
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
			vowel_start = Get start point... phone_tier phone_interval
			vowel_end = Get end point... phone_tier phone_interval
			vowel_dur = (vowel_end - vowel_start)
			vowel_midpoint = vowel_start + (vowel_dur / 2)
			vowel_third = (vowel_dur / 3)

			# Get word annotation
			word_interval = Get interval at time... word_tier vowel_midpoint
			word_label$ = Get label of interval... word_tier word_interval

			# Get phonation annotation
			phonation_interval = Get interval at time... phonation_tier vowel_midpoint
			phonation_label$ = Get label of interval... phonation_tier phonation_interval
			
			# Get jitter
			   call Jitter

			# Get shimmer
			   call Shimmer

			# Get harmonicity
			   call Harmonicity

			# Get F1 and F2
			   call Formants

			# Get F0
				call F0

			# Get energy
				call Energy

			# Make blank things NA
			if word_label$ = ""
				word_label$ = "NA"
			endif
			if phonation_label$ = ""
				phonation_label$ = "NA"
			endif
			
			# Output
			results_line$ = "'gridfile$','vowel_start:3','vowel_end:3','vowel_dur:3','vowel_label$','word_label$','phonation_label$','jitter_ddp:6','jitter_loc:6','jitter_loc_abs:6','jitter_rap:6','jitter_ppq5:6','jitter_output$''shimmer_loc:6','shimmer_loc_dB:6','shimmer_apq3:6','shimmer_apq5:6','shimmer_apq11:6','shimmer_output$''hnr_mean:1','harm_output$''f1:0','f1_output$''f2:0','f2_output$''f0:0','f0_output$''energy:6','energy_output$''newline$'"
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
	# all
	jitter_ddp = Get jitter (ddp)... vowel_start vowel_end 0.0001 0.02 1.3
	jitter_loc = Get jitter (local)... vowel_start vowel_end 0.0001 0.02 1.3
	jitter_loc_abs = Get jitter (local, absolute)... vowel_start vowel_end 0.0001 0.02 1.3
	jitter_rap = Get jitter (rap)... vowel_start vowel_end 0.0001 0.02 1.3
	jitter_ppq5 = Get jitter (ppq5)... vowel_start vowel_end 0.0001 0.02 1.3
	jitter_output$ = ""
	for i from 0 to 2
		start = (vowel_start + vowel_third * i)
		end = (vowel_start + (vowel_third * (i+1)))
		jitter_ddp_j = Get jitter (ddp)... start end 0.0001 0.02 1.3
		jitter_loc_j = Get jitter (local)... start end 0.0001 0.02 1.3
		jitter_loc_abs_j = Get jitter (local, absolute)... start end 0.0001 0.02 1.3
		jitter_rap_j = Get jitter (rap)... start end 0.0001 0.02 1.3
		jitter_ppq5_j = Get jitter (ppq5)... start end 0.0001 0.02 1.3
		output$ = "'jitter_ddp_j:6','jitter_loc_j:6','jitter_loc_abs_j:6','jitter_rap_j:6','jitter_ppq5_j:6',"
		jitter_output$ = jitter_output$ + output$
	endfor
endproc
			
			
procedure Shimmer
	select Sound 'soundname$'
	plus PointProcess 'soundname$'_'soundname$'
	shimmer_loc = Get shimmer (local)... vowel_start vowel_end 0.0001 0.02 1.3 1.6
	shimmer_loc_dB = Get shimmer (local_dB)... vowel_start vowel_end 0.0001 0.02 1.3 1.6
	shimmer_apq3 = Get shimmer (apq3)... vowel_start vowel_end 0.0001 0.02 1.3 1.6
	shimmer_apq5 = Get shimmer (apq5)... vowel_start vowel_end 0.0001 0.02 1.3 1.6
	shimmer_apq11 = Get shimmer (apq11)... vowel_start vowel_end 0.0001 0.02 1.3 1.6
	shimmer_output$ = ""
	for i from 0 to 2
		start = (vowel_start + vowel_third * i)
		end = (vowel_start + (vowel_third * (i+1)))
		shimmer_loc_j = Get shimmer (local)... start end 0.0001 0.02 1.3 1.6
		shimmer_loc_dB_j = Get shimmer (local_dB)... start end 0.0001 0.02 1.3 1.6
		shimmer_apq3_j = Get shimmer (apq3)... start end 0.0001 0.02 1.3 1.6
		shimmer_apq5_j = Get shimmer (apq5)... start end 0.0001 0.02 1.3 1.6
		shimmer_apq11_j = Get shimmer (apq11)... start end 0.0001 0.02 1.3 1.6
		output$ = "'shimmer_loc_j:6','shimmer_loc_dB_j:6','shimmer_apq3_j:6','shimmer_apq5_j:6','shimmer_apq11_j:6',"
		shimmer_output$ = shimmer_output$ + output$
	endfor
endproc



procedure Harmonicity
	select Harmonicity 'soundname$'
	hnr_mean = Get mean... vowel_start vowel_end
	harm_output$ = ""
	for i from 0 to 2
		start = (vowel_start + vowel_third * i)
		end = (vowel_start + (vowel_third * (i+1)))
		harm = Get mean... start end
		output$ = "'harm:1',"
		harm_output$ = harm_output$ + output$
	endfor
endproc
			
			
procedure Formants
	select Formant 'soundname$'
	f1 = Get mean... 1 vowel_start vowel_end Hertz
	f2 = Get mean... 2 vowel_start vowel_end Hertz
	f1_output$ = ""
	f2_output$ = ""
	for i from 0 to 2
		start = (vowel_start + vowel_third * i)
		end = (vowel_start + (vowel_third * (i+1)))
		f1_j = Get mean... 1 start end Hertz
		f2_j = Get mean... 2 start end Hertz
		output_1$ = "'f1_j:0',"
		output_2$ = "'f2_j:0',"
		f1_output$ = f1_output$ + output_1$
		f2_output$ = f2_output$ + output_2$
	endfor
endproc

procedure F0
	select Pitch 'soundname$'
	f0 = Get mean... vowel_start vowel_end Hertz
	f0_output$ = ""
	for i from 0 to 2
		start = (vowel_start + vowel_third * i)
		end = (vowel_start + (vowel_third * (i+1)))
		f0_j = Get mean... start end Hertz
		output$ = "'f0_j:0',"
		f0_output$ = f0_output$ + output$
	endfor
endproc

procedure Energy
	select Sound 'soundname$'
	energy = Get energy... vowel_start vowel_end
	energy_output$ = ""
	for i from 0 to 2
		start = (vowel_start + vowel_third * i)
		end = (vowel_start + (vowel_third * (i+1)))
		energy_j = Get energy... start end
		output$ = "'energy_j:6',"
		energy_output$ = energy_output$ + output$
	endfor
endproc


procedure set_parameters
  minpitch = 75
  maxpitch = 400
  
  harmonicity_timestep = 0.01
  silence_thresh  = 0.1
  periods_per_window = 1
  
  numformants = 5
  maxformant = 5500
  # CHANGE THE ABOVE SETTINGS BY GENDER!

  # Set tier numbers
  phone_tier = 1
  word_tier = 2
  transcription_tier = 3
  phonation_tier = 4
  
endproc