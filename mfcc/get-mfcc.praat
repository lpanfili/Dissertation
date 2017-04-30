# Form to set directories
form Directories
	comment Enter directory with TextGrids:
	sentence Textgrid_directory /Users/Laura/Desktop/Dissertation/test-data4
	#sentence Textgrid_directory /home2/lpanfili/dissertation/ATAROS/
	
	
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
# Initialize results file

results_header$ = "gridfile	vowel_start	vowel_end	vowel_dur	vowel_label	word_label	phonation	mfcc_mean-1	mfcc_mean-2	mfcc_mean-3	mfcc_mean-4	mfcc_mean-5	mfcc_mean-6	mfcc_mean-7	mfcc_mean-8	mfcc_mean-9	mfcc_mean-10	mfcc_mean-11	mfcc_mean-12	mfcc_mean-13	mfcc_mean-14	mfcc_mean-15	mfcc_mean-16	mfcc_mean-17	mfcc_mean-18	mfcc_mean-19	mfcc_mean-20	mfcc_mean-21	mfcc_mean-22	mfcc_mean-23	mfcc_mean-24	stddv-1	stddv-2	stddv-3	stddv-4	stddv-5	stddv-6	stddv-7	stddv-8	stddv-9	stddv-10	stddv-11	stddv-12	stddv-13	stddv-14	stddv-15	stddv-16	stddv-17	stddv-18	stddv-19	stddv-20	stddv-21	stddv-22	stddv-23	stddv-24'newline$'"


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

# set all default arguments for functions (procedure at bottom of script)
   call set_parameters

	phone_intervals = Get number of intervals... phone_tier
	Read from file... 'textgrid_directory$''filename$'

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

			# "file$" is just the name of the file (no ext or prefix)
			file$ = replace$ (filename$, ".wav", "", 1)
			filepart$ = file$ + "_part"

			# Make MFCC
			call makeMFCC

			# Make blank things NA
			if word_label$ = ""
				word_label$ = "NA"
			endif
			if phonation_label$ = ""
				phonation_label$ = "NA"
			endif
			
			# Output

			results_line$ = "'gridfile$'	'start:3'	'end:3'	'duration:3'	'vowel_label$'	'word_label$'	'phonation_label$'"

			# Get MFCCs
			mfccstddv$ = ""
			for j from 1 to 24
				call getMFCC: j
				results_line$ = results_line$ + "	'mfcc_mean'"
				mfccstddv$ = mfccstddv$ + "'stddv'	"
			endfor
			results_line$ = results_line$ + "	" + mfccstddv$ + "'newline$'"
			select Sound 'filepart$'
			Remove
			select Matrix 'filepart$'
			Remove
			select MFCC 'filepart$'
			Remove
			
			fileappend "'results_file$'" 'results_line$'
		endif
	endfor
		select 'gridname$'
		Remove
endfor

# Remove objects	
#select all
#Remove

print Done. 

#-------------------------------------------------------------------------#
# PROCEDURES 
			
procedure makeMFCC
	select Sound 'soundname$'
	Extract part... start end rectangular 1.0 no
	To MFCC... 24 0.010 0.005 100.0 100.0 0.0  
	# The first number is number of coefficients
	To Matrix
	col = Get number of columns
endproc

procedure getMFCC: coeff
	# Gets average of one CC over the whole vowel
	mfcc_total = 0.0
	for i from 1 to col
		mfcc = Get value in cell... coeff i
		# First number is coefficient number
		mfcc_total = mfcc + mfcc_total
	# Gets stddv of one CC over the whole vowel at 10 ms frames
	stddv = Get standard deviation... 0.0 col coeff-0.5 coeff+0.5
	endfor
	mfcc_mean = mfcc_total / col
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