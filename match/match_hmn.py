# Makes one giant spreadsheet by matching up data points
# from Praat and VS output

import csv

# Key is filename + first three digits of start time
# Value is a list of all the measures
praatDict = {}
vs3Dict = {}

# Praat file
with open("/Users/Laura/Desktop/Dissertation/data/hmong/hmong-praat.txt") as praat:
	reader = csv.reader(praat, delimiter = '\t')
	praatheader = next(reader)
	for line in reader:
		measures = [] # contains all data but filename
		filename = line[0][:-9] # Remove ".textgrid"
		if len(line) == 46: # Exclude incomplete lines
			if filename != "": # Exclude missing file names
				if line[5] != "": # Exclude missing phonation types
					start = (float(line[1]) * 1000) # Make start time match VS
					line[1] = start
					idkey = filename + str(start)[:3] # ID key is filename + first 3 # in start
					for i in range(0,46):
						measures.append(line[i])
		praatDict[idkey] = measures

# VS thirds file
with open("/Users/Laura/Desktop/Dissertation/data/hmong/hmong-vs-3.txt") as vs3:
	reader = csv.reader(vs3, delimiter = '\t')
	vs3header = next(reader)
	for line in reader:
		measures = []
		if line[1] == "B" or line[1] == "M" or line[1] == "C": # Exclude missing phonation
			filename = line[0][:-4]
			idkey = filename + line[2][:3]
			for i in range(0,273):
				measures.append(line[i])
		vs3Dict[idkey] = measures

alldata = [] # Contains EVERYTHING
for key in praatDict:
	if key in vs3Dict:
		newline = praatDict[key] + vs3Dict[key]
		if len(newline) == 319: # Exclude any remaining errors
			alldata.append(newline)

header = praatheader + vs3header

skip = [] # Indices of columns to skip
skiplist = ['H1c_mean','H2c_mean','H4c_mean','A1c_mean','A2c_mean','A3c_mean','H2Kc_mean',
	'Filename', 'H1u_mean','H2u_mean','H4u_mean','A1u_mean','A2u_mean','A3u_mean',
	'H2Ku_mean','H5Ku_mean','H1H2u_mean','H2H4u_mean','H1A1u_mean','H1A2u_mean',
	'H1A3u_mean','H42Ku_mean','H2KH5Ku_mean','Label','seg_Start','seg_End','H1u_means001',
	'H1u_means002','H1u_means003','H2u_means001','H2u_means002','H2u_means003',
	'H4u_means001','H4u_means002','H4u_means003','A1u_means001','A1u_means002',
	'A1u_means003','A2u_means001','A2u_means002','A2u_means003','A3u_means001',
	'A3u_means002','A3u_means003','H2Ku_means001','H2Ku_means002','H2Ku_means003',
	'H5Ku_means001','H5Ku_means002','H5Ku_means003','H1H2u_means001','H1H2u_means002',
	'H1H2u_means003','H2H4u_means001','H2H4u_means002','H2H4u_means003','H1A1u_means001',
	'H1A1u_means002','H1A1u_means003','H1A2u_means001','H1A2u_means002','H1A2u_means003',
	'H1A3u_means001','H1A3u_means002','H1A3u_means003','H42Ku_means001','H42Ku_means002',
	'H42Ku_means003','H2KH5Ku_means001','H2KH5Ku_means002','H2KH5Ku_means003','H1c_means001',
	'H1c_means002','H1c_means003','H2c_means001','H2c_means002','H2c_means003',
	'H4c_means001','H4c_means002','H4c_means003','A1c_means001','A1c_means002',
	'A1c_means003','A2c_means001','A2c_means002','A2c_means003','A3c_means001',
	'A3c_means002','A3c_means003','H2Kc_means001','H2Kc_means002','H2Kc_means003']

for i in range(len(header)):
	if header[i][0:3] == "soe" or header[i][0:5] == "epoch" or header[i][0:2] == "oB":
		skip.append(i)
	if header[i][0:2] == "pB" or header[i][0:2] == "sB" or header[i][0:2] == "oF":
		skip.append(i)
	if header[i][1:3] == "F4" or header[i][1:3] == "F3" or header[i][1:3] == "F2":
		skip.append(i)
	if header[i] in skiplist:
		skip.append(i)

# Make a new header with only the columns I care about
newheader = []
for i in range(len(header)):
	if i not in skip:
		newheader.append(header[i])
newheader.insert(0, 'speaker')

w = csv.writer(open("/Users/Laura/Desktop/Dissertation/data/hmong/hmn-all.csv", "w"))
w.writerow(newheader)

B = 0
M = 0
C = 0
for line in alldata:
	newline = []
	speaker = line[0][:2]
	if speaker[1] == "-":
		speaker = speaker[0]
	# Check that phonation type always matches
	praatPhon = line[5]
	vs3Phon = line[47]
	if praatPhon == vs3Phon:
		for i in range(len(header)): # Go through each column
			if i not in skip: # If it's not a column to skip
				newline.append(line[i]) # Write it to a new line
		if line[5] == "B":
			B += 1
		if line[5] == "M":
			M += 1
		if line[5] == "C":
			C += 1
	newline.insert(0, speaker)
	w.writerow(newline)
print("Breathy:", B)
print("Modal:", M)
print("Creaky:", C)



