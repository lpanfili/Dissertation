# Combines VoiceSauce and Praat data for a given language
# Includes only B M C
# For eng, excludes stop words
# For eng, converts surrounding phones into six binary features
# For cmn, converts tone to phonation
# Adds VoPT
# Saves data as new csv

import csv
import argparse
import pandas as pd
import itertools

# Arguments:
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('stopwords', type=str,
                        help='path to the list of stopwords')
    parser.add_argument('lg', type=str, help='the three digit code for the language in question')
    return parser.parse_args()


# Makes and returns a list of all the stopwords for English
def get_stopwords(path):
	stopwords = []
	with open(path) as f:
		for line in f:
			line = line.strip().upper()
			stopwords.append(line)
	return stopwords


# Populates and returns a list of column headers to skip
def make_skip_list():
	skip_list = ['H1c_mean','H2c_mean','H4c_mean','A1c_mean','A2c_mean','A3c_mean','H2Kc_mean',
	'H1u_mean','H2u_mean','H4u_mean','A1u_mean','A2u_mean','A3u_mean',
	'H2Ku_mean','H5Ku_mean','H1H2u_mean','H2H4u_mean','H1A1u_mean','H1A2u_mean',
	'H1A3u_mean','H42Ku_mean','H2KH5Ku_mean','seg_Start','seg_End','H1u_means001',
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
	'A3c_means002','A3c_means003','H2Kc_means001','H2Kc_means002','H2Kc_means003','',
	'oF0_mean','oF0_means001','oF0_means002','oF0_means003','oF1_mean','oF1_means001',
	'oF1_means002','oF1_means003','oF2_mean','oF2_means001','oF2_means002','oF2_means003',
	'oF3_mean','oF3_means001','oF3_means002','oF3_means003','oF4_mean','oF4_means001',
	'oF4_means002','oF4_means003','sB1_mean','sB1_means001','sB1_means002','sB1_means003',
	'sB2_mean','sB2_means001','sB2_means002','sB2_means003','sB3_mean','sB3_means001',
	'sB3_means002','sB3_means003','sB4_mean','sB4_means001','sB4_means002','sB4_means003',
	'pB1_mean','pB1_means001','pB1_means002','pB1_means003','pB2_mean','pB2_means001',
	'pB2_means002','pB2_means003','pB3_mean','pB3_means001','pB3_means002','pB3_means003',
	'pB4_mean','pB4_means001','pB4_means002','pB4_means003','oB1_mean','oB1_means001',
	'oB1_means002','oB1_means003','oB2_mean','oB2_means001','oB2_means002','oB2_means003',
	'oB3_mean','oB3_means001','oB3_means002','oB3_means003','oB4_mean','oB4_means001',
	'oB4_means002','oB4_means003','epoch_mean','epoch_means001','epoch_means002',
	'epoch_means003','soe_mean','soe_means001','soe_means002','soe_means003']
	return skip_list


# Takes in the Praat data
# Returns a dictionary (and Praat header)
# Keys = unique vowel IDs (filename + start + end)
# Values = all the Praat data
def get_praat(lg):
	praat_dict = {}
	path = "../data/lgs/" + lg + "/raw/" + lg + "-praat.txt"
	with open(path) as praat:
		reader = csv.reader(praat, delimiter = '\t')
		praat_header = next(reader)
		for line in reader:
			filename = line[0][:-9] # Remove ".textgrid"
			if len(line) == 49 or len(line) == 42: # Exclude incomplete lines (eng, other length)
				if filename != "": # Exclude missing filenames
					if lg == 'cmn': # Convert Mandarin tone to phonation
						line[5] = tone_to_phonation(line[5])
					if line[5] == 'B' or line[5] == 'M' or line[5] == 'C':
						start = float(line[1]) * 1000
						start = '{:.3f}'.format(round(float(start),3))
						end = float(line[2]) * 1000
						end = '{:.3f}'.format(round(float(end),3))
						vowel_id = filename + str(start)[:5] + str(end)[:5]
						if vowel_id in praat_dict:
							print("Vowel ID already exists: ", vowel_id)
						else:
							praat_dict[vowel_id] = line
	return praat_dict, praat_header


# Takes in the VS data
# Returns a dictionary (and VS header)
# Keys = unique vowel IDs (filename + start + end)
# Values = all the VS data
def get_vs(lg):
	vs_dict = {}
	path = "../data/lgs/" + lg + "/raw/" + lg + "-vs-3.txt"
	with open(path) as vs:
		reader = csv.reader(vs, delimiter = '\t')
		vs_header = next(reader)
		for line in reader:
			if lg == 'cmn': # Convert Mandarin tone to phonation
				line[1] = tone_to_phonation(line[1])
			if line[1] == "B" or line[1] == "M" or line[1] == "C":
				filename = line[0][:-4]
				vowel_id = filename + line[2][:5] + line[3][:5]
				if vowel_id not in vs_dict:
					vs_dict[vowel_id] = line
	return vs_dict, vs_header


# Takes in a Mandarin tone
# Converts it to phonation and returns that
def tone_to_phonation(tone):
	if tone == '01' or tone == '02':
		return 'M'
	if tone == '03':
		return 'C'
	if tone == '04':
		return '4'


# Combines VS and Praat data into a Pandas DF
# Updates the headers
# Cleans up a few things
def combine_data(praat_dict, vs_dict, praat_header, vs_header, skip_list, lg):
	if lg == 'eng':
		praat_header = praat_header[:-1] # Remove trailing header
	# Combine praat_dict and vs_dict into one
	vowel_dict = {}
	for vowel in praat_dict:
		if vowel in vs_dict:
			vowel_dict[vowel] = vs_dict[vowel] + praat_dict[vowel]
	# Convert dict to DF
	data = pd.DataFrame.from_dict(vowel_dict, orient = 'index')
	combined_header = vs_header + praat_header
	data.columns = combined_header
	# Drop columns I don't care about
	for i in skip_list:
		data = data.drop([i], axis = 1)
	# Add column for speaker (format is lg dependant)
	data = add_speaker(data, lg)
	for index, row in data.iterrows():
		# Remove .mat from filename
		row[0] = row[0][:-4]
		# Double check that VS and Praat phonation match
		if row['Label'] != row['phonation']:
			print("Mismatched phonation for " + row[0])
	#data.to_csv("/Users/Laura/Desktop/data.csv")
	return data


# Adds a column with the speaker ID
def add_speaker(data, lg):
	if lg == 'cmn' or lg == 'guj' or lg == 'hmn':
		data['speaker'] = data.apply(lambda row: row['Filename'][:2], axis = 1)
	if lg == 'eng':
		data['speaker'] = data.apply(lambda row: row['Filename'][:6], axis = 1)
	if lg == 'maj':
		data['speaker'] = data.apply(lambda row: row['Filename'][-6:-4], axis = 1)
		for index, row in data.iterrows():
			if row['speaker'] == "10":
				row['speaker'] = "M10"
	if lg == 'zap':
		data['speaker'] = data.apply(lambda row: row['Filename'][7:8], axis = 1)
	return data


# Calculates VoPT and adds it as a new column
# Returns updated DF
def add_vopt(data):
	vopt_list = []
	for index, row in data.iterrows():
		VoPT = 0
		strF0 = row['strF0_mean']
		sF0 = row['sF0_mean']
		pF0 = row['pF0_mean']
		shrF0 = row['shrF0_mean']
		tracks = [strF0, sF0, pF0, shrF0]
		pairs = list(itertools.combinations(tracks, 2))
		for pair in pairs:
			a = float(pair[0])
			b = float(pair[1])
			diff = abs(a - b)
			VoPT += diff
		vopt_list.append(VoPT)
	vopt_list = pd.Series(vopt_list)
	data['VoPT'] = vopt_list.values
	return data


# Removes the stop words and returns the data
def remove_stopwords(data, stopwords):
	data = data[~data['vowel_label'].isin(stopwords)] # Mistake in Praat script swapped vowel and word
	return data


# Converts surrounding phones to six binary features
def surrounding_phones(data):
	is_voiced = ['V', 'L', 'B', 'R', 'JH', 'Y', 'W', 'IY0', 'G', 'NG', 'Z', 'N', 'D',
			 'M', 'AH0', 'DH', 'IY1', 'UW0', 'IH2', 'UW1', 'AE1', 'AY1', 'OW1', 'IH0', 
			 'IH1', 'AE2', 'EH1', 'AH1', 'AA1', 'ZH', 'ER0', 'EY1', 'EH2', 'AY0', 'ER1', 
			 'AO2', 'OW0']
	is_obstruent = ['V', 'T', 'B', 'JH', 'G', 'S', 'K', 'Z', 'P', 'F', 'D', 'SH', 'TH', 
				'CH','DH', 'HH']
	exists = ['V', 'T', 'L', 'B', 'R', 'JH', 'Y', 'W', 'IY0', 'G', 'S', 'NG', 'K', 
		'Z', 'P', 'F', 'N', 'D', 'SH', 'TH', 'CH', 'M', 'AH0', 'DH', 'HH', 'IY1', 
		'UW0', 'IH2', 'UW1', 'AE1', 'AY1', 'OW1', 'IH0', 'IH1', 'AE2', 'EH1', 'AH1', 
		'AA1', 'ZH', 'ER0', 'EY1', 'EH2', 'AY0', 'ER1', 'AO2', 'OW0']
	data['pre_is_voiced'] = data['preceding_phone'].isin(is_voiced)
	data['fol_is_voiced'] = data['following_phone'].isin(is_voiced)
	data['pre_is_obs'] = data['preceding_phone'].isin(is_obstruent)
	data['fol_is_obs'] = data['following_phone'].isin(is_obstruent)
	data['pre_exists'] = data['preceding_phone'].isin(exists)
	data['fol_exists'] = data['following_phone'].isin(exists)
	return data


# Makes the path to save the data
def get_path(lg):
	path = "../data/lgs/" + lg + "/" + lg + "-all.csv"
	return path


def main():
	args = parse_args()
	skip_list = make_skip_list()
	praat_dict, praat_header = get_praat(args.lg)
	vs_dict, vs_header = get_vs(args.lg)
	data = combine_data(praat_dict, vs_dict, praat_header, vs_header, skip_list, args.lg)
	data = add_vopt(data)
	if args.lg == 'eng':
		stopwords = get_stopwords(args.stopwords)
		data = remove_stopwords(data, stopwords)
		data = surrounding_phones(data)
	path = get_path(args.lg)
	data.to_csv(path)


if __name__ == "__main__":
    main()