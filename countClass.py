# Counts the classes and saves a csv for latex

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
	'epoch_means003','soe_mean','soe_means001','soe_means002','soe_means003','sF2_mean',
	'sF2_means001','sF2_means002','sF2_means003','sF3_mean','sF3_means001','sF3_means002',
	'sF3_means003','sF4_mean','sF4_means001','sF4_means002','sF4_means003','pF1_mean',
	'pF1_means001','pF1_means002','pF1_means003','pF2_mean','pF2_means001','pF2_means002',
	'pF2_means003','pF3_mean','pF3_means001','pF3_means002','pF3_means003','pF4_mean',
	'pF4_means001','pF4_means002','pF4_means003']
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
	for index, row in data.iterrows():
		# Double check that VS and Praat phonation match
		if row['Label'] != row['phonation']:
			print("Mismatched phonation for " + row[0])
	return data

# Removes the stop words and returns the data
def remove_stopwords(data, stopwords):
	# Mistake in Praat script swapped vowel and word (just for eng)
	data = data.rename(columns = {'vowel_label': 'word_label1', 'word_label': 'vowel_label1'})
	data = data.rename(columns = {'vowel_label1': 'vowel_label', 'word_label1': 'word_label'})
	data = data[~data['word_label'].isin(stopwords)] 
	return data

def count_classes(data):
	B = 0
	M = 0
	C = 0
	phonation_list = data['Label'].copy().values.tolist()
	for i in phonation_list:
		if i == 'B':
			B += 1
		if i == 'M':
			M += 1
		if i == 'C':
			C += 1
	total = B + M + C
	counts = [['B', 'M', 'C', 'Total'], [B, M,C , total], ['\\textit{' + str( + round(((B / total) * 100),3)) + '\%}', \
				'\\textit{' + str(round(((M / total) * 100),3)) + '\%}', \
				'\\textit{' + str(round(((C / total) * 100),3)) + '\%}', '~']]
	counts = pd.DataFrame(counts)
	counts = counts.transpose()
	counts = counts.set_index([0])
	print(counts)
	return counts


def main():
	args = parse_args()
	skip_list = make_skip_list()
	praat_dict, praat_header = get_praat(args.lg)
	vs_dict, vs_header = get_vs(args.lg)
	data = combine_data(praat_dict, vs_dict, praat_header, vs_header, skip_list, args.lg)
	if args.lg == 'eng':
		stopwords = get_stopwords(args.stopwords)
		data = remove_stopwords(data, stopwords)
	counts = count_classes(data)
	path = '../data/lgs/' + args.lg + "/" + args.lg + "-counts.csv"
	counts.to_csv(path)


if __name__ == "__main__":
    main()