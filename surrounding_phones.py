# Convert preceding and following into:
# is-voiced
# is-obstruent
# exists

import csv

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



w = csv.writer(open("/Users/Laura/Desktop/Dissertation/data/lgs/eng/eng-all-01-nov.csv", "w"))
with open("/Users/Laura/Desktop/Dissertation/data/lgs/eng/eng-all-nov-vopt.csv") as f:
	reader = csv.reader(f)
	header = next(reader)
	header.append("pre_is_voiced")
	header.append("pre_is_obs")
	header.append("pre_exists")
	header.append("fol_is_voiced")
	header.append("fol_is_obs")
	header.append("fol_exists")
	w.writerow(header)
	for line in reader:
		#preceding = line[129]
		#following = line[130]
		preceding = line[8]
		following = line[9]
		preceding_is_voiced = 2
		preceding_is_obstruent = 2
		preceding_exists = 2
		following_is_voiced = 2
		following_is_obstruent = 2
		following_exists = 2
		if preceding in is_voiced:
			preceding_is_voiced = 1
		if preceding in is_obstruent:
			preceding_is_obstruent = 1
		if preceding in exists:
			preceding_exists = 1
		if following in is_voiced:
			following_is_voiced = 1
		if following in is_obstruent:
			following_is_obstruent = 1
		if following in exists:
			following_exists = 1
		line.append(preceding_is_voiced)
		line.append(preceding_is_obstruent)
		line.append(preceding_exists)
		line.append(following_is_voiced)
		line.append(following_is_obstruent)
		line.append(following_exists)
		w.writerow(line)

