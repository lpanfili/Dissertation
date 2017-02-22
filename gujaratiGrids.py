# This script takes a TextGrid with two identical tiers
# The first tier's contents are replaced with the vowel
# The second tier's contents are replaced with the phonation type

import os
import re

for root, dirs, files in os.walk("/Users/Laura/Desktop/Dissertation/Gujarati/Gujarati_TextGrids-original/"):
	for name in files:
		if name != ".DS_Store":
			newname = name.replace("-UCLA", "")
			with open(os.path.join(root,name), 'r') as f:
				with open(os.path.join("/Users/Laura/Desktop/Dissertation/Gujarati/Gujarati_TextGrids-edited/",newname), 'w') as newf:
					lines = f.readlines()
					i = 0
					while 'phonation' not in lines[i]:
						# replace Vh, VH, V= with V
						line = re.sub(r'([^hH=]*)[hH=]', r'\1', lines[i])
						newf.write(line)
						i += 1
					for x in range(i,len(lines)):
						# replace anything ending with '=' with 'B'
						line = re.sub(r'(text\s\=\s").*\=', r'\1B', lines[x])
						# make blanks 9
						line = re.sub(r'text\s=\s""','text = "9"', line)
						# replace but B or 9 with M
						line = re.sub(r'(text\s=\s")[^B|9]*"',r'\1M"', line)
						# replace anything ending with 'h' or 'H' with nothing
						line = re.sub(r'(text\s=\s").*[hH]',r'\1', line)
						# turn 9 back into nothing
						line = re.sub(r'text\s=\s"9"','text = ""', line)
						newf.write(line)