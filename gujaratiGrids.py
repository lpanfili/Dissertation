# This script takes a TextGrid with two identical tiers
# The first tier's contents are replaced with the vowel
# The second tier's contents are replaced with the phonation type

import os
import re

for root, dirs, files in os.walk("/Users/Laura/Desktop/Dissertation/Gujarati/test/"):
	for name in files:
		if name != ".DS_Store":
			newname = name.replace("-UCLA", "")
			with open(os.path.join(root,name), 'r') as f:
				with open(os.path.join(root,newname), 'w') as newf:
					lines = f.readlines()
					i = 0
					while 'phonation' not in lines[i]:
						# replace Vh, VH, V= with V
						line = re.sub(r'([^hH=]*)[hH=]', r'\1', lines[i])
						newf.write(line)
						i += 1
					for x in range(i,len(lines)):
						# replace anything ending with '=' with 'B'
						line = re.sub(r'.*\=', r'B', lines[x])
						# replace anything ending with 'h' or 'H' with nothing
						line = re.sub(r'.*[hH]','',line)
						# replace anything else with M
						line = re.sub(r'.*','M', line)
						newf.write(lines[x])