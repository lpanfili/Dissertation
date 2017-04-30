# This script takes a TextGrid with two identical tiers
# The first tier's contents are replaced with the vowel
# The second tier's contents are replaced with the tone

import os
import re

for root, dirs, files in os.walk("/Users/Laura/Desktop/Mandarin_Grids-old"):
	for name in files:
		if name != ".DS_Store":
			with open(os.path.join(root,name), 'r') as f:
				with open(os.path.join("/Users/Laura/Desktop/Mandarin_Grids-new",name), 'w') as newf:
					lines = f.readlines()
					i = 0
					while 'phonation' not in lines[i]:
						# replace dd_l with a
						line = re.sub(r'0\d_\D', r'a', lines[i])
						newf.write(line)
						i += 1
					for x in range(i,len(lines)):
						# replace with just the number
						line = re.sub(r'(\d\d)_\D', r'\1', lines[x])
						newf.write(line)