# This script takes a TextGrid with two identical tiers
# The first tier's contents are replaced with the vowel
# The second tier's contents are replaced with the phonation type

import os
import re

for root, dirs, files in os.walk("/Users/Laura/Desktop/Zapotec_Grids-old/"):
	for name in files:
		if name != ".DS_Store":
			newname = name.replace("-UCLA", "")
			#newname = name.replace("textgrid", "TextGrid")
			with open(os.path.join(root,name), 'r') as f:
				with open(os.path.join("/Users/Laura/Desktop/Zapotec_Grids/",newname), 'w') as newf:
					lines = f.readlines()
					i = 0
					while 'phonation' not in lines[i]: # While we're on the VOWEL tier
						line = re.sub(r'(text\s=\s")(.)..*"', r'\1\2"', lines[i])
						newf.write(line)
						i += 1
					for x in range(i,len(lines)): # While we're on the PHONATION tier
						line = re.sub(r'(text\s=\s").(.).*"', r'\1\2"', lines[x])
						newf.write(line)