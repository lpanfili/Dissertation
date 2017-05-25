# This script takes a TextGrid with two identical tiers
# The second tier's contents are replaced with the tone letter

import os
import re

toneList = []
correctTones = ['b','d','g','j','m','s','v']
for root, dirs, files in os.walk("/Users/Laura/Desktop/Hmong_Grids-old"):
	for name in files:
		if name != ".DS_Store":
			with open(os.path.join(root,name), 'r') as f:
				with open(os.path.join("/Users/Laura/Desktop/Hmong_Grids-new",name), 'w') as newf:
					word = re.sub(r'.*-([^_]*)(_\d_)?-[wg]_Audio.TextGrid', r'\1', name)
					toneLetter = word[-1]
					if toneLetter == "2":
						toneLetter = word[-2]
					if toneLetter not in correctTones:
						toneLetter = "x"
					if toneLetter not in toneList:
						toneList.append(toneLetter)
					lines = f.readlines()
					i = 0
					while 'phonation' not in lines[i]:
						newf.write(lines[i])
						i += 1
					for x in range(i,len(lines)):
						# Replace vowels with tone letters
						phonation = "".join(['text = "', toneLetter, '"'])
						line = re.sub(r'text\s=\s".+"', phonation, lines[x])
						newf.write(line)
print(toneList)