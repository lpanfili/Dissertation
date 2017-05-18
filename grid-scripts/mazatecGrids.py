# Replaces phonation types with abbreviations
# "breathy" > "B"
# "modal" > "M"
# "creaky" > "C"
# Replaces vowel with JUST the vowel (removes following number and any _)

import os
import re

for root, dirs, files in os.walk("/Users/Laura/Desktop/Mazatec_Grids-old/"):
	for name in files:
		if name != ".DS_Store":
			with open(os.path.join("/Users/Laura/Desktop/Mazatec_Grids-old/", name), 'r') as f:
				with open(os.path.join("/Users/Laura/Desktop/Mazatec_Grids-new", name), 'w') as newf:
					lines = f.readlines()
					for line in lines:
						line = re.sub(r'breathy', r'B', line)
						line = re.sub(r'modal', r'M', line)
						line = re.sub(r'creaky', r'C', line)
						line = re.sub(r'(text\s=\s".*)\d_?_?(".*)',r'\1\2', line)
						newf.write(line)