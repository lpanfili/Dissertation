# Lists all the preceding and following phones in the English data
# Just so that I know what to map later

import csv

phones = []
with open("/Users/Laura/Desktop/Dissertation/data/english/eng-all.csv") as f:
	reader = csv.reader(f)
	header = next(reader)
	for line in reader:
		pre = line[129]
		fol = line[130]
		if pre not in phones:
			phones.append(pre)
		if fol not in phones:
			phones.append(fol)

print(phones)