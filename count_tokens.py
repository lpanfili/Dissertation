import csv

bmcDict = {} # Only B M C stressed vowels
allDict = {} # All stressed vowels

with open("/Users/Laura/Desktop/Dissertation/data/english/All/All-praat-1.txt") as f:
	reader = csv.reader(f, delimiter = "\t")
	for line in reader:
		if line[0] not in bmcDict:
			bmcDict[line[0]] = 0
		if line[0] not in allDict:
			allDict[line[0]] = 0
		allDict[line[0]] += 1
		if line[4] != "0" and line[4] != "1":
			bmcDict[line[0]] += 1
print(bmcDict)
		