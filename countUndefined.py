# python3 countUndefined.py /Users/Laura/Desktop/Dissertation/test-data/phonetic_stoplist.txt /Users/Laura/Desktop/Dissertation/Code/pitch-settings/results-default.txt
import csv
import numpy as np
import argparse

# Inputs should be path for stop list and path for data files (Praat, VS)
def parseArgs():
	parser = argparse.ArgumentParser()
	parser.add_argument("stoplist")
	parser.add_argument("praat")
	return parser.parse_args()

# Make a list from a file of words
# Use this for the stop words to be excluded
def getStopWords(filename):
	stopWords = []
	with open(filename) as f:
		for line in f:
			line = line.strip().upper()
			stopWords.append(line)
	return stopWords

# Remove anything but B M C
# Return array of data and list of headers
# x is the index of the column containing labels
def prepData(filename, x):
	data = []
	phonationLabs = ["B", "C", "M"]
	with open(filename) as f:
		count = 0
		reader = csv.reader(f)
		header = next(reader)
		for line in reader:
			count += 1
			if line[x] in phonationLabs:
				data.append(line)
	print("Vowels, total:", count)
	data = np.array(data)
	return data, header

# Removes stop words
def remStopWords(data, stopWords):
	remove = []
	for row in data:
		remove.append(row[5] not in stopWords)
	remove = np.array(remove)
	data = data[remove]
	np.savetxt('out.txt', data, fmt = '%s')
	print("Vowels, without stop words, 0, or 1:", len(data))
	return data

# Count measured that are "undefined"
# Replace "undefined" with 1
def undefined(x):
	udefCount = 0 # overall undefined count
	udM = 0 # male undefined count
	udF = 0 # female undefined count
	total = 0 # overall total
	mTotal = 0 # male total
	fTotal = 0 # female total
	fB = 0
	fM = 0
	fC = 0
	mB = 0
	mM = 0
	mC = 0
	for row in x:
		if "NWF" in row[0]:
			if row[6] == "B":
				fB += 1
			if row[6] == "M":
				fM += 1
			if row[6] == "C":
				fC += 1
			for i in range(len(row)):
				fTotal += 1
				if row[i] == "--undefined--":
					udF += 1
		if "NWM" in row[0]:
			if row[6] == "B":
				mB += 1
			if row[6] == "M":
				mM += 1
			if row[6] == "C":
				mC += 1
			for i in range(len(row)):
				mTotal += 1
				if row[i] == "--undefined--":
					udM += 1
	total = fTotal + mTotal
	udefCount = udF + udM
	print("Undefined:", udefCount)
	print("Total:", total)
	print("Percent:", float((udefCount/total) * 100))
	print("Male Undefined:", udM)
	print("Male Total:", mTotal)
	print("Male Percent:", float((udM/mTotal) * 100))
	print("Female Undefined:", udF)
	print("Female Total:", fTotal)
	print("Female Percent:", float((udF/fTotal) * 100))
	print("Breathy, F:", fB)
	print("Modal, F:", fM)
	print("Creaky, F:", fC)
	print("Breathy, M:", mB)
	print("Modal, M:", mM)
	print("Creaky, M:", mC)
	print("Total F:", (fB + fM + fC))
	print("Total M:", (mB + mM + mC))

def main():
	args = parseArgs()
	data, praatHeader = prepData(args.praat, 6)
	stopWords = getStopWords(args.stoplist)
	data = remStopWords(data, stopWords)
	undefined(data)

if __name__ == "__main__":
	main()