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
	#print("Vowels, total:", count)
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
	#print("Vowels, without stop words, 0, or 1:", len(data))
	return data

# Count measured that are "undefined"
# Replace "undefined" with 1
def undefined(x):
	udefCount = 0
	total = 0
	for row in x:
		for i in range(len(row)):
			total += 1
			if row[i] == '--undefined--':
				udefCount += 1
	print("Undefined:", udefCount)
	print("Total:", total)
	print("Percent:", float((udefCount/total) * 100))

def main():
	args = parseArgs()
	data, praatHeader = prepData(args.praat, 6)
	stopWords = getStopWords(args.stoplist)
	data = remStopWords(data, stopWords)
	undefined(data)

if __name__ == "__main__":
	main()