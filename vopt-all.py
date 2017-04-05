# Calculates and plots the VoPT separately for each speaker

import matplotlib.pyplot as plt
from matplotlib import rc
import csv
import itertools

# Set font for plots to use CM from LaTeX
def setFont():
	rc('font',**{'family':'serif','serif':['Computer Modern Roman']})
	params = {'backend': 'ps',
		'axes.labelsize': 25,
		'text.fontsize': 25,
		'legend.fontsize': 10,
		'xtick.labelsize': 15,
		'ytick.labelsize': 15,
		'text.usetex': True}
	plt.rcParams.update(params)

# Make and return a list of stop words
def getStopWords():
	stopWords = []
	with open("/Users/Laura/Desktop/Dissertation/data/phonetic_stoplist.txt") as f:
		for line in f:
			line = line.strip().upper()
			stopWords.append(line)
	return stopWords

# Make and return a list of all the words in the data
def getWords():
	words = []
	with open("/Users/Laura/Desktop/Dissertation/data/english/All/All-praat-1.txt") as f:
		reader = csv.reader(f, delimiter = '\t')
		header = next(reader)
		for line in reader:
			words.append(line[3])
	return words

# Remove vowels with the label 0 or 1
def remove01():
	data = []
	with open("/Users/Laura/Desktop/Dissertation/data/english/english-vs-1.txt") as f:
		reader = csv.reader(f, delimiter = '\t')
		header = next(reader)
		for line in reader:
			if line[1] != "0" and line[1] != "1":
				data.append(line)
	return data

# Include only speaker, phonation type, and four pitch tracks
def cutData(data):
	cutData = []
	for line in data:
		speaker = line[0][0:6] # Remove .mat from speaker name
		phonation = line[1]
		strF0 = line[40]
		sF0 = line[41]
		pF0 = line[42]
		shrF0 = line[43]
		newLine = [speaker, phonation, strF0, sF0, pF0, shrF0]
		cutData.append(newLine)
	return cutData

# Remove stop words
def removeStopWords(data, stopWords, words):
	cutData = []
	count = 0
	for line in data:
		if words[count] not in stopWords:
			cutData.append(line)
		count += 1
	return cutData

def calculateVOPT(data):
	newData = []
	for line in data:
		VoPT = 0
		speaker = line[0]
		phonation = line[1]
		strF0 = line[2]
		sF0 = line[3]
		pF0 = line[4]
		shrF0 = line[5]
		tracks = [strF0, sF0, pF0, shrF0]
		pairs = (list(itertools.combinations(tracks, 2)))
		for pair in pairs:
			a = float(pair[0])
			b = float(pair[1])
			diff = abs(a - b)
			VoPT += diff
		newLine = [speaker, phonation, VoPT]
		newData.append(newLine)
	return newData

def makeDict(data):
	speakerDict = {} 
	# Keys are speakers, values are dictionaries with B M C as keys, list of VoPTs as values
	for line in data:
		if line[0] not in speakerDict:
			B = []
			M = []
			C = []
			speakerDict[line[0]] = [B, M, C]
		if line[1] == "B":
			speakerDict[line[0]][0].append(line[2])
		if line[1] == "M":
			speakerDict[line[0]][1].append(line[2])
		if line[1] == "C":
			speakerDict[line[0]][2].append(line[2])
	return speakerDict

def makePlot(speakerDict):
	for key in speakerDict:
		speaker = key
		B = speakerDict[key][0]
		M = speakerDict[key][1]
		C = speakerDict[key][2]
		toPlot = [B, M, C]
		plt.boxplot(toPlot, labels = ["B", "M", "C"])
		#plt.ylim([0,1600])
		plt.title(speaker)
		#plt.show()
		plt.savefig(''.join(["/Users/Laura/Desktop/Dissertation/Dissertation/Appendices/VoPT-all/images/", speaker]), dpi = "figure")
		plt.clf()

def main():
	setFont()
	stopWords = getStopWords()
	words = getWords()
	data = remove01()
	data = cutData(data)
	data = removeStopWords(data, stopWords, words)
	data = calculateVOPT(data)
	speakerDict = makeDict(data)
	makePlot(speakerDict)

if __name__ == "__main__":
	main()