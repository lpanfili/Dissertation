import matplotlib.pyplot as plt
from matplotlib import rc
import csv
import numpy as np
from numpy import std

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

# Make new data set without stop words or vowels tagged as 0 or 1
def cleanData(stopWords):
	newData = []
	with open("/Users/Laura/Desktop/Dissertation/Code/mfcc/zapotec-results.txt") as f:
		reader = csv.reader(f, delimiter = '\t')
		header = next(reader)
		for line in reader:
			if line[6] == "B" or line[6] == "M" or line[6] == "C":
				if line[5] not in stopWords:
					vowel = line[4]
					if ord(vowel[-1]) >=48 and ord(vowel[-1]) <=50:
						line[4] = vowel[:-1]
					newData.append(line)
		return newData


def getMeanData(data):
	meanData = []
	for line in data:
		newline = [line[6], line[4], float(line[7]), float(line[8]), float(line[9]), 
			float(line[10]), float(line[11]), float(line[12]), float(line[13]), 
			float(line[14]), float(line[15]), float(line[16]), float(line[17]), 
			float(line[18]), float(line[19]), float(line[20]), float(line[21]), 
			float(line[22]), float(line[23]), float(line[24]), float(line[25]), 
			float(line[26]), float(line[27]), float(line[28]), float(line[29]), 
			float(line[30])]
		meanData.append(newline)
	return meanData

def getStddevData(data):
	stddevData = []
	for line in data:
		newline = [line[6], line[4], float(line[31]), float(line[32]), float(line[33]), 
			float(line[34]), float(line[35]), float(line[36]), float(line[37]), 
			float(line[38]), float(line[39]), float(line[40]), float(line[41]), 
			float(line[42]), float(line[43]), float(line[44]), float(line[45]), 
			float(line[46]), float(line[47]), float(line[48]), float(line[49]), 
			float(line[50]), float(line[51]), float(line[52]), float(line[53]), 
			float(line[54])]
		stddevData.append(newline)
	return stddevData
					

def makeMeanDict(data):
	mfccDict = {} # Dictionary, keys are vowels, vals are dicts with CCs as keys, BMC lists as vals
	for row in data:
		vowel = row[1]
		if vowel not in mfccDict:
			mfccDict[vowel] = {}
			for i in range(2,26):
				mfccDict[vowel][i] = {"B":[], "M":[], "C":[]}
		for i in range(2,26):
			mfccDict[vowel][i][row[0]].append(row[i])
	for vowel in mfccDict:
		for	cc in mfccDict[vowel]:
			meanPlot(mfccDict[vowel][cc]["B"], mfccDict[vowel][cc]["M"], mfccDict[vowel][cc]["C"], cc, vowel)

def verticalSTDDV(data):
	mfccDict = {"B":[], "M":[], "C":[]} # Dictionary, keys are BMC, vals are CC stdvs
	ccList = [] # Contains the mean of CCs 1-24
	for row in data:
		for i in range(2,26):
			ccList.append(row[i])
		stddev = np.std(ccList)
		mfccDict[row[0]].append(stddev)
	toPlot = [mfccDict["B"], mfccDict["M"], mfccDict["C"]]
	plt.boxplot(toPlot, labels = ["B", "M", "C"])
	plt.title("StdDv, 24 MFCCs")
	#plt.show()
	plt.savefig("/Users/Laura/Desktop/Dissertation/Code/mfcc/stddv-vertical", dpi = "figure")
	plt.clf()
		

"""
def makeStddvDict(data):
	mfccDict = {}
	for row in data:
		vowel = row[1]
		if vowel not in mfccDict:
			mfccDict[vowel] = {}
			for i in range(2,26):
				mfccDict[vowel][i] = {"B":[], "M":[], "C":[]}
		for i in range(2,26):
			mfccDict[vowel][i][row[0]].append(row[i])
	for vowel in mfccDict:
		for	cc in mfccDict[vowel]:
			stddvPlot(mfccDict[vowel][cc]["B"], mfccDict[vowel][cc]["M"], mfccDict[vowel][cc]["C"], cc, vowel)
"""
def makeStddvDict(data):
	mfccDict = {}
	for i in range(2,26):
		mfccDict[i] = {"B":[], "M":[], "C":[]}
	for row in data:
		for j in range(2,26):
			mfccDict[j][row[0]].append(row[j])
	for	cc in mfccDict:
		stddvPlot(mfccDict[cc]["B"], mfccDict[cc]["M"], mfccDict[cc]["C"], cc)

def meanPlot(B, M, C, i, vowel):
	toPlot = [B, M, C]
	plt.boxplot(toPlot, labels = ["B", "M", "C"])
	number = "MFCC " + str(i - 1) + ", " + vowel
	plt.title(number)
	#plt.show()
	#plt.savefig(''.join(["/Users/Laura/Desktop/Dissertation/Code/mfcc/mfcc-means/", str(i - 1), vowel]), dpi = "figure")
	plt.clf()

"""
def stddvPlot(B, M, C, i, vowel):
	toPlot = [B, M, C]
	plt.boxplot(toPlot, labels = ["B", "M", "C"])
	number = "Std Dv of MFCC " + str(i - 1) + ", " + vowel
	plt.title(number)
	#plt.show()
	plt.savefig(''.join(["/Users/Laura/Desktop/Dissertation/Code/mfcc/mfcc-stddv/", str(i - 1), "-std-", vowel]), dpi = "figure")
	plt.clf()
	"""

def stddvPlot(B, M, C, i):
	toPlot = [B, M, C]
	plt.boxplot(toPlot, labels = ["B", "M", "C"])
	number = "Std Dv of MFCC " + str(i - 1)
	plt.title(number)
	#plt.show()
	plt.savefig(''.join(["/Users/Laura/Desktop/Dissertation/Code/mfcc/zap-mfcc-stddv/", str(i - 1), "-std"]), dpi = "figure")
	plt.clf()

def main():
	setFont()
	stopWords = getStopWords()
	data = cleanData(stopWords)
	meanData = getMeanData(data)
	#stddevData = getStddevData(data)
	#makeMeanDict(meanData)
	#makeStddvDict(stddevData)
	verticalSTDDV(meanData)

if __name__ == "__main__":
	main()