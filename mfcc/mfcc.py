import matplotlib.pyplot as plt
from matplotlib import rc
import csv

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
	newdata = []
	with open("/Users/Laura/Desktop/Dissertation/Code/mfcc/results-mfcc.txt") as f:
		reader = csv.reader(f, delimiter = '\t')
		header = next(reader)
		for line in reader:
			if line[6] == "B" or line[6] == "M" or line[6] == "C":
				if line[5] not in stopWords:
					vowel = line[4]
					if ord(vowel[-1]) >=48 and ord(vowel[-1]) <=50:
						vowel = vowel[:-1]
					newline = [line[6], vowel, float(line[7]), float(line[8]), float(line[9]), 
					float(line[10]), float(line[11]), float(line[12]), 
					float(line[13]), float(line[14]), float(line[15]), float(line[16]), 
					float(line[17]), float(line[18]), float(line[19]), float(line[20]),
					float(line[21]), float(line[22]), float(line[23]), float(line[24]), 
					float(line[25]), float(line[26]), float(line[27]), float(line[28]), 
					float(line[29]), float(line[30])]
					newdata.append(newline)
	return newdata


def makeDict(data):
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
			plot(mfccDict[vowel][cc]["B"], mfccDict[vowel][cc]["M"], mfccDict[vowel][cc]["C"], cc, vowel)


def plot(B, M, C, i, vowel):
	toPlot = [B, M, C]
	plt.boxplot(toPlot, labels = ["B", "M", "C"])
	number = "MFCC " + str(i - 1) + ", " + vowel
	plt.title(number)
	#plt.show()
	plt.savefig(''.join(["/Users/Laura/Desktop/Dissertation/Code/mfcc/mfcc-means/", str(i - 1), vowel]), dpi = "figure")
	plt.clf()

def main():
	setFont()
	stopWords = getStopWords()
	data = cleanData(stopWords)
	makeDict(data)

if __name__ == "__main__":
	main()