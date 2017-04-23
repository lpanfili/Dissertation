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
			if line[6] != "0" and line[6] != "1":
				if line[5] not in stopWords:
					newline = [line[6], float(line[7]), float(line[8]), float(line[9]), 
					float(line[10]), float(line[11]), float(line[12]), 
					float(line[13]), float(line[14]), float(line[15]), float(line[16]), 
					float(line[17]), float(line[18]), float(line[19]), float(line[20]),
					float(line[21]), float(line[22]), float(line[23]), float(line[24]), 
					float(line[25]), float(line[26]), float(line[27]), float(line[28]), 
					float(line[29]), float(line[30])]
					newdata.append(newline)
	return newdata

def makeDict(data):
	for i in range(1,25):
		B = []
		M = []
		C = []
		for row in data:
			if row[0] == "B":
				B.append(row[i])
			if row[0] == "M":
				M.append(row[i])
			if row[0] == "C":
				C.append(row[i])
		plot(B, M, C, i)


def plot(B, M, C, i):
	toPlot = [B, M, C]
	plt.boxplot(toPlot, labels = ["B", "M", "C"])
	number = "MFCC " + str(i)
	plt.title(number)
	#plt.show()
	plt.savefig(''.join(["/Users/Laura/Desktop/Dissertation/Code/mfcc/mfcc-means/", str(i)]), dpi = "figure")
	plt.clf()


def main():
	setFont()
	stopWords = getStopWords()
	data = cleanData(stopWords)
	makeDict(data)

if __name__ == "__main__":
	main()