import csv
import random
from sklearn import svm

# Make a list from a file of words
# Use this for the stop words to be excluded
def makeList(filename):
	stopWords = []
	with open(filename) as f:
		for line in f:
			line = line.strip().upper()
			stopWords.append(line)
	return stopWords

# Removes tokens labeled as "0" or "1"
# Removes tokens belonging to stop words
# Rewrites into results file
def remove(filename, results, stopList):
	stopWords = makeList(stopList)
	with open(filename) as f:
		with open(results, "w") as rf:
			reader = csv.DictReader(f, delimiter = "\t")
			writer = csv.DictWriter(rf, reader.fieldnames)
			writer.writeheader()
			for line in reader:
				if line["phonation"] != "0" and line["phonation"] != "1" and line["word_label"] not in stopWords:
					writer.writerow(line)

def pickFeatures(line):
	label = line["phonation"]
	features = [line["jitter_ddp"], line["jitter_loc"], line["jitter_loc_abs"], line["jitter_rap"], line["jitter_ppq5"], 
			line["shimmer_loc"], line["shimmer_apq3"], line["shimmer_apq5"], line["shimmer_apq11"], line["shimmer_dda"]]
	features = [1 if i == "--undefined--" else i for i in features] # list comprehension, convert undefinds to 1
	return features, label


def main():
	filename = "/Users/Laura/Desktop/Dissertation/test-data/results.txt"
	stopWords = makeList("/Users/Laura/Desktop/Dissertation/test-data/phonetic_stoplist.txt")
	dataPoints = []
	phonationLabs = ["B", "C", "M"]
	with open(filename) as f:
		reader = csv.DictReader(f, delimiter = "\t")
		for line in reader:
			if line["phonation"] in phonationLabs and line["word_label"] not in stopWords:
				dataPoints.append(pickFeatures(line))
	random.shuffle(dataPoints)
	total = len(dataPoints)
	cutoff = int(total * .8)
	train = dataPoints[:cutoff]
	test = dataPoints[cutoff:]
	resultsTrain = "/Users/Laura/Desktop/Dissertation/test-data/trainingdata.csv"
	with open(resultsTrain, "w") as f:
		writer = csv.writer(f)
		for i in train:
			writer.writerow(i)
	trainx, trainy = zip(*train)
	testx, testy = zip(*test)
	clf = svm.SVC()
	clf.fit(trainx, trainy) 
	predictedy = clf.predict(testx)
	numCorrect = 0
	numTotal = 0
	for i in range(len(predictedy)):
		numTotal += 1
		if predictedy[i] == testy[i]:
			numCorrect += 1
	print(numCorrect, " / ", numTotal, " = ", 1.0 * numCorrect/numTotal)


	# Open the CSV with raw data

	# For each line in that CSV, process and turn it into x, y pair
	# then add to list of (x, y) pairs

	# Randomly select 80?% of the pairs to be train data, use rest as test
	# Split the pairs into separate lists

	# Train machine learning algorithm on the train data

	# Run ML model on the test data and report results

	#remove("/Users/Laura/Desktop/Dissertation/test-data/results.txt", 
	#"/Users/Laura/Desktop/Dissertation/test-data/results-clean.txt", 
	#"/Users/Laura/Desktop/Dissertation/test-data/phonetic_stoplist.txt")

if __name__ == "__main__":
	main()
