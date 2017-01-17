import csv
import random
from sklearn import svm, metrics
from sklearn.model_selection import cross_val_predict
import matplotlib.pyplot as plt
import numpy as np
import itertools

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

# Picks which of the features to include
# Replaces "undefined" with "1"
# Returns a tuple of features, label
def pickFeatures(line):
	label = line["phonation"]
	features = [line["jitter_ddp"], line["jitter_loc"], line["jitter_loc_abs"], line["jitter_rap"], line["jitter_ppq5"], 
			line["shimmer_loc"], line["shimmer_apq3"], line["shimmer_apq5"], line["shimmer_apq11"], line["shimmer_dda"]]
	features = [1 if i == "--undefined--" else i for i in features]
	return features, label

# Returns confusion matrix
def confusionMatrix(testy, predictedy):
	return metrics.confusion_matrix(testy, predictedy, labels = ["B", "M", "C"])

# Plots confusion matrix
def plot_confusion_matrix(cm, classes,
                          normalize = False,
                          title = "Confusion matrix",
                          cmap = plt.cm.Blues):
    plt.imshow(cm, interpolation = 'nearest', cmap = cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color = "white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

# Reads from a file
# Writes only lines that have B M C as phonation and aren't stopwords
# Calls pickFeatures to add only specific features to the list
def getData(filename, stopWords):
	dataPoints = []
	phonationLabs = ["B", "C", "M"]
	with open(filename) as f:
		reader = csv.DictReader(f, delimiter = "\t")
		for line in reader:
			if line["phonation"] in phonationLabs and line["word_label"] not in stopWords:
				dataPoints.append(pickFeatures(line))
	return dataPoints

def printAccuracy(testy, predictedy):
	numCorrect = 0
	numTotal = 0
	for i in range(len(predictedy)):
		numTotal += 1
		if predictedy[i] == testy[i]:
			numCorrect += 1
	print(numCorrect, " / ", numTotal, " = ", 1.0 * numCorrect/numTotal)

def main():
	stopWords = makeList("/Users/Laura/Desktop/Dissertation/test-data/phonetic_stoplist.txt")
	dataPoints = getData("/Users/Laura/Desktop/Dissertation/test-data/results.txt", stopWords)
	random.shuffle(dataPoints)
	trainx, trainy = zip(*dataPoints)
	clf = svm.SVC()
	predictedy = cross_val_predict(clf, trainx, trainy, cv=10)
	print(metrics.accuracy_score(trainy, predictedy))
	cnf_matrix = confusionMatrix(trainy, predictedy)               
	plt.figure()
	plot_confusion_matrix(cnf_matrix, classes = ["B", "M", "C"], title = 'Confusion Matrix')
	plt.show()

if __name__ == "__main__":
	main()
