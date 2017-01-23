import csv
import random
from sklearn import svm, metrics
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_predict, StratifiedShuffleSplit, GridSearchCV
from matplotlib.colors import Normalize
import matplotlib.pyplot as plt
import numpy as np
import itertools
import argparse
import statistics

# Make a list from a file of words
# Use this for the stop words to be excluded
def makeList(filename):
	stopWords = []
	with open(filename) as f:
		for line in f:
			line = line.strip().upper()
			stopWords.append(line)
	return stopWords

# Picks which of the features to include
# Replaces "undefined" with "1"
# Returns a tuple of features, label
def pickFeatures(line):
	label = line["phonation"]
	features = [line["gridfile"], line["jitter_ddp"], line["jitter_loc"], line["jitter_loc_abs"], line["jitter_rap"], line["jitter_ppq5"], 
			line["shimmer_loc"], line["shimmer_apq3"], line["shimmer_apq5"], line["shimmer_apq11"], line["F0"]]
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
		reader = csv.DictReader(f)
		for line in reader:
			if line["phonation"] in phonationLabs and line["word_label"] not in stopWords:
				dataPoints.append(pickFeatures(line))
	return dataPoints

# Inputs should be path for stop list and path for data
def parseArgs():
	parser = argparse.ArgumentParser()
	parser.add_argument("stoplist")
	parser.add_argument("data")
	return parser.parse_args()

# take the data that will go in the model plus the speaker id
# znormalize all features within speaker
# go through all the data and make speaker specific lists
# key = speaker; value = list of tuple of features + label
def zNorm(data):
	bySpeaker = {}
	for i in data:
		if i[0][0] not in bySpeaker:
			bySpeaker[i[0][0]] = []
		bySpeaker[i[0][0]].append(i)
	for datapoints in bySpeaker.values():
		numColumns = len(datapoints[0][0])
		for i in range(1,numColumns):
			featureVals = []
			for datapoint in datapoints:
				if datapoint[0][i] != "--undefined--":
					featureVals.append(float(datapoint[0][i]))
			featureMean = statistics.mean(featureVals)
			featureSTD = statistics.stdev(featureVals)
			for datapoint in datapoints:
				if datapoint[0][i] != "--undefined--":
					datapoint[0][i] = (float(datapoint[0][i]) - featureMean) / featureSTD
				else:
					datapoint[0][i] = 1
	# Get rid of speaker IDs
	for datapoint in data:
		datapoint[0].pop(0)

	print(data)

def main():
	args = parseArgs()
	stopWords = makeList(args.stoplist)
	dataPoints = getData(args.data, stopWords)
	zNorm(dataPoints)
	# Remove speaker name
	#stopWords = makeList("/Users/Laura/Desktop/Dissertation/test-data/phonetic_stoplist.txt")
	#dataPoints = getData("/Users/Laura/Desktop/Dissertation/test-data/results.txt", stopWords)
	random.shuffle(dataPoints)
	trainx, trainy = zip(*dataPoints)

	#clf = svm.SVC(kernel = 'rbf', C = .1, gamma = 0.001)
	#clf = svm.SVC(kernel = 'rbf')
	clf = svm.SVC()
	predictedy = cross_val_predict(clf, trainx, trainy, cv=10)
	print(metrics.accuracy_score(trainy, predictedy))
	cnf_matrix = confusionMatrix(trainy, predictedy)               
	plt.figure()
	plot_confusion_matrix(cnf_matrix, classes = ["B", "M", "C"], title = 'Confusion Matrix')
	plt.show()
"""
# Grid search
	C_range = np.logspace(-3, 2, 6)
	gamma_range = np.logspace(-3, 2, 6)
	param_grid = dict(gamma=gamma_range, C=C_range)
	cv = StratifiedShuffleSplit(n_splits=10, test_size=0.1, random_state=42)
	grid = GridSearchCV(clf, param_grid=param_grid, cv=cv)
	grid.fit(trainx, trainy)
	print("The best parameters are %s with a score of %0.2f"
      % (grid.best_params_, grid.best_score_))
"""
if __name__ == "__main__":
	main()
