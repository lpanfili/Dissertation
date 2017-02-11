# python3 runClass2.py /Users/Laura/Desktop/Dissertation/test-data/phonetic_stoplist.txt /Users/Laura/Desktop/Dissertation/NWF089/NWF089-praat.txt /Users/Laura/Desktop/Dissertation/NWF089/NWF089-VS.txt
import csv
import random
from sklearn import svm, metrics
from sklearn.svm import SVC
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import cross_val_predict, StratifiedShuffleSplit, GridSearchCV
from matplotlib.colors import Normalize
import matplotlib.pyplot as plt
import numpy as np
import itertools
import argparse
import statistics
import pandas as pd
import sys

# Inputs should be path for stop list and path for data files (Praat, VS)
def parseArgs():
	parser = argparse.ArgumentParser()
	parser.add_argument("stoplist")
	parser.add_argument("praat")
	parser.add_argument("VS")
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
def prepData(filename, x):
	data = []
	phonationLabs = ["B", "C", "M"]
	with open(filename) as f:
		reader = csv.reader(f)
		header = next(reader)
		for line in reader:
			if line[x] in phonationLabs:
				data.append(line)
	data = np.array(data)
	return data, header

# Combines Praat and VS data
# At this point, 0 and 1 have been removed but stop words have not; undefined remains
def combineData(praat, VS):
	data = np.concatenate((praat, VS), axis = 1)
	return data

# Removes stop words
def remStopWords(data, stopWords):
	remove = []
	for row in data:
		remove.append(row[5] not in stopWords)
	remove = np.array(remove)
	data = data[remove]
	np.savetxt('out.txt', data, fmt = '%s')
	return data

# Returns a new data set with only the features to be included
def pickFeatures(data):
	xlist = [] # x = features
	y = [] # y = labels
	for row in data:
		# local jitter, CPP mean, energy mean, HNR25, strF0
		xline = [row[8], row[82], row[83], row[86], row[104]] 
		xlist.append(xline) 
		y.append(row[6])
	x = np.array(xlist)
	return x, y

# Count measured that are "undefined"
# Replace "undefined" with 1
def undefined(x):
	udefCount = 0
	for row in x:
		for i in range(len(row)):
			if row[i] == '--undefined--':
				udefCount += 1
				row[i] = 1 # Change this line to change what --undefined-- becomes
	print("Undefined:", udefCount)
	return x

def confusionMatrix(trainy, predictedy):
	y_actu = pd.Series(trainy, name='Actual')
	y_pred = pd.Series(predictedy, name='Predicted')
	df_confusion = pd.crosstab(y_actu, y_pred, rownames=['Actual'], colnames=['Predicted'], margins=True)
	return df_confusion

def prfs(trainy, predictedy):
	y_true = np.array(trainy)
	y_pred = np.array(predictedy)
	return precision_recall_fscore_support(y_true, y_pred, average='weighted')

def main():
	args = parseArgs()
	praatData, praatHeader = prepData(args.praat, 6)
	VSData, VSHeader = prepData(args.VS, 1)
	data = combineData(praatData, VSData)
	stopWords = getStopWords(args.stoplist)
	data = remStopWords(data, stopWords)
	x, y = pickFeatures(data)
	x = undefined(x)
	# Z-normalize
	# Shuffle?
	clf = svm.SVC()
	predictedy = cross_val_predict(clf, x, y, cv = 10)
	p, r, f, s = prfs(y, predictedy)
	print("Precision: ", p)   
	print("Recall: ", r)
	print("F-score: ", f)  
	print("Accuracy:", metrics.accuracy_score(y, predictedy))
	print("-------------------------")
	print(confusionMatrix(y, predictedy))

"""	
# Grid search - not tested yet in smaller version
	C_range = np.logspace(-3, 2, 6)
	gamma_range = np.logspace(-3, 2, 6)
	param_grid = dict(gamma=gamma_range, C=C_range)
	cv = StratifiedShuffleSplit(n_splits=10, test_size=0.1, random_state=42)
	grid = GridSearchCV(clf, param_grid=param_grid, cv=cv)
	grid.fit(x, y)
	print("The best parameters are %s with a score of %0.2f"
      % (grid.best_params_, grid.best_score_))
"""

if __name__ == "__main__":
	main()