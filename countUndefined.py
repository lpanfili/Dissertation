# python3 countUndefined.py /Users/Laura/Desktop/Dissertation/test-data/phonetic_stoplist.txt /Users/Laura/Desktop/Dissertation/test-data2/results.txt
import csv
#import random
#from sklearn import svm, metrics
#from sklearn.svm import SVC
#from sklearn.metrics import precision_recall_fscore_support, mean_squared_error
#from sklearn.model_selection import cross_val_predict, StratifiedShuffleSplit, GridSearchCV
#from matplotlib import rc
#from matplotlib.colors import Normalize
#import matplotlib.pyplot as plt
import numpy as np
#import itertools
import argparse
#import statistics
#import pandas as pd
#import sys
#import itertools
#from statsmodels.tools import tools
#from statsmodels.discrete.discrete_model import MNLogit

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
	for row in x:
		for i in range(len(row)):
			if row[i] == '--undefined--':
				udefCount += 1
	print("Undefined:", udefCount)

# Runs z normalization over whatever features listed
def zNormFeatures(x, speakerList, featureList):
	for feature in featureList:
		zNorm(x, speakerList, feature)

def main():
	args = parseArgs()
	data, praatHeader = prepData(args.praat, 6)
	stopWords = getStopWords(args.stoplist)
	data = remStopWords(data, stopWords)
	undefined(data)

if __name__ == "__main__":
	main()