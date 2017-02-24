# python3 comparePitch.py /Users/Laura/Desktop/Dissertation/test-data/phonetic_stoplist.txt NWF090
import csv
import random
from sklearn import svm, metrics
from sklearn.svm import SVC
from sklearn.metrics import mean_squared_error
from matplotlib import rc
import matplotlib.pyplot as plt
import numpy as np
import itertools
import argparse
import statistics
import pandas as pd
import sys
import itertools
from statsmodels.tools import tools

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

# Inputs should be path for stop list and path for data file
def parseArgs():
	parser = argparse.ArgumentParser()
	parser.add_argument("stoplist")
	parser.add_argument("speaker")
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
		reader = csv.reader(f, delimiter = '\t')
		header = next(reader)
		for line in reader:
			count += 1
			if line[x] in phonationLabs:
				data.append(line)
	print("Vowels, total:", count)
	data = np.array(data)
	return data, header

# Removes stop words
def remStopWords(data, stopWords):
	remove = []
	for row in data:
		remove.append(row[5] not in stopWords)
	remove = np.array(remove)
	data = data[remove]
	print("Vowels, without stop words, 0, or 1:", len(data))
	return data

# Gets data ready for pitch comparisons
def clean(data, stopWords):
	data, VSHeader = prepData(data, 1)
	data = remStopWords(data, stopWords)
	return data

# Calculates the difference between the four pitch tracks
def meanPitchDiff(data):
	BDiff = []
	MDiff = []
	CDiff = []
	allDiff = []
	for row in data:
		speaker = row[0][:6]
		total = 0
		phonation = row[1]
		strF0 = row[40]
		sF0 = row[41]
		pF0 = row[42]
		shrF0 = row[43]
		tracks = [strF0, sF0, pF0, shrF0]
		pairs = (list(itertools.combinations(tracks, 2)))
		for pair in pairs:
			a = float(pair[0])
			b = float(pair[1])
			diff = abs(a - b)
			total += diff
		allDiff.append([phonation, total])
		if phonation == "B":
			BDiff.append(total)
		if phonation == "M":
			MDiff.append(total)
		if phonation == "C":
			CDiff.append(total)
	toPlot = [BDiff, MDiff, CDiff]
	plt.title(speaker)
	plt.ylim([0,1600])
	plt.boxplot(toPlot, labels = ["B", "M", "C"])
	plt.show()
	# Compare modal to non-modal
	#NMDiff = BDiff + CDiff
	#toPlot = [MDiff, NMDiff]
	#plt.title(speaker)
	#plt.boxplot(toPlot, labels = ["Modal", "Non-Modal"])
	#plt.show()

def pitchDiff3(data):
	BDiff3 = []
	MDiff3 = []
	CDiff3 = []
	#allDiff = []
	for row in data:
		speaker = row[0][:6]
		total = 0
		phonation = row[1]
		strF0 = [float(row[149]), float(row[150]), float(row[151])]
		sF0 = [float(row[153]), float(row[154]), float(row[155])]
		pF0 = [float(row[157]), float(row[158]), float(row[159])]
		shrF0 = [float(row[161]), float(row[162]), float(row[163])]
		tracks = [strF0, sF0, pF0, shrF0]
		tracks = [strF0, sF0, pF0, shrF0]
		pairs = (list(itertools.combinations(tracks, 2)))
		for pair in pairs:
			a = pair[0]
			b = pair[1]
			rmse = mean_squared_error(a, b)
			total += rmse
		#allDiff.append([phonation, total])
		if phonation == "B":
			BDiff3.append(total)
		if phonation == "M":
			MDiff3.append(total)
		if phonation == "C":
			CDiff3.append(total)
	toPlot = [BDiff3, MDiff3, CDiff3]
	plt.title(speaker)
	plt.ylim([0,600000])
	plt.boxplot(toPlot, labels = ["B", "M", "C"])
	plt.show()
	#NMDiff3 = BDiff3 + CDiff3
	#toPlot = [MDiff3, NMDiff3]
	#plt.title(speaker)
	#plt.boxplot(toPlot, labels = ["Modal", "Non-Modal"])
	#plt.show()

def pitchDiff10(data):
	BDiff10 = []
	MDiff10 = []
	CDiff10 = []
	for row in data:
		speaker = row[0][:6]
		total = 0
		phonation = row[1]
		strF0 = [float(row[401]), float(row[402]), float(row[403]), float(row[404]), 
				float(row[405]), float(row[406]), float(row[407]), float(row[408]), 
				float(row[409]), float(row[410])]
		sF0 = [float(row[412]), float(row[413]), float(row[414]), float(row[415]), 
				float(row[416]), float(row[417]), float(row[418]), float(row[419]), 
				float(row[420]), float(row[421])]
		pF0 = [float(row[423]), float(row[424]), float(row[425]), float(row[426]), 
				float(row[427]), float(row[428]), float(row[429]), float(row[430]), 
				float(row[431]), float(row[432])]
		shrF0 = [float(row[434]), float(row[435]), float(row[436]), float(row[437]), 
				float(row[438]), float(row[439]), float(row[440]), float(row[441]), 
				float(row[442]), float(row[443])]
		tracks = [strF0, sF0, pF0, shrF0]
		tracks = [strF0, sF0, pF0, shrF0]
		pairs = (list(itertools.combinations(tracks, 2)))
		for pair in pairs:
			a = pair[0]
			b = pair[1]
			rmse = mean_squared_error(a, b)
			total += rmse
		#allDiff.append([phonation, total])
		if phonation == "B":
			BDiff10.append(total)
		if phonation == "M":
			MDiff10.append(total)
		if phonation == "C":
			CDiff10.append(total)
	toPlot = [BDiff10, MDiff10, CDiff10]
	plt.title(speaker)
	plt.ylim([0,600000])
	plt.boxplot(toPlot, labels = ["B", "M", "C"])
	plt.show()
	#NMDiff10 = BDiff10 + CDiff10
	#toPlot = [MDiff10, NMDiff10]
	#plt.title(speaker)
	#plt.boxplot(toPlot, labels = ["Modal", "Non-Modal"])
	#plt.show()

def main():
	setFont()
	args = parseArgs()
	speaker = args.speaker
	stopWords = getStopWords(args.stoplist)
	one = ''.join(["/Users/Laura/Desktop/Dissertation/Code/vopt/", speaker, "-1.txt"])
	three = ''.join(["/Users/Laura/Desktop/Dissertation/Code/vopt/", speaker, "-3.txt"])
	ten = ''.join(["/Users/Laura/Desktop/Dissertation/Code/vopt/", speaker, "-10.txt"])
	dataOne = clean(one, stopWords)
	dataThree = clean(three, stopWords)
	dataTen = clean(ten, stopWords)
	meanPitchDiff(dataOne)
	pitchDiff3(dataThree)
	pitchDiff10(dataTen)

if __name__ == "__main__":
	main()