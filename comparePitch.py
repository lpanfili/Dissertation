# python3 comparePitch.py /Users/Laura/Desktop/Dissertation/data/phonetic_stoplist.txt NWF090
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
import math

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

# Returns a list of all the words in the dataset using the Praat data
def getWords(filename):
	words = []
	with open(filename) as f:
		reader = csv.reader(f, delimiter = '\t')
		header = next(reader)
		for line in reader:
			words.append(line[3])
	return words

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
			if line[x] in phonationLabs: # Removes 0 and 1
				data.append(line)
	#print("Vowels, total:", count)
	data = np.array(data)
	return data, header

# Gets data ready for pitch comparisons
def clean(data, stopWords):
	data, VSHeader = prepData(data, 1)
	return data

# Calculates the difference between the four pitch tracks
def meanPitchDiff(data, words, stopWords):
	BDiff = []
	MDiff = []
	CDiff = []
	count = 0
	for row in data:
		count += 1
		if words[count] not in stopWords:
			speaker = row[0][:6]
			VoPT = 0
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
				VoPT += diff
			BDiff, MDiff, CDiff = separatePhonation(VoPT, phonation, BDiff, MDiff, CDiff)
	plotBMC(BDiff, MDiff, CDiff, speaker, "Mean")
	plotMNM(BDiff, MDiff, CDiff, speaker, "Mean")

def meanPitchDiff3(data):
	BDiff = []
	MDiff = []
	CDiff = []
	for row in data:
		speaker = row[0][:6]
		VoPT = 0
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
			rmse = math.sqrt(mean_squared_error(a, b))
			VoPT += rmse
		#allDiff.append([phonation, total])
		BDiff, MDiff, CDiff = separatePhonation(VoPT, phonation, BDiff, MDiff, CDiff)
	plotBMC(BDiff, MDiff, CDiff, speaker, "Mean, Thirds")
	plotMNM(BDiff, MDiff, CDiff, speaker, "Mean, Thirds")

def meanPitchDiff10(data):
	BDiff = []
	MDiff = []
	CDiff = []
	for row in data:
		speaker = row[0][:6]
		VoPT = 0
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
		pairs = (list(itertools.combinations(tracks, 2)))
		for pair in pairs:
			a = pair[0]
			b = pair[1]
			rmse = math.sqrt(mean_squared_error(a, b))
			VoPT += rmse
		#allDiff.append([phonation, total])
		BDiff, MDiff, CDiff = separatePhonation(VoPT, phonation, BDiff, MDiff, CDiff)
	plotBMC(BDiff, MDiff, CDiff, speaker, "Mean, Tenths")
	plotMNM(BDiff, MDiff, CDiff, speaker, "Mean, Tenths")

def pitchDiff3(data, words, stopWords):
	speaker = data[0][0][:6]
	BDiff = []
	MDiff = []
	CDiff = []
	allDiff = []
	straight = []
	snack = []
	praat = []
	shr = []
	seg = data[0][2] # Start with the first segment
	count = 1
	for row in data:
		phonation = row[1]
		if row[2] != seg: # If we're moving on to a different vowel
			count += 1
			if words[count] not in stopWords:
				strF0, sF0, pF0, shrF0 = pick3Points(straight, snack, praat, shr) # Do these for the old one
				VoPT = comparePoints(strF0, sF0, pF0, shrF0)
				BDiff, MDiff, CDiff = separatePhonation(VoPT, phonation, BDiff, MDiff, CDiff)
			straight = [] # Then clear everything
			snack = []
			praat = []
			shr = []
			seg = row[2]
		straight.append(row[40]) # Otherwise just add everything
		snack.append(row[41])
		praat.append(row[42])
		shr.append(row[43])
	if words[count] not in stopWords:
		strF0, sF0, pF0, shrF0 = pick3Points(straight, snack, praat, shr) # Then do all this again for what's left
		VoPT = comparePoints(strF0, sF0, pF0, shrF0)
		BDiff, MDiff, CDiff = separatePhonation(VoPT, phonation, BDiff, MDiff, CDiff)
	plotBMC(BDiff, MDiff, CDiff, speaker, "RMSE3")
	plotMNM(BDiff, MDiff, CDiff, speaker, "RMSE3")

def pitchDiff10(data, words, stopWords):
	speaker = data[0][0][:6]
	BDiff = []
	MDiff = []
	CDiff = []
	allDiff = []
	straight = []
	snack = []
	praat = []
	shr = []
	seg = data[0][2] # Start with the first segment
	count = 1
	for row in data:
		phonation = row[1]
		if row[2] != seg:
			count += 1
			if words[count] not in stopWords:
				strF0, sF0, pF0, shrF0 = pick10Points(straight, snack, praat, shr)
				VoPT = comparePoints(strF0, sF0, pF0, shrF0)
				BDiff, MDiff, CDiff = separatePhonation(VoPT, phonation, BDiff, MDiff, CDiff)
			straight = []
			snack = []
			praat = []
			shr = []
			seg = row[2]
		straight.append(row[40])
		snack.append(row[41])
		praat.append(row[42])
		shr.append(row[43])
	if words[count] not in stopWords:
		strF0, sF0, pF0, shrF0 = pick10Points(straight, snack, praat, shr)
		VoPT = comparePoints(strF0, sF0, pF0, shrF0)
		BDiff, MDiff, CDiff = separatePhonation(VoPT, phonation, BDiff, MDiff, CDiff)
	plotBMC(BDiff, MDiff, CDiff, speaker, "RMSE10")
	plotMNM(BDiff, MDiff, CDiff, speaker, "RMSE10")

def pick3Points(straight, snack, praat, shr):
	#print("All:", len(straight))
	chunk = int(len(straight) / 4)
	#print("Third:", chunk)
	# Make track lists that have THREE points per vowel
	strTrack = [float(straight[chunk]), float(straight[(chunk * 2)]), float(straight[(chunk * 3)])]
	sTrack = [float(snack[chunk]), float(snack[(chunk * 2)]), float(snack[(chunk * 3)])]
	pTrack = [float(praat[chunk]), float(praat[(chunk * 2)]), float(praat[(chunk * 3)])]
	shrTrack = [float(shr[chunk]), float(shr[(chunk * 2)]), float(shr[(chunk * 3)])]
	return strTrack, sTrack, pTrack, shrTrack

def pick10Points(straight, snack, praat, shr):
	#print("All:", len(straight))
	chunk = int(len(straight) / 11)
	#print("Third:", chunk)
	# Make track lists that have TEN points per vowel
	strTrack = [float(straight[chunk]), float(straight[(chunk * 2)]), float(straight[(chunk * 3)]), 
				float(straight[(chunk * 4)]), float(straight[(chunk * 5)]), float(straight[(chunk * 6)]),
				float(straight[(chunk * 7)]), float(straight[(chunk * 8)]), float(straight[(chunk * 9)]), float(straight[(chunk * 10)])]
	sTrack = [float(snack[chunk]), float(snack[(chunk * 2)]), float(snack[(chunk * 3)]),
			float(snack[(chunk * 4)]), float(snack[(chunk * 5)]), float(snack[(chunk * 6)]),
			float(snack[(chunk * 7)]), float(snack[(chunk * 8)]), float(snack[(chunk * 9)]), float(snack[(chunk * 10)])]
	pTrack = [float(praat[chunk]), float(praat[(chunk * 2)]), float(praat[(chunk * 3)]),
			float(praat[(chunk * 4)]), float(praat[(chunk * 5)]), float(praat[(chunk * 6)]),
			float(praat[(chunk * 7)]), float(praat[(chunk * 8)]), float(praat[(chunk * 9)]), float(praat[(chunk * 10)])]
	shrTrack = [float(shr[chunk]), float(shr[(chunk * 2)]), float(shr[(chunk * 3)]),
				float(shr[(chunk * 4)]), float(shr[(chunk * 5)]), float(shr[(chunk * 5)]),
				float(shr[(chunk * 7)]), float(shr[(chunk * 8)]), float(shr[(chunk * 9)]), float(shr[(chunk * 10)])]
	return strTrack, sTrack, pTrack, shrTrack

def comparePoints(straight, snack, praat, shr):
	total = 0
	tracks = [straight, snack, praat, shr]
	pairs = (list(itertools.combinations(tracks, 2)))
	for pair in pairs:
		a = pair[0]
		b = pair[1]
		rmse = math.sqrt(mean_squared_error(a, b))
		total += rmse
	return total

# Adds the VoPT to the appropriate list depending on its vowel's pohonation
# Returns the three lists
def separatePhonation(VoPT, phonation, BDiff, MDiff, CDiff):
	if phonation == "B":
		BDiff.append(VoPT)
	if phonation == "M":
		MDiff.append(VoPT)
	if phonation == "C":
		CDiff.append(VoPT)
	return BDiff, MDiff, CDiff

def plotBMC(BDiff, MDiff, CDiff, speaker, method):
	toPlot = [BDiff, MDiff, CDiff]
	plt.title(''.join([speaker, ", ", method]))
	if method == "Mean":
		plt.ylim([0,1600])
	if method == "RMSE3":
		plt.ylim([0,1800])	
	if method == "RMSE10":
		plt.ylim([0,1800])

	plt.boxplot(toPlot, labels = ["B", "M", "C"])
	#plt.show()
	plt.savefig(''.join(["/Users/Laura/Desktop/Dissertation/VOPT/", speaker, "-", method]), dpi = "figure")
	plt.clf()

def plotMNM(BDiff, MDiff, CDiff, speaker, method):
	NMDiff = BDiff + CDiff
	toPlot = [MDiff, NMDiff]
	plt.title(''.join([speaker, ", ", method]))
	plt.ylim([0,1600])
	plt.boxplot(toPlot, labels = ["Modal", "Non-Modal"])
	plt.savefig(''.join(["/Users/Laura/Desktop/Dissertation/VOPT/", speaker, "-", method, "MNM"]), dpi = "figure")
	plt.clf()
	
def main():
	setFont()
	args = parseArgs()
	speaker = args.speaker
	stopWords = getStopWords(args.stoplist)
	praat = ''.join(["/Users/Laura/Desktop/Dissertation/data/english/", speaker, "/", speaker, "-praat-1.txt"])
	words = getWords(praat)
	one = ''.join(["/Users/Laura/Desktop/Dissertation/data/english/", speaker, "/", speaker, "-vs-1.txt"])
	#three = ''.join(["/Users/Laura/Desktop/Dissertation/data/", speaker, "/", speaker, "-vs-3.txt"])
	#ten = ''.join(["/Users/Laura/Desktop/Dissertation/data", speaker, "/", speaker, "-vs-10.txt"])
	big = ''.join(["/Users/Laura/Desktop/Dissertation/data/english/", speaker, "/", speaker, "-vs-all.txt"])
	dataOne = clean(one, stopWords)
	#dataThree = clean(three, stopWords)
	#dataTen = clean(ten, stopWords)
	dataBig = clean(big, stopWords)
	meanPitchDiff(dataOne, words, stopWords)
	#meanPitchDiff3(dataThree)
	#meanPitchDiff10(dataTen)
	pitchDiff3(dataBig, words, stopWords)
	pitchDiff10(dataBig, words, stopWords)

if __name__ == "__main__":
	main()