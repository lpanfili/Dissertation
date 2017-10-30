# For one type of sampling (user-specified)
# Sort the feature weights for a language by magnitude
# Outputs a CSV that can be dumped into LaTeX

import pandas as pd
import csv
import argparse

def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('lg', type = str, help = 'the three digit code for the language in question')
	parser.add_argument('datatype', type = str, help = 'imb (imbalanced) or rs (resampled)')
	return parser.parse_args()

# Get path to csv containing feature weights
def get_path(lg, datatype):
	RFpath = "/Users/Laura/Desktop/Dissertation/data/weights/RF-" + lg + "-" + datatype + ".csv"
	SVMpath = "/Users/Laura/Desktop/Dissertation/data/weights/SVM-" + lg + "-" + datatype + ".csv"
	return RFpath, SVMpath

def sortSVM(path, lg):
	weights = pd.read_csv(path)
	weights = weights.set_index('latex-feat')
	if lg == 'guj':
		weights['BMabs'] = weights['BM'].abs()
		weights = weights.sort_values(by = 'BMabs', ascending = False)
		weights = weights[['BM']].copy().reset_index()
		weights.rename(columns = {'BM': 'BM-SVM-weight'}, inplace = True)
	elif lg == 'cmn':
		weights['CMabs'] = weights['CM'].abs()
		weights = weights.sort_values(by = 'CMabs', ascending = False)
		weights = weights[['CM']].copy().reset_index()
		weights.rename(columns = {'CM': 'CM-SVM-weight'}, inplace = True)
	else:
		weights['BCabs'] = weights['BC'].abs()
		weights['BMabs'] = weights['BM'].abs()
		weights['CMabs'] = weights['CM'].abs()
		weights = weights.sort_values(by = 'BCabs', ascending = False)
		BC = weights[['BC']].copy().reset_index()
		weights = weights.sort_values(by = 'BMabs', ascending = False)
		BM = weights[['BM']].copy().reset_index()
		weights = weights.sort_values(by = 'CMabs', ascending = False)
		CM = weights[['CM']].copy().reset_index()
		weights = pd.concat([BC, BM, CM], axis=1)
		weights.rename(columns = {'BC': 'BC-SVM-weight', 'BM': 'BM-SVM-weight', 'CM': 'CM-SVM-weight'}, inplace = True)
	weights = weights.round(decimals = 3)
	return weights

def sortRF(path, lg):
	weights = pd.read_csv(path)
	weights = weights.set_index('latex-feat')
	weights = weights.round(decimals = 3)
	weights = weights.sort_values(by = 'weight', ascending = False)
	weights.rename(columns = {'weight': 'RF-weight'}, inplace = True)
	weights = weights.reset_index()
	return weights


def main():
	args = parse_args()
	RFpath, SVMpath = get_path(args.lg, args.datatype)
	SVMweights = sortSVM(SVMpath, args.lg)
	RFWeights = sortRF(RFpath, args.lg)
	weights = pd.concat([SVMweights, RFWeights], axis = 1)
	path = "/Users/Laura/Desktop/Dissertation/data/weights/sortedWeights-" + args.lg + "-" + args.datatype + ".csv"
	weights.to_csv(path)

if __name__ == "__main__":
	main()