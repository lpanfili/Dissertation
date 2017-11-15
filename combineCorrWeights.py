# Combines the sorted correlations and the weights
# For a given language and a given sample type (imb or rs)
# Into a single data frame PER CONTRAST
# Outputs a CSV that can be dumped into LaTeX

import pandas as pd
import csv
import argparse

# Requires lg code and sample type as args
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('lg', type = str, help = 'the three digit code for the language in question')
    parser.add_argument('datatype', type = str, help = 'imb (imbalanced) or rs (resampled)')
    return parser.parse_args()

def get_paths(lg, datatype):
	corr = "/Users/Laura/Desktop/Dissertation/data/correlations/sortedCorr-" + lg + "-" + datatype + ".csv"
	weights = "/Users/Laura/Desktop/Dissertation/data/weights/sortedWeights-" + lg + "-" + datatype + ".csv"
	return corr, weights

def combine(corr, weights, lg, datatype):
	corr = pd.read_csv(corr)
	weights = pd.read_csv(weights)
	if lg == 'guj':
		BMCorr = corr[['latex-feature.2','BM-corr']]
		BMWeight = weights[['latex-feat', 'BM-SVM-weight']]
		RFWeight = weights[['latex-feat.1', 'RF-weight']]
		BM = pd.concat([BMCorr, BMWeight], axis = 1)
		combined = pd.concat([BM, RFWeight], axis = 1)
	elif lg == 'cmn':
		CMCorr = corr[['latex-feature.2','CM-corr']]
		CMWeight = weights[['latex-feat', 'CM-SVM-weight']]
		RFWeight = weights[['latex-feat.1', 'RF-weight']]
		CM = pd.concat([CMCorr, CMWeight], axis = 1)
		combined = pd.concat([CM, RFWeight], axis = 1)
	else:
		BCCorr = corr[['latex-feature','BC-corr']]
		BMCorr = corr[['latex-feature.1', 'BM-corr']]
		CMCorr = corr[['latex-feature.2','CM-corr']]
		BCWeight = weights[['latex-feat', 'BC-SVM-weight']]
		BMWeight = weights[['latex-feat.1', 'BM-SVM-weight']]
		CMWeight = weights[['latex-feat.2', 'CM-SVM-weight']]
		RFWeight = weights[['latex-feat.3', 'RF-weight']]
		BC = pd.concat([BCCorr, BCWeight], axis = 1)
		BM = pd.concat([BMCorr, BMWeight], axis = 1)
		CM = pd.concat([CMCorr, CMWeight], axis = 1)
		combined = pd.concat([BC, BM, CM, RFWeight], axis = 1)
	combined = combined.round(decimals = 3)
	path = "/Users/Laura/Desktop/Dissertation/data/corr-weights/corr-weights-" + lg + "-" + datatype + ".csv"
	combined.to_csv(path)

def main():
	args = parse_args()
	corr, weights = get_paths(args.lg, args.datatype)
	combine(corr, weights, args.lg, args.datatype)

if __name__ == "__main__":
    main()