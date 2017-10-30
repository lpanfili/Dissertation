# Sorts the correlations for a given language
# And by a given contrast
# Sorts by magnitude -- largest to smallest
# Outputs a CSV that can be dumped into LaTeX

import pandas as pd
import csv
import argparse

# Requires lg code as arg
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('lg', type = str, help = 'the three digit code for the language in question')
    return parser.parse_args()

def getPath(lg):
	path = "/Users/Laura/Desktop/Dissertation/data/correlations/correlationMatrixNorm-" + lg + ".csv"
	return path

def sort(data, type, lg):
	corr = pd.read_csv(data)
	corr = corr.set_index('latex-feature')
	corr = corr.round(decimals = 3)
	corr = corr.rename(columns = {'breathy': 'BC-corr', 'breathy.1': 'BM-corr', 'creaky': 'CM-corr'})
	corr = corr[pd.notnull(corr.index)] # Remove breathy, creaky, and modal as features
	# Add columns for absolute values
	corr['BC-abs'] = corr['BC-corr'].abs()
	corr['BM-abs'] = corr['BM-corr'].abs()
	corr['CM-abs'] = corr['CM-corr'].abs()
	# Sort each column by magnitude and pull out each comparison out as its own dataframe
	corr = corr.sort_values(by = 'BC-abs', ascending = False)
	BC  = corr[['BC-corr']].copy().reset_index()
	corr = corr.sort_values(by = 'BM-abs', ascending = False)
	BM = corr[['BM-corr']].copy().reset_index()
	corr = corr.sort_values(by = 'CM-abs', ascending = False)
	CM = corr[['CM-corr']].copy().reset_index()
	# Put the three into one dataframe
	sortedCorr = pd.concat([BC, BM, CM], axis = 1)
	path = "/Users/Laura/Desktop/Dissertation/data/correlations/sortedCorr-" + lg + "-" + type + ".csv"
	sortedCorr.to_csv(path)


def main():
	args = parse_args()
	path = getPath(args.lg)
	sort(path, "imb", args.lg)

if __name__ == "__main__":
    main()