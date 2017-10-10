# Sorts the features weights from imbalanced and resampled SVM
# (RF is a different script because the output is so different)
# Outputs a CSV that can be dumped into LaTeX

import pandas as pd
import csv
import argparse

# Requires lg code as arg
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('lg', type = str, help = 'the three digit code for the language in question')
    return parser.parse_args()

def getPaths(lg):
	SVMimb = "/Users/Laura/Desktop/Dissertation/data/weights/SVM-" + lg + "-imb.csv"
	SVMrs = "/Users/Laura/Desktop/Dissertation/data/weights/SVM-" + lg + "-rs.csv"
	return SVMimb, SVMrs

def getTop(data, top, type, lg):
	weights = pd.read_csv(data)
	weights = weights.set_index('latex-feat')
	if lg == 'guj':
		weights['BMabs'] = weights['BM'].abs()
		weights = weights.sort_values(by = 'BMabs', ascending = False)
		BM = weights.head(top)['BM']
		BM = BM.reset_index()
		topList = BM
	elif lg == 'cmn':
		weights['CMabs'] = weights['CM'].abs()
		weights = weights.sort_values(by = 'CMabs', ascending = False)
		CM = weights.head(top)['CM']
		CM = CM.reset_index()
		topList = CM
	else:
		weights['BCabs'] = weights['BC'].abs()
		weights['BMabs'] = weights['BM'].abs()
		weights['CMabs'] = weights['CM'].abs()
		weights = weights.sort_values(by = 'BCabs', ascending = False)
		BC = weights.head(top)['BC']
		BC = BC.reset_index()
		weights = weights.sort_values(by = 'BMabs', ascending = False)
		BM = weights.head(top)['BM']
		BM = BM.reset_index()
		weights = weights.sort_values(by = 'CMabs', ascending = False)
		CM = weights.head(top)['CM']
		CM = CM.reset_index()
		topList = pd.concat([BC, BM, CM], axis=1)
	topList = topList.round(decimals = 3)
	path = "/Users/Laura/Desktop/Dissertation/data/weights/top/" + lg + "-" + type + ".csv"
	topList.to_csv(path)

def main():
	args = parse_args()
	SVMimb, SVMrs = getPaths(args.lg)
	# Change top to determine how many are printed
	top = 10
	getTop(SVMimb, top, "SVM-imb", args.lg)

if __name__ == "__main__":
    main()