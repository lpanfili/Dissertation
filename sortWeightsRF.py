# Sorts the features weights from imbalanced and resampled RF
# (SVM) is a different script because the output is so different)
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
	SVMimb = "/Users/Laura/Desktop/Dissertation/data/weights/RF-" + lg + "-imb.csv"
	SVMrs = "/Users/Laura/Desktop/Dissertation/data/weights/RF-" + lg + "-rs.csv"
	return SVMimb, SVMrs

def getTop(data, top, type, lg):
	
	weights = pd.read_csv(data)
	weights = weights.set_index('latex-feat')
	weights['abs'] = weights['weight'].abs()
	weights = weights.sort_values(by = 'abs', ascending = False)
	#print(weights)
	path = "/Users/Laura/Desktop/Dissertation/data/weights/top/" + lg + "-" + type + ".csv"
	weights.to_csv(path)

def main():
	args = parse_args()
	RFimb, RFrs = getPaths(args.lg)
	# Change top to determine how many are printed
	top = 10
	getTop(RFimb, top, "RF-imb", args.lg)

if __name__ == "__main__":
    main()