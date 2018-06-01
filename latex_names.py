# Takes a file with a column of non-latex feature names
# Prints a list of those features with latex-friendly names

import pandas as pd
import csv
import argparse

def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('features_csv', type = str)
	parser.add_argument('input_file', type = str)
	return parser.parse_args()

def make_dict(features_csv):
	features = pd.read_csv(features_csv)
	feat_dict = features.set_index('feature').to_dict()['feature-latex']
	return feat_dict

def get_orig(input_file):
	file_orig = pd.read_csv(input_file)
	feat_orig = file_orig[[0]]
	return feat_orig

def match_feat(feat_dict, feat_orig):
	s = feat_orig[feat_orig.columns[0]]
	feat_orig['feat_new'] = s.map(feat_dict)
	for i in feat_orig['feat_new']:
		print(i)

def main():
	args = parse_args()
	feat_dict = make_dict(args.features_csv)
	feat_orig = get_orig(args.input_file)
	feat_new = match_feat(feat_dict, feat_orig)

if __name__ == "__main__":
	main()