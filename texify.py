import pandas as pd
import argparse
import csv

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_path', type=str,
                        help='path to the input csv')
    parser.add_argument('output_path', type=str,
    	                help='path to the output csv')
    parser.add_argument('features_csv', type=str,
    	                help='path to the features metadata csv')
    return parser.parse_args()
# Makes a dictionary mapping the original feature names
# To the LaTeX-friendly feature names
def make_feature_dict(features_csv):
    feature_dict = {}
    with open(features_csv) as f:
        reader = csv.reader(f)
        header = next(reader)
        for line in reader:
            feature = line[0]
            latex_feature = line[1]
            if feature not in feature_dict:
                feature_dict[feature] = ""
            feature_dict[feature] = latex_feature
    return feature_dict



def texify(df, output_path, feature_dict):

    # Replace feature names with latex-friendly names
    df = df.rename(columns = lambda col: feature_dict[col] if col in feature_dict else col)

    df.to_csv(output_path, index=False, sep='&', line_terminator=' \\\\\n', quoting=3, escapechar='\\')

def main():
	args = parse_args()
	texify(pd.read_csv(args.input_path), args.output_path, make_feature_dict(args.features_csv))

if __name__ == '__main__':
	main()