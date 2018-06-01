import pandas as pd
import sklearn
from sklearn.metrics import f1_score
import argparse
import collections

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('lg', type=str, help='the three digit code for the language in question')
    return parser.parse_args()

# Get a list of the actual ys
def get_y(lg):
	csv = "../data/lgs/" + lg + "/" + lg + "-all.csv"
	data = pd.read_csv(csv, na_values=["--undefined--",0])
	y = data['phonation'].tolist()
	return y

# Make a list of fake predicted y
# Contains as many elements as the original data set
# But all are majority class
def get_ypred(y):
	length = len(y)
	counter = collections.Counter(y)
	print(type(counter))
	maximum = [0, 'X']
	for i in counter:
		if counter[i] > maximum[0]:
			maximum = [counter[i], i]
	maj_class = maximum[1]
	print(maj_class)
	ypred = [maj_class] * length
	return ypred



def get_f1(y, ypred):
	fscore = f1_score(y, ypred, average = 'weighted')
	return fscore

def main():
    args = parse_args()
    y = get_y(args.lg)
    ypred = get_ypred(y)
    fscore = get_f1(y, ypred)
    print(fscore)

if __name__ == "__main__":
    main()