# Makes a bar plot of correlations, weights, or importances
# Colors = categories
# y-axis = metric

import matplotlib.pyplot as plt
from matplotlib import rc
import pandas as pd
import argparse
import numpy as np

# Two args, lg and metric
def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('features_csv', type = str, help = 'features csv location')
	parser.add_argument('lg', type = str, help = 'three letter language code')
	parser.add_argument('metric', type = str, help = 'corr, weight, imp, abl')
	return parser.parse_args()


# Use LaTeX font
def set_font():
	rc('font',**{'family':'serif','serif':['Computer Modern Roman']})
	params = {'backend': 'ps',
		'axes.labelsize': 20,
		'text.fontsize': 20,
		'legend.fontsize': 10,
		'xtick.labelsize': 15,
		'ytick.labelsize': 15,
		'text.usetex': True}
	plt.rcParams.update(params)


# Makes and returns a df of the data for correlations
def get_data_corr(lg):
	path = "../data/lgs/" + lg + "/" + lg + "-corr-all.csv"
	data = pd.read_csv(path, na_values = "")
	data = data.set_index(['Unnamed: 0'])
	return data

def get_data_imp(lg):
	path = "../data/lgs/" + lg + "/" + lg + "-importance_rs.csv"
	data = pd.read_csv(path, na_values = "")
	data = data.set_index(['feat'])
	return data

def get_data_weight(lg):
	path = "../data/lgs/" + lg + "/" + lg + "-weights-rs.csv"
	data = pd.read_csv(path, na_values = "")
	data = data.set_index(['feat'])
	return data


# Makes and returns a list of tuples
# Feature, metric
def make_list(data):
	feat_val_list = []
	feat_dict = data.to_dict()
	for contrast in feat_dict:
		for feat in feat_dict[contrast]:
			pair = [feat, feat_dict[contrast][feat]]
			feat_val_list.append(pair)
	return feat_val_list


def plot_feat(feat_val_list, features_csv, metric, lg):
	# Sort by absolute value
	feat_val_list.sort(key = lambda x: abs(x[1]), reverse = True)
	feat, val = zip(*feat_val_list)
	colors = get_color(feat, features_csv)
	val = [abs(n) for n in val] # Convert to absolute values
	# Plot
	x_pos = np.arange(len(feat))
	plt.bar(x_pos, val, color = colors, edgecolor = "none")
	plt.xlabel('Feature')
	if metric == 'corr':
		plt.ylabel('Correlation (Absolute Value)')
	if metric == 'imp':
		plt.ylabel('Importance')
	if metric == 'weight':
		plt.ylabel('Weight (Absolute Value)')
	plt.xticks([])
	plt.show()


# Make the list of colors that corresponds with the feature categories
def get_color(feat, features_csv):
	colors = []
	color_dict = {
		'0': '#a6cee3',
		'1': '#1f78b4',
		'2': '#b2df8a',
		'3': '#33a02c',
		'4': '#fb9a99',
		'5': '#e31a1c',
		'6': '#fdbf6f',
		'7': '#ff7f00',
		'8': '#cab2d6',
		'9': '#6a3d9a',
		'10': '#ffff99',
		'11': '#b15928',
		'12': '#636363'
	}
	features_all = pd.read_csv(features_csv)
	features = pd.concat([features_all['feature'], features_all['category']], axis = 1)
	features = features.set_index(['feature'])
	category_dict = features.to_dict()
	category_dict = category_dict['category']
	for i in feat:
		category = category_dict[i]
		colors.append(color_dict[category])
	return colors


# Adds a column for the category of the data
def group_features(data, features_csv):
	features_all = pd.read_csv(features_csv)
	features = pd.concat([features_all['feature'], features_all['category']], axis = 1)
	features = features.set_index(['feature'])
	category_dict = features.to_dict()
	category_dict = category_dict['category']
	# Replace features in data with category numbers
	category_list = []
	for index, row in data.iterrows():
		category_list.append(category_dict[row[0]])
	category_list = pd.Series(category_list)
	data['Category'] = category_list.values
	return data


def main():
	args = parse_args()
	set_font()
	if args.metric == 'corr':
		data = get_data_corr(args.lg)
	if args.metric == 'imp':
		data = get_data_imp(args.lg)
	if args.metric == 'weight':
		data = get_data_weight(args.lg)
	feat_val_list = make_list(data)
	plot_feat(feat_val_list, args.features_csv, args.metric, args.lg)

if __name__ == "__main__":
	main()