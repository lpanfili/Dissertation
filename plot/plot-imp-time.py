# Plots Random Forest feature importance
# Bars = features
# Colors = feature categories
# Magnitude = importance

import matplotlib.pyplot as plt
from matplotlib import rc
import pandas as pd
import argparse
import numpy as np
import matplotlib.patches as mpatches

# Required arguments:
# Features CSV
# Language code
def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('features_csv', type = str, help = 'features csv location')
	parser.add_argument('lg', type = str, help = 'three letter language code')
	return parser.parse_args()


# Use LaTeX font
def set_font():
	rc('font',**{'family':'serif','serif':['Computer Modern Roman']})
	params = {'backend': 'ps',
		'axes.labelsize': 15,
		'text.fontsize': 15,
		'legend.fontsize': 13,
		'xtick.labelsize': 15,
		'ytick.labelsize': 15,
		'text.usetex': True}
	plt.rcParams.update(params)


# Read in data
# Return data as Pandas DF
def get_data(lg):
	path = "../../data/lgs/" + lg + "/" + lg + "-importance_rs.csv"
	data = pd.read_csv(path, na_values = "")
	data = data.set_index(['feat'])
	data = data['importance'].copy()
	return data


# Make and return a list
# Containing tuples of features and importance values
def make_list(data):
	feat_val_list = []
	feat_dict = data.to_dict()
	for feat in feat_dict:
		pair = [feat, feat_dict[feat]]
		feat_val_list.append(pair)
	return feat_val_list


# Plot Importance values
# Sorted by magnitude
# Color coded by feature category
def plot_feat(feat_val_list, features_csv, lg, title):
	feat_val_list.sort(key = lambda x: abs(x[1]), reverse = True)
	feat, val = zip(*feat_val_list)
	colors = get_color(feat, features_csv)
	x_pos = np.arange(len(feat))
	plt.bar(x_pos, val, color = colors, edgecolor = "none")
	plt.xlabel('Feature')
	plt.ylabel('Importance')
	plt.xticks([])
	plt.xlim([0,len(val)])
	mean = mpatches.Patch(color = '#d7191c', label = 'Mean')
	first = mpatches.Patch(color = '#fdae61', label = 'First Third')
	second = mpatches.Patch(color = '#ffffbf', label = 'Second Third')
	third = mpatches.Patch(color = '#abd9e9', label = 'Final Third')
	na = mpatches.Patch(color = '#2c7bb6', label = 'N/A')
	plt.legend(handles=[mean, first, second, third, na], bbox_to_anchor = (1.1, 1.05))
	#plt.title(title)
	plt.show()


# Make and return ordered list
# Of colors based on feature time span
def get_color(feat, features_csv):
	colors = []
	color_dict = {
		'14': '#d7191c', # mean
		'15': '#fdae61', # 1st
		'16': '#ffffbf', # 2nd
		'17': '#abd9e9', # 3rd
		'18': '#2c7bb6' # other
	}
	features_all = pd.read_csv(features_csv)
	features = pd.concat([features_all['feature'], features_all['category']], axis = 1)
	features = features.set_index(['feature'])
	category_dict = features.to_dict()
	category_dict = category_dict['category']
	for i in category_dict:
		if i.endswith('1'):
			category_dict[i] = '15'
		elif i.endswith('2'):
			category_dict[i] = '16'
		elif i.endswith('3'):
			category_dict[i] = '17'
		elif i.endswith('mean'):
			category_dict[i] = '14'
		else:
			category_dict[i] = '18'
	for i in feat:
		category = category_dict[i]
		colors.append(color_dict[category])
	return colors

# Returns title of plot
# Depending on language
def make_title(lg):
	if lg == 'eng':
		title = 'English Random Forest Importance'
	if lg == 'guj':
		title = 'Gujarati Random Forest Importance'
	if lg == 'hmn':
		title = 'Hmong Random Forest Importance'
	if lg == 'cmn':
		title = 'Mandarin Random Forest Importance'
	if lg == 'maj':
		title = 'Mazatec Random Forest Importance'
	if lg == 'zap':
		title = 'Zapotec Random Forest Importance'
	return title


def main():
	args = parse_args()
	set_font()
	title = make_title(args.lg)
	data = get_data(args.lg)
	feat_val_list = make_list(data)
	plot_feat(feat_val_list, args.features_csv, args.lg, title)


if __name__ == "__main__":
	main()