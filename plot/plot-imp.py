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
	f0 = mpatches.Patch(color = '#a6cee3', label = 'f0')
	vopt = mpatches.Patch(color = '#1f78b4', label = 'VoPT')
	jitter = mpatches.Patch(color = '#000080', label = 'Jitter')
	cpp = mpatches.Patch(color = '#b2df8a', label = 'CPP')
	rmse = mpatches.Patch(color = '#33a02c', label = 'RMS Energy')
	shimmer = mpatches.Patch(color = '#fb9a99', label = "Shimmer")
	hnr = mpatches.Patch(color = '#e31a1c', label = 'HNR')
	shr = mpatches.Patch(color = '#fdbf6f', label = 'SHR')
	tilt = mpatches.Patch(color = '#ff7f00', label = "Spectral Tilt")
	f1 = mpatches.Patch(color = '#cab2d6', label = 'F1')
	dur = mpatches.Patch(color = '#6a3d9a', label = 'Duration')
	if lg == 'eng':
		pos = mpatches.Patch(color = '#ffff99', label = 'Prosodic Position')
		surr = mpatches.Patch(color = '#b15928', label = 'Surrounding Phones')
		plt.legend(handles=[f0, vopt, jitter, cpp, rmse, shimmer, hnr, shr, tilt, f1, dur, pos, surr])
	else:
		plt.legend(handles=[f0, vopt, jitter, cpp, rmse, shimmer, hnr, shr, tilt, f1, dur])
	plt.title(title)
	plt.show()


# Make and return ordered list
# Of colors based on feature category
def get_color(feat, features_csv):
	colors = []
	color_dict = {
		'5': '#a6cee3', # f0
		'11': '#1f78b4', # VoPT
		'8': '#000080', # jitter
		'1': '#b2df8a', # CPP
		'2': '#33a02c', # RMSE
		'9': '#fb9a99', # shimmer
		'3': '#e31a1c', # HNR
		'4': '#fdbf6f', # SHR
		'0': '#ff7f00', # Tilt
		'6': '#cab2d6', # F1
		'7': '#6a3d9a', # dur
		'10': '#ffff99', # prosodic pos
		'12': '#b15928', # surrounding
		'13': '#ffffff', # white for any I want to hide
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