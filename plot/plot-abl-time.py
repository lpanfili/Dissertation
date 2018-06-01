# Plots changed in weighted F1 score
# When each category is ablated
# For both SVM and Random Forest
# Bars = feature categories
# Colors = categories
# Magnitude = changed in weighted F1 score

import matplotlib.pyplot as plt
from matplotlib import rc
import pandas as pd
import argparse
import numpy as np
import matplotlib.patches as mpatches

# Required arguments:
# Features CSV
# Language code
# SVM and RF weighted F1 scores (baseline)
def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('features_csv', type = str, help = 'features csv location')
	parser.add_argument('lg', type = str, help = 'three letter language code')
	parser.add_argument('svm_f1', help = 'baseline SVM weighted F1 score')
	parser.add_argument('rf_f1', help = 'baseline RF weighted F1 score')
	return parser.parse_args()


# Use LaTeX font
def set_font():
	rc('font',**{'family':'serif','serif':['Computer Modern Roman']})
	params = {'backend': 'ps',
		'axes.labelsize': 25,
		'text.fontsize': 15,
		'legend.fontsize': 13,
		'xtick.labelsize': 15,
		'ytick.labelsize': 15,
		'text.usetex': True}
	plt.rcParams.update(params)


# Reads data for one clf at a time
# Returns Pandas DF of new F1 minus baseline f1 per category
def get_data(lg, f1, clf):
	f1 = float(f1)
	path = "../../data/lgs/" + lg + "/" + lg + "-abl-time-" + clf + ".csv"
	data = pd.read_csv(path)
	abl_f1 = data['f1'].copy()
	cat = data['cat'].copy()
	abl_f1 = abl_f1.apply(lambda i: float(i - f1))
	results = pd.concat([abl_f1, cat], axis = 1).set_index(['cat'])
	return results


# Make and return a list
# Containing tuples of categories and change in F1
# For one clf at a time
def make_list(data):
	feat_val_list = []
	feat_dict = data.to_dict()
	for feat in feat_dict:
		for cat in feat_dict[feat]:
			val = feat_dict[feat][cat]
			pair = [cat, val]
			feat_val_list.append(pair)
	return feat_val_list


# Make and return ordered list
# Of colors based on feature category
def get_color_time(feat, features_csv):
	colors = []
	color_dict = {
		'0': '#d7191c', # mean
		'1': '#fdae61', # 1st
		'2': '#ffffbf', # 2nd
		'3': '#abd9e9', # 3rd
		'4': '#2c7bb6' # other
		}
	for i in feat:
		colors.append(color_dict[str(i)])
	return colors


# Make a stand-alone legend
def plot_legend():
	fig = plt.figure()
	# Colors, by feature time span
	mean = mpatches.Patch(color = '#d7191c', label = 'Mean')
	first = mpatches.Patch(color = '#fdae61', label = 'First Third')
	second = mpatches.Patch(color = '#ffffbf', label = 'Second Third')
	third = mpatches.Patch(color = '#abd9e9', label = 'Final Third')
	na = mpatches.Patch(color = '#2c7bb6', label = 'N/A')
	labels = [mean, first, second, third, na]
	text = ['Mean', 'First Third', 'Second Third', 'Final Third', 'N/A']
	fig.legend(labels, text, 'upper left', ncol = 3)
	plt.show()


# Plot change in F1
# Sorted by magnitude
# Color coded by feature category
# For one clf
def plot_abl(to_plot, features_csv, lg, clf, min_lim, max_lim):
	fig = plt.figure()
	plt.xticks([])
	plt.ylabel('Change in weighted F1 score')
	to_plot.sort(key = lambda x: x[1])
	feat, val = zip(*to_plot)
	plt.ylim([min_lim + (min_lim * 0.1), max_lim + (max_lim * 0.1)])
	colors = get_color_time(feat, features_csv)
	x_pos = np.arange(len(feat))
	plt.bar(x_pos, val, color = colors, edgecolor = "none")
	plt.show()


# Of the changes in F1 for both clfs
# Return the max and min
def get_lims(svm, rf):
	all_vals = []
	for i in svm:
		all_vals.append(i[1])
	for i in rf:
		all_vals.append(i[1])
	min_val = min(all_vals)
	max_val = max(all_vals)
	return min_val, max_val

# Make a stand-alone legend
def plot_legend():
	fig = plt.figure()
	# Colors, by feature type
	mean = mpatches.Patch(color = '#d7191c', label = 'Mean')
	first = mpatches.Patch(color = '#fdae61', label = 'First Third')
	second = mpatches.Patch(color = '#ffffbf', label = 'Second Third')
	third = mpatches.Patch(color = '#abd9e9', label = 'Final Third')
	na = mpatches.Patch(color = '#2c7bb6', label = 'N/A')
	labels = [mean, first, second, third, na]
	text = ['Mean', 'First', 'Second', 'Third', 'N/A']
	fig.legend(labels, text)
	plt.show()


def main():
	args = parse_args()
	set_font()
	svm = get_data(args.lg, args.svm_f1, 'SVM')
	rf = get_data(args.lg, args.rf_f1, 'RF')
	to_plot_svm = make_list(svm)
	to_plot_rf = make_list(rf)
	min_lim, max_lim = get_lims(to_plot_svm, to_plot_rf)
	#plot_legend()
	plot_abl(to_plot_svm, args.features_csv, args.lg, 'SVM', min_lim, max_lim)
	plot_abl(to_plot_rf, args.features_csv, args.lg, 'RF', min_lim, max_lim)
	plot_legend()

if __name__ == "__main__":
	main()