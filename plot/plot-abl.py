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
	path = "../../data/lgs/" + lg + "/" + lg + "-abl-cat-" + clf + ".csv"
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
def get_color_cat(feat, features_csv):
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
		'12': '#b15928' # surrounding
		}
	for i in feat:
		colors.append(color_dict[str(i)])
	return colors


# Make a stand-alone legend
def plot_legend():
	fig = plt.figure()
	# Colors, by feature type
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
	pos = mpatches.Patch(color = '#ffff99', label = 'Prosodic Position')
	surr = mpatches.Patch(color = '#b15928', label = 'Surrounding Phones')
	labels = [f0, vopt, jitter, cpp, rmse, shimmer, hnr, shr, tilt, f1, dur, pos, surr]
	text = ['f0', 'VoPT', 'Jitter', 'CPP', 'Energy', 'Shimmer', 'HNR', 'SHR', 'Spectral Tilt', 'F1', 'Duration', 'Prosodic Position', 'Surrounding Phones']
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
	colors = get_color_cat(feat, features_csv)
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

if __name__ == "__main__":
	main()