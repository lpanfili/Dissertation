# Plots correlation coefficients
# Bars = features
# Colors = feature categories
# Magnitude = correlation

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


# Reads data
# Makes and returns a separate Pandas DF for each contrast
def get_data(lg):
	path = "../../data/lgs/" + lg + "/" + lg + "-corr-all.csv"
	data = pd.read_csv(path, na_values = "")
	data = data.set_index(['Unnamed: 0'])
	BC = data['BC-corr'].copy()
	BM = data['BM-corr'].copy()
	CM = data['CM-corr'].copy()
	return BC, BM, CM

# Make and return a list
# Containing tuples of features and correlations
# For just one contrast at a time
def make_list_contrast(data):
	feat_val_list = []
	feat_dict = data.to_dict()
	for feat in feat_dict:
		pair = [feat, feat_dict[feat]]
		feat_val_list.append(pair)
	return feat_val_list


# Plot correlations
# Sorted by magnitude
# Color coded by feature category
def plot_feat_flipped(feat_val_list, features_csv, lg, title, left, right):
	feat_val_list.sort(key = lambda x: abs(x[1]), reverse = True)
	feat, val = zip(*feat_val_list)
	colors = get_color(feat, features_csv)
	x_pos = np.arange(len(feat))
	fig = plt.figure()
	ax1 = fig.add_subplot(111)
	ax1.barh(x_pos, val, color = colors, edgecolor = "none")
	ax1.set_xlabel('Correlation')
	ax1.set_xlim([-1,1])
	ylim = ax1.get_ylim()[1]
	ax1.text(0.5, -ylim/10, right[0], size = 20) # Label right phonation
	ax1.text(-0.5, -ylim/10, left[0], size = 20) # Label left phonation
	ax1.set_yticks([])
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
		ax1.legend(handles=[f0, vopt, jitter, cpp, rmse, shimmer, hnr, shr, tilt, f1, dur, pos, surr], bbox_to_anchor = (1.1, 1))
	else:
		ax1.legend(handles=[f0, vopt, jitter, cpp, rmse, shimmer, hnr, shr, tilt, f1, dur], bbox_to_anchor = (1.1, 1.05))
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
		title = 'English Correlations'
	if lg == 'guj':
		title = 'Gujarati Correlations'
	if lg == 'hmn':
		title = 'Hmong Correlations'
	if lg == 'cmn':
		title = 'Mandarin Correlations'
	if lg == 'maj':
		title = 'Mazatec Correlations'
	if lg == 'zap':
		title = 'Zapotec Correlations'
	return title

def main():
	args = parse_args()
	set_font()
	title = make_title(args.lg)
	BC, BM, CM = get_data(args.lg)
	if args.lg == 'guj':
		BM_feat_val_list = make_list_contrast(BM)
		plot_feat_flipped(BM_feat_val_list, args.features_csv, args.lg, title + ', B vs. M', 'Modal', 'Breathy')
	elif args.lg == 'cmn':
		CM_feat_val_list = make_list_contrast(CM)
		plot_feat_flipped(CM_feat_val_list, args.features_csv, args.lg, title + ', C vs. M', 'Modal', 'Creaky')
	else:
		BC_feat_val_list = make_list_contrast(BC)
		BM_feat_val_list = make_list_contrast(BM)
		CM_feat_val_list = make_list_contrast(CM)
		plot_feat_flipped(BC_feat_val_list, args.features_csv, args.lg, title + ', B vs. C', 'Creaky', 'Breathy')
		plot_feat_flipped(BM_feat_val_list, args.features_csv, args.lg, title + ', B vs. M', 'Modal', 'Breathy')
		plot_feat_flipped(CM_feat_val_list, args.features_csv, args.lg, title + ', C vs. M', 'Modal', 'Creaky')
			
if __name__ == "__main__":
	main()