# Makes a bar plot of correlations, weights, or importances
# Colors = categories
# y-axis = metric

import matplotlib.pyplot as plt
from matplotlib import rc
import pandas as pd
import argparse
import numpy as np
import matplotlib.patches as mpatches

# Two args, lg and metric
def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('features_csv', type = str, help = 'features csv location')
	parser.add_argument('lg', type = str, help = 'three letter language code')
	parser.add_argument('metric', type = str, help = 'corr, weight, imp, abl')
	parser.add_argument('svm_acc', nargs='?', help = 'baseline svm acc, optional')
	parser.add_argument('rf_acc', nargs='?', help = 'baseline rf acc, optional')
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


# Makes and returns a dataframe from each contrast
def get_data_corr(lg):
	path = "../data/lgs/" + lg + "/" + lg + "-corr-all.csv"
	data = pd.read_csv(path, na_values = "")
	data = data.set_index(['Unnamed: 0'])
	BC = data['BC-corr'].copy()
	BM = data['BM-corr'].copy()
	CM = data['CM-corr'].copy()
	return data, BC, BM, CM

def get_data_imp(lg):
	path = "../data/lgs/" + lg + "/" + lg + "-importance_rs.csv"
	data = pd.read_csv(path, na_values = "")
	data = data.set_index(['feat'])
	data = data['importance'].copy()
	return data

def get_data_weight(lg):
	path = "../data/lgs/" + lg + "/" + lg + "-weights-rs.csv"
	data = pd.read_csv(path, na_values = "")
	data = data.set_index(['feat'])
	if lg == 'cmn':
		CM = data['CM-weight'].copy()
		BC = ""
		BM = ""
	elif lg == 'guj':
		BM = data['BM-weight'].copy()
		BC = ""
		CM = ""
	else:
		BC = data['BC-weight'].copy()
		BM = data['BM-weight'].copy()
		CM = data['CM-weight'].copy()
	return data, BC, BM, CM

def get_data_abl(lg, svm_acc, rf_acc):
	# SVM
	svm_acc = float(svm_acc)
	path_svm = "../data/lgs/" + lg + "/" + lg + "-abl-SVM.csv"
	svm_data = pd.read_csv(path_svm)
	svm_data = svm_data.set_index(['feat'])
	svm = svm_data['acc'].copy()
	svm = svm.apply(lambda i: float(i - svm_acc))
	# RF
	rf_acc = float(rf_acc)
	path_rf = "../data/lgs/" + lg + "/" + lg + "-abl-RF.csv"
	rf_data = pd.read_csv(path_rf)
	rf_data = rf_data.set_index(['feat'])
	rf = rf_data['acc'].copy()
	rf = rf.apply(lambda i: float(i - rf_acc))
	data = pd.concat([svm, rf])
	data.to_csv("/Users/Laura/Desktop/test.csv")
	return data, svm, rf


# Makes and returns a list of tuples for just one contrast
# Feature, metric
def make_list_contrast(data):
	feat_val_list = []
	feat_dict = data.to_dict()
	for feat in feat_dict:
		pair = [feat, feat_dict[feat]]
		feat_val_list.append(pair)
	return feat_val_list


def plot_feat(feat_val_list, features_csv, metric, lg, title):
	if metric != 'abl':
		# Sort by absolute value
		feat_val_list.sort(key = lambda x: abs(x[1]), reverse = True)
	else:
		# Sort by value
		feat_val_list.sort(key = lambda x: x[1])
	feat, val = zip(*feat_val_list)
	colors = get_color(feat, features_csv)
	if metric != 'abl':
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
	if metric == 'abl':
		plt.ylabel('Accuracy Change')
	plt.xticks([])
	"""
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
		if metric == 'abl':
			plt.legend(handles=[f0, vopt, jitter, cpp, rmse, shimmer, hnr, shr, tilt, f1, dur, pos, surr], loc = 4)
		else:
			plt.legend(handles=[f0, vopt, jitter, cpp, rmse, shimmer, hnr, shr, tilt, f1, dur, pos, surr])
	else:
		if metric == 'abl':
			plt.legend(handles=[f0, vopt, jitter, cpp, rmse, shimmer, hnr, shr, tilt, f1, dur], loc = 4)
		else:
			plt.legend(handles=[f0, vopt, jitter, cpp, rmse, shimmer, hnr, shr, tilt, f1, dur])
	"""
	# LEGEND BY TIME PERIOD
	m = mpatches.Patch(color = '#d7191c', label = 'Mean')
	first = mpatches.Patch(color = '#fdae61', label = 'Beginning')
	second = mpatches.Patch(color = '#abd9e9', label = 'Middle')
	third = mpatches.Patch(color = '#2c7bb6', label = 'End')
	if metric == 'abl':
		plt.legend(handles = [m, first, second, third], loc = 4)
	else:
		plt.legend(handles = [m, first, second, third])
	plt.title(title)
	plt.show()


# Make the list of colors that corresponds with the feature categories
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
		'16': '#abd9e9', # 2nd
		'17': '#2c7bb6' # 3rd
	}
	features_all = pd.read_csv(features_csv)
	features = pd.concat([features_all['feature'], features_all['category']], axis = 1)
	features = features.set_index(['feature'])
	category_dict = features.to_dict()
	category_dict = category_dict['category']
	# Make all that end in 2 (middle third) cat 13
	#for i in category_dict:
	#	if i.endswith('2'):
	#		category_dict[i] = '13'
	# Convert to color codes by timing
	for i in category_dict:
		if i.endswith('1'):
			category_dict[i] = '15'
		elif i.endswith('2'):
			category_dict[i] = '16'
		elif i.endswith('3'):
			category_dict[i] = '17'
		else:
			category_dict[i] = '14'
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



def make_title(lg, metric):
	if lg == 'eng':
		title = 'English'
	if lg == 'guj':
		title = 'Gujarati'
	if lg == 'hmn':
		title = 'Hmong'
	if lg == 'cmn':
		title = 'Mandarin'
	if lg == 'maj':
		title = 'Mazatec'
	if lg == 'zap':
		title = 'Zapotec'
	if metric == 'corr':
		title += ' Correlations'
	if metric == 'weight':
		title += ' SVM Weights'
	if metric == 'imp':
		title += ' RF Importance'
	if metric == 'abl':
		title += ' Ablation'
	return title

def append_title(title, clf):
	if clf == 'svm':
		title += ', SVM'
	if clf == 'rf':
		title += ', Random Forest'
	return title


def main():
	args = parse_args()
	set_font()
	# Make partial title
	title = make_title(args.lg, args.metric)
	# Get the data depending on which information you want
	if args.metric != 'imp' and args.metric != 'abl':
		if args.metric == 'corr':
			data, BC, BM, CM = get_data_corr(args.lg)
		if args.metric == 'weight':
			data, BC, BM, CM = get_data_weight(args.lg)
		# Plot for lgs with two-way contrast
		if args.lg == 'guj':
			BM_feat_val_list = make_list_contrast(BM)
			plot_feat(BM_feat_val_list, args.features_csv, args.metric, args.lg, title)
		elif args.lg == 'cmn':
			CM_feat_val_list = make_list_contrast(CM)
			plot_feat(CM_feat_val_list, args.features_csv, args.metric, args.lg, title)
		# Plot contrats + all for other lgs
		else:
			BC_feat_val_list = make_list_contrast(BC)
			BM_feat_val_list = make_list_contrast(BM)
			CM_feat_val_list = make_list_contrast(CM)
			plot_feat(BC_feat_val_list, args.features_csv, args.metric, args.lg, title + ', B vs. C')
			plot_feat(BM_feat_val_list, args.features_csv, args.metric, args.lg, title + ', B vs. M')
			plot_feat(CM_feat_val_list, args.features_csv, args.metric, args.lg, title + ', C vs. M')
			# All
			all_feat_val_list = BC_feat_val_list + BM_feat_val_list + CM_feat_val_list
			plot_feat(all_feat_val_list, args.features_csv, args.metric, args.lg, title +', All Contrasts')
	if args.metric == 'imp':
		data = get_data_imp(args.lg)
		feat_val_list = make_list_contrast(data)
		plot_feat(feat_val_list, args.features_csv, args.metric, args.lg, title)
	if args.metric == 'abl':
		data, svm, rf = get_data_abl(args.lg, args.svm_acc, args.rf_acc)
		feat_val_list = make_list_contrast(svm)
		title_svm = append_title(title, 'svm')
		plot_feat(feat_val_list, args.features_csv, args.metric, args.lg, title_svm)
		feat_val_list = make_list_contrast(rf)
		title_rf = append_title(title, 'rf')
		plot_feat(feat_val_list, args.features_csv, args.metric, args.lg, title_rf)
	

# TODO: Add titles to plots
# TODO: Add ablation

if __name__ == "__main__":
	main()