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

def get_data_abl_cat(lg, svm_acc, rf_acc):
	# SVM
	svm_acc = float(svm_acc)
	path_svm = "../data/lgs/" + lg + "/" + lg + "-abl-cat-SVM.csv"
	svm_data = pd.read_csv(path_svm)
	svm_abl_acc = svm_data['acc'].copy()
	svm_cat = svm_data['cat'].copy()
	svm_abl_acc = svm_abl_acc.apply(lambda i: float(i - svm_acc))
	svm_res = pd.concat([svm_abl_acc, svm_cat], axis = 1).set_index(['cat'])
	# RF
	rf_acc = float(rf_acc)
	path_rf = "../data/lgs/" + lg + "/" + lg + "-abl-cat-RF.csv"
	rf_data = pd.read_csv(path_rf)
	rf_abl_acc = rf_data['acc'].copy()
	rf_cat = rf_data['cat'].copy()
	rf_abl_acc = rf_abl_acc.apply(lambda i: float(i - svm_acc))
	rf_res = pd.concat([rf_abl_acc, rf_cat], axis = 1).set_index(['cat'])
	return 0, svm_res, rf_res


# Makes and returns a list of tuples for just one contrast
# Feature, metric
def make_list_contrast(data):
	feat_val_list = []
	feat_dict = data.to_dict()
	for feat in feat_dict:
		pair = [feat, feat_dict[feat]]
		feat_val_list.append(pair)
	return feat_val_list

# Makes and returns a list of tuples for just one contrast
# Feature, metric
def make_list_contrast_abl(data):
	feat_val_list = []
	feat_dict = data.to_dict()
	for feat in feat_dict:
		for cat in feat_dict[feat]:
			val = feat_dict[feat][cat]
			pair = [cat, val]
			feat_val_list.append(pair)
	return feat_val_list


def plot_feat(feat_val_list, features_csv, metric, lg, title):
	if metric != 'abl' and metric != 'abl-cat':
		# Sort by absolute value
		feat_val_list.sort(key = lambda x: abs(x[1]), reverse = True)
	else:
		# Sort by value
		feat_val_list.sort(key = lambda x: x[1])
	feat, val = zip(*feat_val_list)
	if metric == 'abl-cat':
		colors = get_color_cat(feat, features_csv)
	else:
		colors = get_color(feat, features_csv)
	if metric != 'abl' and metric != 'abl-cat':
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
	plt.xlim([0,len(val)])
	# Normal colors, by feature type
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
		elif metric == 'abl-cat':
			plt.legend(handles=[f0, vopt, jitter, cpp, rmse, shimmer, hnr, shr, tilt, f1, dur, pos, surr], bbox_to_anchor = (.39, 1.005))
		else:
			plt.legend(handles=[f0, vopt, jitter, cpp, rmse, shimmer, hnr, shr, tilt, f1, dur, pos, surr])
	else:
		if metric == 'abl':
			plt.legend(handles=[f0, vopt, jitter, cpp, rmse, shimmer, hnr, shr, tilt, f1, dur], loc = 4)
		elif metric == 'abl-cat':
			plt.legend(handles=[f0, vopt, jitter, cpp, rmse, shimmer, hnr, shr, tilt, f1, dur], bbox_to_anchor = (.3, 1))
		else:
			plt.legend(handles=[f0, vopt, jitter, cpp, rmse, shimmer, hnr, shr, tilt, f1, dur])
	"""
	# LEGEND BY TIME PERIOD
	m = mpatches.Patch(color = '#d7191c', label = 'Mean')
	first = mpatches.Patch(color = '#fdae61', label = 'Beginning')
	second = mpatches.Patch(color = '#ffffbf', label = 'Middle')
	third = mpatches.Patch(color = '#abd9e9', label = 'End')
	other = mpatches.Patch(color = '#2c7bb6', label = 'n/a')
	if metric == 'abl':
		plt.legend(handles = [m, first, second, third, other], loc = 4)
	else:
		plt.legend(handles = [m, first, second, third, other])
	"""
	plt.title(title)
	plt.show()


def plot_feat_flipped(feat_val_list, features_csv, metric, lg, title):
	feat_val_list.sort(key = lambda x: abs(x[1]), reverse = True)
	feat, val = zip(*feat_val_list)
	colors = get_color(feat, features_csv)
	# Plot
	x_pos = np.arange(len(feat))
	fig = plt.figure()
	ax1 = fig.add_subplot(111)
	ax1.barh(x_pos, val, color = colors, edgecolor = "none")
	if metric == 'corr':
		ax1.set_xlabel('Correlation')
		ax1.set_xlim([-1,1])
	if metric == 'weight':
		ax1.set_xlabel('Weight')
	ax1.set_yticks([])
	ax2 = ax1.twiny()
	ax2.xaxis.set_ticks_position("bottom")
	ax2.xaxis.set_label_position("bottom")
	# Offset the twin axis below the host
	ax2.spines["bottom"].set_position(("axes", -0.15))
	ax2.set_frame_on(True)
	ax2.patch.set_visible(False)
	for sp in ax2.spines.values():
	    sp.set_visible(False)
	ax2.spines["bottom"].set_visible(True)
	#ax2 = fig.add_axes((0.1,0.1,0.8,0.0))	
	# Normal colors, by feature type
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

	"""
	# LEGEND BY TIME PERIOD
	m = mpatches.Patch(color = '#d7191c', label = 'Mean')
	first = mpatches.Patch(color = '#fdae61', label = 'Beginning')
	second = mpatches.Patch(color = '#ffffbf', label = 'Middle')
	third = mpatches.Patch(color = '#abd9e9', label = 'End')
	other = mpatches.Patch(color = '#2c7bb6', label = 'n/a')
	if metric == 'abl':
		plt.legend(handles = [m, first, second, third, other], loc = 4)
	else:
		plt.legend(handles = [m, first, second, third, other])
	"""
	"""
	"""
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
		'16': '#ffffbf', # 2nd
		'17': '#abd9e9', # 3rd
		'18': '#2c7bb6' # other
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
	"""
	# Convert to color codes by timing
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
	"""
	for i in feat:
		category = category_dict[i]
		colors.append(color_dict[category])
	return colors

# Make the list of colors that corresponds with the feature categories
# When the input is the cat number
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
	if metric == 'abl-cat':
		title += ' Category Ablation'
	return title

def append_title(title, clf):
	if clf == 'svm':
		title += ', SVM'
	if clf == 'rf':
		title += ', Random Forest'
	return title

def plot_feat_cat(feat_val_list_svm, feat_val_list_rf, features_csv, lg, title):
	fig = plt.figure()
	title = "English Category Ablation"
	fig.suptitle(title, fontsize = 17)
	fig.text(0.06, 0.5, 'Accuracy Change', ha='center', va='center', rotation='vertical')
	plt.xticks([])
	plt.yticks([])
	# SVM
	feat_val_list_svm.sort(key = lambda x: x[1])
	feat_svm, val_svm = zip(*feat_val_list_svm)
	feat_val_list_rf.sort(key = lambda x: x[1])
	feat_rf, val_rf = zip(*feat_val_list_rf)
	max_y = max(val_svm + val_rf)
	min_y = min(val_svm + val_rf)
	colors = get_color_cat(feat_svm, features_csv)
	x_pos = np.arange(len(feat_svm))
	ax1 = fig.add_subplot(211)
	ax1.bar(x_pos, val_svm, color = colors, edgecolor = "none")
	ax1.axhline(color = 'black', linestyle = '--')
	ax1.set_ylim([int(min_y) - 1, int(max_y) + 1])
	ax1.set_title("SVM")
	plt.xticks([])
	# RF
	colors = get_color_cat(feat_rf, features_csv)
	x_pos = np.arange(len(feat_rf))
	ax2 = fig.add_subplot(212)
	ax2.bar(x_pos, val_rf, color = colors, edgecolor = "none")
	ax2.axhline(color = 'black', linestyle = '--')
	ax2.set_ylim([int(min_y) - 1, int(max_y) + 1])
	ax2.set_title("Random Forest")
	plt.xticks([])
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
	if lg == 'eng':
		pos = mpatches.Patch(color = '#ffff99', label = 'Prosodic Position')
		surr = mpatches.Patch(color = '#b15928', label = 'Surrounding Phones')
		plt.legend(handles=[f0, vopt, jitter, cpp, rmse, shimmer, hnr, shr, tilt, f1, dur, pos, surr], ncol = 3, bbox_to_anchor = (0.5, 0), loc = 'center')
		#plt.legend(handles=[f0, vopt, jitter, cpp, rmse, shimmer, hnr, shr, tilt, f1, dur, pos, surr], ncol = 1, loc='center left', bbox_to_anchor=(.95, 1))
	else:
		#plt.legend(handles=[f0, vopt, jitter, cpp, rmse, shimmer, hnr, shr, tilt, f1, dur], ncol = 3, bbox_to_anchor = (0.5, .15), loc='center')
		#plt.legend(handles=[f0, vopt, jitter, cpp, rmse, shimmer, hnr, shr, tilt, f1, dur], ncol = 3, bbox_to_anchor = (0.5, .05), loc='center')
		plt.legend(handles=[f0, vopt, jitter, cpp, rmse, shimmer, hnr, shr, tilt, f1, dur], ncol = 3, bbox_to_anchor = (0.5, -0.1), loc='center')
	plt.show()


def main():
	args = parse_args()
	set_font()
	# Make partial title
	title = make_title(args.lg, args.metric)
	# Get the data depending on which information you want
	if args.metric != 'imp' and args.metric != 'abl' and args.metric != 'abl-cat':
		if args.metric == 'corr':
			data, BC, BM, CM = get_data_corr(args.lg)
		if args.metric == 'weight':
			data, BC, BM, CM = get_data_weight(args.lg)
		# Plot for lgs with two-way contrast
		if args.lg == 'guj':
			BM_feat_val_list = make_list_contrast(BM)
			plot_feat(BM_feat_val_list, args.features_csv, args.metric, args.lg, title)
			# # # # # 
			plot_feat_flipped(BM_feat_val_list, args.features_csv, args.metric, args.lg, title)
		elif args.lg == 'cmn':
			CM_feat_val_list = make_list_contrast(CM)
			plot_feat(CM_feat_val_list, args.features_csv, args.metric, args.lg, title)
			plot_feat_flipped(CM_feat_val_list, args.features_csv, args.metric, args.lg, title)
		# Plot contrats + all for other lgs
		else:
			BC_feat_val_list = make_list_contrast(BC)
			BM_feat_val_list = make_list_contrast(BM)
			CM_feat_val_list = make_list_contrast(CM)
			plot_feat(BC_feat_val_list, args.features_csv, args.metric, args.lg, title + ', B vs. C')
			plot_feat(BM_feat_val_list, args.features_csv, args.metric, args.lg, title + ', B vs. M')
			plot_feat(CM_feat_val_list, args.features_csv, args.metric, args.lg, title + ', C vs. M')
			plot_feat_flipped(BC_feat_val_list, args.features_csv, args.metric, args.lg, title + ', B vs. C')
			plot_feat_flipped(BM_feat_val_list, args.features_csv, args.metric, args.lg, title + ', B vs. M')
			plot_feat_flipped(CM_feat_val_list, args.features_csv, args.metric, args.lg, title + ', C vs. M')
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
	if args.metric == "abl-cat":
		# SVM
		data, svm, rf = get_data_abl_cat(args.lg, args.svm_acc, args.rf_acc)
		feat_val_list_svm = make_list_contrast_abl(svm)
		title_svm = append_title(title, 'svm')
		feat_val_list_rf = make_list_contrast_abl(rf)
		title_rf = append_title(title, 'rf')
		plot_feat_cat(feat_val_list_svm, feat_val_list_rf, args.features_csv, args.lg, title_svm)
		#plot_feat(feat_val_list, args.features_csv, args.metric, args.lg, title_svm)
		# RF

		#plot_feat(feat_val_list, args.features_csv, args.metric, args.lg, title_rf)
	

# TODO: Add titles to plots
# TODO: Add ablation

if __name__ == "__main__":
	main()