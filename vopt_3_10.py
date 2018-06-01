# Calculates and plots the RMSE 3 and 10 VoPT
# Overall
# By speaker

import matplotlib.pyplot as plt
from matplotlib import rc
import csv
import itertools
import pandas as pd
from sklearn.metrics import mean_squared_error
import math

# Set LaTeX font
def setFont():
	rc('font',**{'family':'serif','serif':['Computer Modern Roman']})
	params = {'backend': 'ps',
		'axes.labelsize': 25,
		'text.fontsize': 25,
		'legend.fontsize': 10,
		'xtick.labelsize': 15,
		'ytick.labelsize': 15,
		'text.usetex': True}
	plt.rcParams.update(params)


# Read data, return as df
def read_data_3():
	path = '../data/lgs/eng/eng-all.csv'
	data_all = pd.read_csv(path)
	speaker = data_all['Filename'].copy()
	phonation = data_all['Label'].copy()
	strF0_1 = data_all['strF0_means001'].copy()
	strF0_2 = data_all['strF0_means002'].copy()
	strF0_3 = data_all['strF0_means003'].copy()
	sF0_1 = data_all['sF0_means001'].copy()
	sF0_2 = data_all['sF0_means002'].copy()
	sF0_3 = data_all['sF0_means003'].copy()
	pF0_1 = data_all['pF0_means001'].copy()
	pF0_2 = data_all['pF0_means002'].copy()
	pF0_3 = data_all['pF0_means003'].copy()
	shrF0_1 = data_all['shrF0_means001'].copy()
	shrF0_2 = data_all['shrF0_means002'].copy()
	shrF0_3 = data_all['shrF0_means003'].copy()
	data = pd.concat([speaker, phonation, strF0_1, strF0_2, strF0_3, \
					sF0_1, sF0_2, sF0_3, pF0_1, pF0_2, pF0_3, shrF0_1, shrF0_2, shrF0_3], axis = 1)
	return data
	

# Calculate mean VoPT for each vowel
# Return data frame
# Speaker, phonation, VoPT
def get_vopt(data):
	vopt_data = []
	for index, row in data.iterrows():
		VoPT = 0
		speaker = row[0]
		phonation = row[1]
		strF0 = row[2]
		sF0 = row[3]
		pF0 = row[4]
		shrF0 = row[5]
		tracks = [strF0, sF0, pF0, shrF0]
		pairs = (list(itertools.combinations(tracks, 2)))
		for pair in pairs:
			a = float(pair[0])
			b = float(pair[1])
			diff = abs(a - b)
			VoPT += diff
			new_line = [speaker, phonation, VoPT]
			vopt_data.append(new_line)
	vopt_data = pd.DataFrame(vopt_data, columns = ['speaker', 'phonation', 'vopt'])
	return vopt_data

# Calculate RMSE3 VoPT for each vowel
# Return data frame
# Speaker, phonation, VoPT
def get_vopt_3(data):
	vopt_data = []
	for index, row in data.iterrows():
		VoPT = 0
		speaker = row[0]
		phonation = row[1]
		strF0 = [row[2], row[3], row[4]]
		sF0 = [row[5], row[6], row[7]]
		pF0 = [row[8], row[9], row[10]]
		shrF0 = [row[11], row[12], row[13]]
		tracks = [strF0, sF0, pF0, shrF0]
		pairs = (list(itertools.combinations(tracks, 2)))
		for pair in pairs:
			a = pair[0]
			b = pair[1]
			rmse = math.sqrt(mean_squared_error(a, b))
			VoPT += rmse
			new_line = [speaker, phonation, VoPT]
			vopt_data.append(new_line)
	vopt_data = pd.DataFrame(vopt_data, columns = ['speaker', 'phonation', 'vopt'])
	return vopt_data


# Makes a plot for each speaker
def plot_by_speaker(vopt_data):
	by_speaker = vopt_data.groupby('speaker')
	for i in by_speaker:
		data = i[1]
		print(type(data))
		print(len(data))
		B, C, M = data.groupby('phonation')['vopt'].apply(list)
		speaker = data['speaker'].iloc[0]
		plt.boxplot([B, M, C], labels = ["Breathy", "Modal", "Creaky"], showmeans = True)
		plt.title(speaker)
		plt.show()
		#plt.savefig(''.join(["/Users/Laura/Desktop/Dissertation/Dissertation/Appendices/VoPT-all/images/", speaker]), dpi = "figure")
		plt.clf()


# Makes a plot of all the data
def plot_all(vopt_data):
	B, C, M = vopt_data.groupby('phonation')['vopt'].apply(list)
	plt.boxplot([B, M, C], labels = ["Breathy", "Modal", "Creaky"], showmeans = True)
	plt.show()


# Makes a plot of all the data
# Modal vs. Non-Modal
def plot_modal_nonmodal_all(vopt_data):
	# Add column for whether or not something is modal
	vopt_data['modal'] = vopt_data.apply(lambda row: int(row['phonation'] == 'M'), axis = 1)
	NM, M = vopt_data.groupby('modal')['vopt'].apply(list)
	plt.boxplot([M, NM], labels = ["Modal", "Non-Modal"], showmeans = True)
	plt.show()


def main():
	setFont()
	data = read_data_3()
	vopt_data = get_vopt_3(data)
	plot_by_speaker(vopt_data)
	#plot_all(vopt_data)
	#plot_modal_nonmodal_all(vopt_data)


if __name__ == "__main__":
	main()