# Calculates and plots the MEAN VoPT
# Overall
# By speaker

import matplotlib.pyplot as plt
from matplotlib import rc
import csv
import itertools
import pandas as pd

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
def read_data():
	path = '../data/lgs/eng/eng-all.csv'
	data_all = pd.read_csv(path)
	speaker = data_all['Filename'].copy()
	phonation = data_all['Label'].copy()
	strF0 = data_all['strF0_mean'].copy()
	sF0 = data_all['sF0_mean'].copy()
	pF0 = data_all['pF0_mean'].copy()
	shrF0 = data_all['shrF0_mean'].copy()
	data = pd.concat([speaker, phonation, strF0, sF0, pF0, shrF0], axis = 1)
	return data
	

# Calculate VoPT for each vowel
# Returns data frame
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


# Makes a plot for each speaker
def plot_by_speaker(vopt_data):
	by_speaker = vopt_data.groupby('speaker')
	for i in by_speaker:
		data = i[1]
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
	data = read_data()
	vopt_data = get_vopt(data)
	#plot_by_speaker(vopt_data)
	#plot_all(vopt_data)
	plot_modal_nonmodal_all(vopt_data)

if __name__ == "__main__":
	main()