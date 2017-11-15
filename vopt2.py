# Calculates the VoPT for each vowel in the entire data set
# Based on the combined Praat and VS data, which excludes 0, 1, and stopwords

import matplotlib.pyplot as plt
from matplotlib import rc
import itertools
import pandas as pd

# Use LaTeX font
def set_font():
	rc('font',**{'family':'serif','serif':['Computer Modern Roman']})
	params = {'backend': 'ps',
		'axes.labelsize': 25,
		'text.fontsize': 25,
		'legend.fontsize': 10,
		'xtick.labelsize': 15,
		'ytick.labelsize': 15,
		'text.usetex': True}
	plt.rcParams.update(params)


# Calculate VoPT for each vowel
# Return df containing speaker, phonation, and vopt
def get_vopt(data):
	vopt_data_list = []
	for index, row in data.iterrows():
		vopt = 0
		speaker = row['speaker']
		phonation = row['phonation']
		strF0 = row['strF0_mean']
		sF0 = row['sF0_mean']
		pF0 = row['pF0_mean']
		shrF0 = row['shrF0_mean']
		tracks = [strF0, sF0, pF0, shrF0]
		pairs = (list(itertools.combinations(tracks, 2)))
		for pair in pairs:
			a = float(pair[0])
			b = float(pair[1])
			diff = abs(a - b)
			vopt += diff
		newLine = [speaker, phonation, vopt]
		vopt_data_list.append(newLine)
	vopt = pd.DataFrame(vopt_data_list, columns = ['speaker', 'phonation', 'vopt'])
	return vopt


# Plot for all speakers combined
def plot_mean(vopt):
	# Make a list of vopts for each phonation type
	B, C, M = vopt.groupby('phonation')['vopt'].apply(list)
	# Plot
	plt.boxplot([B, M, C], labels = ["B", "M", "C"], showmeans = True)
	plt.title("Variance of Pitch Tracks")
	plt.show()


# Plots modal vs. non-modal for all speakers combined
def plot_m_nm(vopt):
	# Add column for whether or not something is modal
	vopt['modal'] = vopt.apply(lambda row: int(row['phonation'] == 'M'), axis = 1)
	NM, M = vopt.groupby('modal')['vopt'].apply(list)
	plt.boxplot([M, NM], labels = ["Modal", "Non-Modal"], showmeans = True)
	plt.title("Variance of Pitch Tracks")
	plt.show()


# Plots B M C separately per speaker
def plot_by_speaker(vopt):
	by_speaker = vopt.groupby('speaker')
	for i in by_speaker:
		data = i[1]
		B, C, M = data.groupby('phonation')['vopt'].apply(list)
		speaker = data['speaker'].iloc[0]
		plt.boxplot([B, M, C], labels = ["Breathy", "Modal", "Creaky"], showmeans = True)
		plt.title(speaker)
		path = "/Users/Laura/Desktop/Dissertation/Dissertation/Appendices/VoPT-all/images/" + speaker
		plt.savefig(path, dpi = 'figure')
		plt.clf()
		#plt.show()

def main():
	set_font()
	data = pd.read_csv("/Users/Laura/Desktop/Dissertation/data/lgs/eng/eng.csv")
	vopt = get_vopt(data)
	#plot_mean(vopt) # Plot for all speakers together, B M C
	#plot_m_nm(vopt) # Plot for all speakers together, modal vs. non-modal
	plot_by_speaker(vopt)


if __name__ == "__main__":
	main()