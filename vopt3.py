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

# Returns a dictionary of unique vowel IDs and their word
def get_praat():
	word_dict = {}
	praat = pd.read_csv("/Users/Laura/Desktop/Dissertation/data/lgs/eng/english-praat-nov.txt", sep = '\t')
	for index, row in praat.iterrows():
		start = row['vowel_start'] * 1000
		end = row['vowel_end'] * 1000
		speaker = row['gridfile'][:6]
		phonation = str(row['phonation'])
		vowel_id = speaker + str(start)[:3] + str(end)[:3] + phonation
		word_dict[vowel_id] = row['word_label']
	return word_dict

def get_stopwords():
	stop_words = []
	with open("/Users/Laura/Desktop/Dissertation/data/phonetic_stoplist.txt") as f:
		for line in f:
			line = line.strip().upper()
			stop_words.append(line)
	return stop_words

def get_one(speaker, word_dict):
	path = "/Users/Laura/Desktop/Dissertation/Code/vopt/" + speaker + "-1.txt"
	one = pd.read_csv(path, sep = '\t')
	for index, row in one.iterrows():
		start = row['seg_Start']
		end = row['seg_End']
		speaker = row['Filename'][:6]
		phonation = str(row['Label'])
		vowel_id = speaker + str(start)[:3] + str(end)[:3] + phonation
		if vowel_id not in word_dict:
			print(vowel_id)


def main():
	set_font()
	word_dict = get_praat()
	stop_words = get_stopwords()
	#speakers = ['NWF089', 'NWF090', 'NWF092']
	speakers = ['NWF092']
	for speaker in speakers:
		one = get_one(speaker, word_dict)
	"""
	data = pd.read_csv("/Users/Laura/Desktop/Dissertation/data/lgs/eng/eng.csv")
	vopt = get_vopt(data)
	#plot_mean(vopt) # Plot for all speakers together, B M C
	#plot_m_nm(vopt) # Plot for all speakers together, modal vs. non-modal
	plot_by_speaker(vopt)
	"""


if __name__ == "__main__":
	main()