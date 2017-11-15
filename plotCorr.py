# Takes in the output of correlations
# Plots the percent of each lg's features that are weakly, moderately, and strongly correlated

from matplotlib import rc
import csv
import matplotlib.pyplot as plt
import numpy as np

def setFont():
	rc('font',**{'family':'serif','serif':['Computer Modern Roman']})
	params = {'backend': 'ps',
		'axes.labelsize': 20,
		'text.fontsize': 15,
		'legend.fontsize': 15,
		'xtick.labelsize': 15,
		'ytick.labelsize': 15,
		'text.usetex': True}
	plt.rcParams.update(params)

def makeList(lg):
	breathy = []
	modal = []
	creaky = []
	path = '/Users/Laura/Desktop/Dissertation/data/correlations/correlationMatrixNorm-' + lg + '.csv'
	with open(path) as f:
		reader = csv.reader(f)
		header = next(reader)
		for line in reader:
			if line[0] != "modal" and line[0] != "creaky" and line[0] != "breathy" and line[0] != " ":
				if line[1] != "":
					breathy.append(abs(float(line[1])))
				modal.append(abs(float(line[2])))
				if line[3] != "":
					creaky.append(abs(float(line[3])))
	total = len(modal)
	lgDict = {'W':0, 'M':0, 'S':0}
	for i in range(total):
		if modal[i] < 0.3:
			lgDict['W'] += 1
		elif modal[i] < 0.5:
			lgDict['M'] += 1
		else:
			lgDict['S'] += 1
		if breathy != []:
			if breathy[i] < 0.3:
				lgDict['W'] += 1
			elif breathy[i] < 0.5:
				lgDict['M'] += 1
			else:
				lgDict['S'] += 1
		if creaky != []:
			if creaky[i] < 0.3:
				lgDict['W'] += 1
			elif creaky[i] < 0.5:
				lgDict['M'] += 1
			else:
				lgDict['S'] += 1
	W = lgDict['W'] # Count of weak correlations
	M = lgDict['M']
	S = lgDict['S']
	if lg == 'guj' or lg == 'cmn':
		wPer = float((W/(total * 2))*100)
		mPer = float((M/(total * 2))*100)
		sPer = float((S/(total * 2))*100)
		wmslist = [wPer, mPer, sPer]
	else:
		wPer = float((W/(total * 3))*100)
		mPer = float((M/(total * 3))*100)
		sPer = float((S/(total * 3))*100)
	wmslist = [wPer, mPer, sPer]
	return wmslist

def main():
	setFont()
	engList = makeList('eng')
	gujList = makeList('guj')
	hmnList = makeList('hmn')
	cmnList = makeList('cmn')
	majList = makeList('maj')
	zapList = makeList('zap')
	# PLOT
	wList = [engList[0], gujList[0], hmnList[0], cmnList[0], majList[0], zapList[0]]
	mList = [engList[1], gujList[1], hmnList[1], cmnList[1], majList[1], zapList[1]]
	sList = [engList[2], gujList[2], hmnList[2], cmnList[2], majList[2], zapList[2]]
	n_groups = 6
	index = np.arange(n_groups)
	plt.bar(index, wList, color = 'red', label = "Weak")
	plt.bar(index, mList, bottom = wList, color = 'yellow', label = "Moderate")
	plt.bar(index, sList, bottom = (np.array(mList) + np.array(wList)), 
		color = 'green', label = "Strong")
	plt.xlabel('Language')
	plt.ylabel('Percent of Features')
	plt.title('Feature Correlation Strength')
	langs = ['eng','guj','hmn','cmn','maj','zap']
	bar_width = 0.3
	plt.xticks(index + (bar_width), langs)
	plt.ylim([0,105])
	plt.legend(loc = 3)
	plt.show()


if __name__ == "__main__":
	main()
			