import csv
import matplotlib.pyplot as plt
from matplotlib import rc
import numpy as np

rc('font',**{'family':'serif','serif':['Computer Modern Roman']})
params = {'backend': 'ps',
	'axes.labelsize': 25,
	'text.fontsize': 25,
	'legend.fontsize': 10,
	'xtick.labelsize': 15,
	'ytick.labelsize': 15,
	'text.usetex': True}
plt.rcParams.update(params)

with open("/Users/Laura/Desktop/Dissertation/Code/acc-udef/acc-udef.csv") as f:
	reader = csv.reader(f)
	accList = []
	udefList = []
	for line in reader:
		acc = float(line[0])
		udef = float(line[1])
		accList.append(acc)
		udefList.append(udef)
	plt.scatter(udefList, accList, marker = '.')
	plt.plot(np.unique(udefList), np.poly1d(np.polyfit(udefList, accList, 1))(np.unique(udefList)), color = 'black')
	plt.xlabel("Percent Undefined")
	plt.ylabel("Accuracy")
	plt.xlim([0,100])
	plt.ylim([0,100])
	plt.show()