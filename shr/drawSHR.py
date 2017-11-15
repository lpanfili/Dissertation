import matplotlib.pyplot as plt
import csv
from matplotlib import rc

strF0 = []
sF0 = []
pF0 = []
shrF0 = []
time = []

rc('font',**{'family':'serif','serif':['Computer Modern Roman']})
params = {'backend': 'ps',
	'axes.labelsize': 20,
	'text.fontsize': 10,
	'legend.fontsize': 10,
	'xtick.labelsize': 20,
	'ytick.labelsize': 20,
	'text.usetex': True}
plt.rcParams.update(params)

def readData(filename):
	bSHR = []
	mSHR = []
	cSHR = []
	with open(filename) as f:
		reader = csv.reader(f, delimiter = "\t")
		next(reader)
		for line in reader:
			shr = [line[4], line[5]]
			if line[0].startswith('B'):
				bSHR.append(shr)
			if line[0].startswith('M'):
				mSHR.append(shr)
			if line[0].startswith('C'):
				cSHR.append(shr)
	return bSHR, mSHR, cSHR


def draw(data):
	time, shr = zip(*data)
	plt.scatter(time, shr, marker = '.', color = 'black')
	plt.xlabel('Time (ms)')
	plt.ylabel('SHR')
	plt.axis([0, 160, 0, 1.0])
	plt.show()
	"""
	plt.scatter(time, shrF0, marker = 'o', color = [0,0,0,.6], label = 'SHR F0')
	plt.scatter(time, pF0, marker = 'o', color = [0,0,1,.5], label = 'Praat F0')
	plt.scatter(time, strF0, marker = 'o', color = [1,0,0,.4], label = 'STRAIGHT F0')
	plt.scatter(time, sF0, marker = 'o', color = [0,1,0,.3], label = 'Snack F0')
	plt.axis([0, 140, 0, 300])
	plt.xlabel('Time (ms)', fontsize = 25)
	plt.ylabel('Hz', fontsize = 25)
	plt.title('Breathy Pitch Tracks', fontsize = 35)
	#plt.legend(fontsize = 12)
	plt.show()
	"""

def main():
	bSHR, mSHR, cSHR = readData("/Users/Laura/Desktop/Dissertation/Code/shr/shr-output.txt")
	draw(bSHR)
	draw(mSHR)
	draw(cSHR)
	#plt.savefig('/Users/Laura/Desktop/')

if __name__ == "__main__":
    main()