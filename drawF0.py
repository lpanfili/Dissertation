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
	'axes.labelsize': 10,
	'text.fontsize': 10,
	'legend.fontsize': 10,
	'xtick.labelsize': 8,
	'ytick.labelsize': 8,
	'text.usetex': True}
plt.rcParams.update(params)

def readData(filename):
	with open(filename) as f:
		reader = csv.reader(f)
		next(reader)
		for line in reader:
			time.append(line[1])
			strF0.append(line[2])
			sF0.append(line[3])
			pF0.append(line[4])
			shrF0.append(line[5])

def draw():
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
def main():
	readData("/Users/Laura/Desktop/Dissertation/Code/breathy-F0.csv")
	draw()
	#plt.savefig('/Users/Laura/Desktop/')

if __name__ == "__main__":
    main()