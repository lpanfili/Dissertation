import matplotlib.pyplot as plt
from matplotlib import rc
import random
import numpy as np

rc('font',**{'family':'serif','serif':['Computer Modern Roman']})
params = {'backend': 'ps',
	'axes.labelsize': 25,
	'text.fontsize': 20,
	'legend.fontsize': 15,
	'xtick.labelsize': 15,
	'ytick.labelsize': 15,
	'text.usetex': True}
plt.rcParams.update(params)

m, b = -1, 10
lower, upper = 0, 10
num_points = 15

x1 = [random.randrange(start=1, stop=9) for i in range(num_points)]
x2 = [random.randrange(start=1, stop=9) for i in range(num_points)]
x3 = [(7.5, 2.5)]
x4 = [(5.5, 6)]

y1 = [random.randrange(start=lower, stop=m*x+b-1) for x in x1]
y2 = [random.randrange(start=m*x+b+1, stop=upper) for x in x2]
y3 = [(1.5, 6.5)]
y4 = [(5.5, 5)]

plt.plot(np.arange(10), m*np.arange(10)+b)
plt.ylim([-1,11])
plt.xlabel("Tumor Size")
plt.ylabel("Patient Age")
plt.xlim([-1,11])
plt.tick_params(axis='both', which='both', bottom='off', 
	top='off', labelbottom='off', right='off', 
	left='off', labelleft='off')
plt.scatter(x1, y1, c='blue', label='benign')
plt.scatter(x2, y2, c='red', label='malignant')
plt.scatter(x3, y3, c='blue', marker='*', s=80)
plt.scatter(x4, y4, c='red', marker='*', s=80)
plt.title('Tumor Classification')
plt.legend()
plt.show()