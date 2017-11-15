import csv
from sklearn.metrics import cohen_kappa_score

laura = []
nicole = []
lauraBMC = [] # Collapses 0 and 1
nicoleBMC = [] # Collapses 0 and 1

with open("/Users/Laura/Desktop/Dissertation/interrater-reliability/results.csv") as f:
	reader = csv.reader(f)
	for line in reader:
		laura.append(line[3])
		nicole.append(line[7])
	
		if line[3] == "1":
			lauraBMC.append("0")
		else:
			lauraBMC.append(line[3])
		if line[7] == "1":
			nicoleBMC.append("0")
		else:
			nicoleBMC.append(line[7])

print(cohen_kappa_score(laura, nicole))
print(cohen_kappa_score(lauraBMC, nicoleBMC))
