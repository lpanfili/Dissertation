# Counts the number of undefined measures for the various shimmer and jitter calculations

import csv

check = [11, 15, 19, 23, 27, 31, 35, 39, 43]
udefDict = {11:0, 15:0, 19:0, 23:0, 27:0, 31:0, 35:0, 39:0, 43:0}

with open("/Users/Laura/Desktop/Dissertation/data/zapotec/zap-all.csv") as f:
	reader = csv.reader(f)
	header = next(reader)
	for line in reader:
		for i in check:
			if line[i] == "--undefined--":
				udefDict[i] += 1

jitter = str(udefDict[11]) + " & " + str(udefDict[15]) + " & " + str(udefDict[19]) + " & " + str(udefDict[23])
print(jitter)
shimmer = str(udefDict[27]) + " & " + str(udefDict[31]) + " & " + str(udefDict[35]) + " & " + str(udefDict[39]) + " & " + str(udefDict[43])
print(shimmer)



"""
		jitter_loc = line[11]
		jitter_loc_abs = line[15]
		jitter_rap = line[19]
		jitter_ppq5 = line[23]
		shimmer_loc = line[27]
		shimmer_loc_db = line[31]
		shimmer_apq3 = line[35]
		shimmer_apq5 = line[39]
		shimmer_apq11 = line[43]
"""

