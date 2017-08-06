# Given an almost-complete data set, calculates and adds VoPT

import csv
import numpy as np
import itertools

w = csv.writer(open("/Users/Laura/Desktop/Dissertation/data/english/eng-all.csv", "w"))
with open("/Users/Laura/Desktop/Dissertation/data/english/eng-all0.csv") as f:
	reader = csv.reader(f)
	header = next(reader)
	newheader = header
	newheader.insert(119, "VoPT")
	w.writerow(newheader)
	for line in reader:
		VoPT = 0
		strF0 = line[103]
		sF0 = line[107]
		pF0 = line[111]
		shrF0 = line[115]
		tracks = [strF0, sF0, pF0, shrF0]
		pairs = (list(itertools.combinations(tracks, 2)))
		for pair in pairs:
			a = float(pair[0])
			b = float(pair[1])
			diff = abs(a - b)
			VoPT += diff
		line.insert(119, VoPT)
		w.writerow(line)

for i in range(len(newheader)):
	if newheader[i] == "preceding_phone":
		print("Pre", i)
	if newheader[i] == "following_phone":
		print("Fol", i)