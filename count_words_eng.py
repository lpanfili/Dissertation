import csv
import pandas as pd

words = {}

with open("/Users/Laura/Desktop/Dissertation/data/english/eng-all.csv") as f:
	reader = csv.reader(f)
	header = next(reader)
	for line in reader:
		word = line[6]
		if word not in words:
			words[word] = 0
		words[word] += 1

counts = []
for key in words:
	counts.append([words[key], key])

counts = sorted(counts, reverse = True)

for line in counts:
	count = "".join(["(", str(line[0]), ")", " \\\\"])
	wordtt = "".join(["\\texttt{", line[1], "}"])
	print(wordtt, count)

