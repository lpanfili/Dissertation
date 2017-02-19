import csv
from operator import itemgetter
"""
with open("/Users/Laura/Desktop/Dissertation/ATAROS/results.txt") as f:
	words = {}
	reader = csv.reader(f)
	header = next(reader)
	for line in reader:
		if line[2] not in words:
			words[line[2]] = 0
		words[line[2]] += 1
	w = csv.writer(open("/Users/Laura/Desktop/Dissertation/ATAROS/wordcount.csv", "w"))
	for key, val in reversed(sorted(words.items(), key = itemgetter(1))):
		w.writerow([key, val])
	"""
stopWords = []
with open("/Users/Laura/Desktop/Dissertation/test-data/phonetic_stoplist.txt") as f:
	for line in f:
		line = line.strip().upper()
		stopWords.append(line)

with open("/Users/Laura/Desktop/Dissertation/ATAROS/results.txt") as f:
	words = {}
	reader = csv.reader(f)
	header = next(reader)
	for line in reader:
		if line[2] not in words:
			words[line[2]] = [0]
		words[line[2]][0] += 1
	for key in words:
		if key in stopWords:
			words[key].append("x")
		else:
			words[key].append("")
w = csv.writer(open("/Users/Laura/Desktop/Dissertation/ATAROS/wordcount.csv", "w"))
for key, val in reversed(sorted(words.items(), key = itemgetter(1))):
	w.writerow([key, val[0], val[1]])