import pandas as pd
import numpy as np
import sklearn
from sklearn import svm, metrics
from sklearn.svm import SVC
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import StratifiedKFold
import argparse
from imblearn.over_sampling import SMOTE
from collections import Counter
import csv

# Arguments
# 1. CSV with data
# 2. CSV with features metadata
# 3. 3-letter language code
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_csv', type=str,
                        help='a CSV file (with headers) containing the input data')
    parser.add_argument('features_csv', type=str,
                        help='a CSV file (with header) containing the features metadata')
    parser.add_argument('lg', type=str, help='the three digit code for the language in question')
    return parser.parse_args()

# Makes a list of features based on the features metadata csv
# Uses only features indicated with 1 in the csv, and features with <15% udef
# Makes lists of features normalized in different ways (by speaker, overall, not at all)
def getFeatures(features_csv, lg):
    features = []
    BY_SPEAKER = [] # features to be normalized by speaker
    NO_NORMALIZE = [] # features that will not be normalized
    TO_NORMALIZE = [] # features to be normalized overall
    with open(features_csv) as f:
        reader = csv.reader(f)
        header = next(reader)
        for i in range(len(header)):
            if header[i] == lg:
                lgIndex = i
        # Pick only features marked as 1 and with < 15% udef
        for line in reader:
            if line[lgIndex] == '1':
                if float(line[lgIndex + 1]) < 15.0:
                    features.append(line[0])
                    if line[1] == '1':
                        NO_NORMALIZE.append(line[0])
                    if line[2] == '1':
                        BY_SPEAKER.append(line[0])
                    if line[1] == "0" and line[2] == "0":
                        TO_NORMALIZE.append(line[0])
    return features, BY_SPEAKER, NO_NORMALIZE, TO_NORMALIZE

# Replaces undefineds and zeros with the mean
# If the mean is undefined, replaces with 0
def fillNaOrZero(x):
    if not np.isnan(x.mean()): # If it's a number
        return x.fillna(x.mean())
    else: # If it's NaN
        return x.fillna(0)


def main():
    args = parse_args()
    features, BY_SPEAKER, NO_NORMALIZE, TO_NORMALIZE = getFeatures(args.features_csv, args.lg)
    # Replace undefined and 0 with NA
    data = pd.read_csv(args.input_csv, na_values=["--undefined--",0])
    # Define z-score
    zscore = lambda x: (x - x.mean())/x.std()
    # Normalize some by speaker
    normalized_by_speaker = data[BY_SPEAKER+["speaker"]].groupby("speaker").transform(zscore)
    # Normalize some overall
    normalized = zscore(data[TO_NORMALIZE])
    # The features that won't be normalized
    notnormalized = data[NO_NORMALIZE]
    # List of phonation types
    y = data['phonation'].tolist()
    # Returns all the normalized (or not) data to one place
    normalized = pd.concat([normalized, normalized_by_speaker, notnormalized], axis=1)
    # Makes dummy df with phonation back as a column
    normalizedP = pd.concat([normalized, data['phonation']], axis=1)
    # Convert phonation type into three binary features
    normalizedP['modal'] = normalizedP.apply(lambda row: int(row['phonation'] == 'M'), axis = 1)
    normalizedP['creaky'] = normalizedP.apply(lambda row: int(row['phonation'] == 'C'), axis = 1)
    normalizedP['breathy'] = normalizedP.apply(lambda row: int(row['phonation'] == 'B'), axis = 1)
    # Make dfs that exclude one phonation type
    BC = normalizedP.drop(normalizedP[normalizedP.phonation == 'M'].index)
    BM = normalizedP.drop(normalizedP[normalizedP.phonation == 'C'].index)
    CM = normalizedP.drop(normalizedP[normalizedP.phonation == 'B'].index)
    # Make one vs one correlation matrices
    bcCorr = BC.corr()['breathy']
    bmCorr = BM.corr()['breathy']
    cmCorr = CM.corr()['creaky']
    corrMat = pd.concat([bcCorr, bmCorr, cmCorr], axis = 1)
    """
    # Make individual correlation matrices
    mCorr = normalizedP.corr()['modal']
    bCorr = normalizedP.corr()['breathy']
    cCorr = normalizedP.corr()['creaky']
    # Combine them and save it
    corrMat = pd.concat([bCorr, mCorr, cCorr], axis = 1)
    """
    # Get latex feature name from metadata
    metadata = pd.read_csv(args.features_csv, index_col='feature')
    corrMat = corrMat.round(decimals=3)
    corrMat['latex-feature'] = metadata['feature-latex']
    path = '/Users/Laura/Desktop/Dissertation/data/correlations/correlationMatrixNorm-' + args.lg + '.csv'
    corrMat.to_csv(path)
    """
    # Calculating the percent undefined
    udefcount = []
    udefDict = {}
    for feature in features:
        nonNullCount = normalized[feature].count()
        fullCount = normalized[feature].fillna(0).count()
        percent = (fullCount-nonNullCount)/fullCount
        udefDict[feature] = percent
        #percent = count * 100
        #udefcount.append(str(round(percent,3)))
    """
    x = normalized[features]

if __name__ == "__main__":
    main()