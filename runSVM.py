# Runs an SVM based on all features indicated in a metadata spreadsheet
# Can be resampled or not
# Outputs accuracy and by-class prf
# Outputs feature weights

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
import re

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

# DOES NOT RESAMPLE
def runClassImb(x, y, features,lg, feat_csv):
    clf = svm.SVC(kernel = 'linear')
    skf = StratifiedKFold(n_splits=5)
    sm = SMOTE(random_state=42)
    x = x.as_matrix()
    y = np.array(y)
    y_pred_all = np.array([])
    y_test_all = np.array([])
    accuracyList = []
    # Indented block happens within the fold
    for train_index, test_index in skf.split(x,y):
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]
        # Replace undefined
        x_train = pd.DataFrame(x_train)
        x_test = pd.DataFrame(x_test)
        x_train = x_train.groupby(y_train).transform(fillNaOrZero)
        x_test = x_test.groupby(y_test).transform(fillNaOrZero)
        # Fit classifier
        clf.fit(x_train, y_train)
        y_pred = clf.predict(x_test)
        y_pred_all = np.append(y_pred_all, y_pred)
        y_test_all = np.append(y_test_all, y_test)
    #confmat = confusion_matrix(y_test_all, y_pred_all)
    report = classification_report(y_test_all, y_pred_all)
    acc = accuracy_score(y_test_all, y_pred_all)
    acc = round((acc * 100),3)
    # Feature weights
    getWeights(clf.coef_, features, 'imb', lg, feat_csv)
    return report, acc

# RESAMPLES
def runClassRS(x, y, features,lg, feat_csv):
    clf = svm.SVC(kernel = 'linear')
    skf = StratifiedKFold(n_splits=5)
    sm = SMOTE(random_state=42)
    x = x.as_matrix()
    y = np.array(y)
    y_pred_all = np.array([])
    y_test_all = np.array([])
    accuracyList = []
    # Indented block happens within the fold
    for train_index, test_index in skf.split(x,y):
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]
        # Replace undefined
        x_train = pd.DataFrame(x_train)
        x_test = pd.DataFrame(x_test)
        x_train = x_train.groupby(y_train).transform(fillNaOrZero)
        x_test = x_test.groupby(y_test).transform(fillNaOrZero)
        # Resample (twice because there are three classes)
        x_res, y_res = sm.fit_sample(x_train, y_train)
        x_res, y_res = sm.fit_sample(x_res, y_res)
        # Fit classifier
        clf.fit(x_res, y_res)
        y_pred = clf.predict(x_test)
        y_pred_all = np.append(y_pred_all, y_pred)
        y_test_all = np.append(y_test_all, y_test)
    #confmat = confusion_matrix(y_test_all, y_pred_all)
    report = classification_report(y_test_all, y_pred_all)
    acc = accuracy_score(y_test_all, y_pred_all)
    acc = round((acc * 100),3)
    # Feature weights
    getWeights(clf.coef_, features, 'rs', lg, feat_csv)
    return report, acc

def getWeights(coef, features, samp, lg, feat_csv):
    if lg == 'cmn':
        CvM = coef[0]
        CvMWeights = list(zip(features,CvM))
        weights = pd.DataFrame(CvMWeights, columns = ['feat', 'CM'])
        weights = weights.set_index('feat')
    elif lg == 'guj':
        BvM = coef[0]
        BvMWeights = list(zip(features,BvM))
        weights = pd.DataFrame(BvMWeights, columns = ['feat', 'BM'])
        weights = weights.set_index('feat')
    else:
        BvC = coef[0]
        BvM = coef[1]
        CvM = coef[2]
        BvCWeights = list(zip(features,BvC))
        weights = pd.DataFrame(BvCWeights, columns = ['feat', 'BC'])
        weights = weights.set_index('feat')
        weights['BM'] = BvM
        weights['CM'] = CvM
    path = "/Users/Laura/Desktop/Dissertation/data/weights/SVM-" + lg + "-" + samp + ".csv"
    metadata = pd.read_csv(feat_csv, index_col='feature')
    weights['latex-feat'] = metadata['feature-latex']
    weights = weights.set_index('latex-feat')
    weights.to_csv(path)

def classifaction_report(report, lg):
    report_data = []
    lines = report.split('\n')
    if lg == 'cmn':
        for line in lines[2:-3]:
            row = []
            row_data = re.split('       |      ',line)
            row = row_data[1].strip(), float(row_data[2]), float(row_data[3]), float(row_data[4])
            report_data.append(row)
        cP = str(report_data[0][1])
        cR = str(report_data[0][2])
        cF = str(report_data[0][3])
        mP = str(report_data[1][1])
        mR = str(report_data[1][2])
        mF = str(report_data[1][3])
        prf = "--" + "&" + mP + "&" + cP + "&" + "--" + "&" + mR + "&" + cR + "&" + "--" + "&" + mF + "&" + cF
    elif lg == 'guj':
        for line in lines[2:-3]:
            row = []
            row_data = re.split('       |      ',line)
            row = row_data[1].strip(), float(row_data[2]), float(row_data[3]), float(row_data[4])
            report_data.append(row)
        bP = str(report_data[0][1])
        bR = str(report_data[0][2])
        bF = str(report_data[0][3])
        mP = str(report_data[1][1])
        mR = str(report_data[1][2])
        mF = str(report_data[1][3])
        prf = bP + "&" + mP + "&" + "--" + "&" + bR + "&" + mR + "&" + "--" + "&" + bF + "&" + mF + "&" + "--"
    else:
        for line in lines[2:-3]:
            row = []
            row_data = re.split('       |      ',line)
            row = row_data[1].strip(), float(row_data[2]), float(row_data[3]), float(row_data[4])
            report_data.append(row)
        bP = str(report_data[0][1])
        bR = str(report_data[0][2])
        bF = str(report_data[0][3])
        cP = str(report_data[1][1])
        cR = str(report_data[1][2])
        cF = str(report_data[1][3])
        mP = str(report_data[2][1])
        mR = str(report_data[2][2])
        mF = str(report_data[2][3])
        prf = bP + "&" + mP + "&" + cP + "&" + bR + "&" + mR + "&" + cR + "&" + bF + "&" + mF + "&" + cF
    print(prf)

def main():
    args = parse_args()
    features, BY_SPEAKER, NO_NORMALIZE, TO_NORMALIZE = getFeatures(args.features_csv, args.lg)
    # Replace undefined and 0 with NA
    data = pd.read_csv(args.input_csv, na_values=["--undefined--",0])
    # Define z-score
    zscore = lambda x: (x - x.mean())/x.std()
    # Normalize some features by speaker
    normalized_by_speaker = data[BY_SPEAKER+["speaker"]].groupby("speaker").transform(zscore)
    # Normalize some features overall
    normalized = zscore(data[TO_NORMALIZE])
    # The features that won't be normalized
    notnormalized = data[NO_NORMALIZE]
    # List of phonation types
    y = data['phonation'].tolist()
    # Returns all the normalized (or not) data to one place
    normalized = pd.concat([normalized, normalized_by_speaker, notnormalized], axis=1)
    x = normalized[features]
    #report, acc = runClassImb(x,y,features,args.lg, args.features_csv)
    report, acc = runClassRS(x,y,features,args.lg, args.features_csv)
    # Print prf in LaTeX-friendly format
    classifaction_report(report, args.lg)
    print('accuracy', acc)

if __name__ == "__main__":
    main()