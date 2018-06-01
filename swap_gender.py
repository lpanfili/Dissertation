# For a given language:
# Calculates correlations (saves CSV of all, unsorted)
# Runs a resampled SVM and RF FOR THE OTHER LANGS

import pandas as pd
import numpy as np
import sklearn
from sklearn import svm, metrics
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import StratifiedKFold
import argparse
from imblearn.over_sampling import SMOTE
import csv
import re

# Arguments:
# 1. CSV with features metadata
# 2. 3-letter language code
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('features_csv', type=str,
                        help='a CSV file (with header) containing the features metadata')
    parser.add_argument('lg', type=str, help='the three digit code for the language in question')
    return parser.parse_args()


# Makes a list of features based on the features metadata csv
# Uses only features indicated with 1 in the csv, and features with <15% udef
# Returns lists of features normalized in different ways (by speaker, overall, not at all)
# Features are in LaTeX-friendly format
# EXCLUDES eng-specific features
def get_features(features_csv, lg):
    features = []
    by_speaker = [] # features to be normalized by speaker
    no_normalize = [] # features that will not be normalized
    to_normalize = [] # features to be normalized overall
    with open(features_csv) as f:
        reader = csv.reader(f)
        header = next(reader)
        for i in range(len(header)):
            if header[i] == lg[:3]:
                lgIndex = i
        # Pick only features marked as 1 and with < 15% udef
        for line in reader:
            if line[lgIndex] == '1':
                feature_name = line[0]
                if line[3] != 'x':
                    #features.append(feature_dict[line[0]])
                    features.append(feature_name)
                if line[3] == '1':
                    #no_normalize.append(feature_dict[line[0]])
                    no_normalize.append(feature_name)
                if line[4] == '1':
                    #by_speaker.append(feature_dict[line[0]])
                    by_speaker.append(feature_name)
                if line[3] == "0" and line[4] == "0":
                    #to_normalize.append(feature_dict[line[3]])
                    to_normalize.append(feature_name)
    return features, by_speaker, no_normalize, to_normalize


# Like get_features, but without the <15% undefined requirement
def get_features_all(features_csv, lg):
    features = []
    by_speaker = [] # features to be normalized by speaker
    no_normalize = [] # features that will not be normalized
    to_normalize = [] # features to be normalized overall
    with open(features_csv) as f:
        reader = csv.reader(f)
        header = next(reader)
        for i in range(len(header)):
            if header[i] == lg:
                lgIndex = i
        for line in reader:
            feature_name = line[0]
            if line[3] != 'x':
                features.append(feature_name)
            if line[3] == '1':
                no_normalize.append(feature_name)
            if line[4] == '1':
                by_speaker.append(feature_name)
            if line[3] == "0" and line[4] == "0":
                to_normalize.append(feature_name)
    #return features, by_speaker, no_normalize, to_normalize
    # Remove eng-specific features but keep all others
    features_clean = []
    by_speaker_clean = []
    no_normalize_clean = []
    to_normalize_clean = []
    skip_eng = ['word_per', 'utt_per', 'ms_from_word_end', 'ms_from_utt_end', 'pre_is_voiced', 'fol_is_voiced', 'pre_is_obs', 'fol_is_obs', 'pre_exists', 'fol_exists']
    for i in features:
        if i not in skip_eng:
            features_clean.append(i)
    for i in by_speaker:
        if i not in skip_eng:
            by_speaker_clean.append(i)
    for i in no_normalize:
        if i not in skip_eng:
            no_normalize_clean.append(i)
    for i in to_normalize:
        if i not in skip_eng:
            to_normalize_clean.append(i)
    return features_clean, by_speaker_clean, no_normalize_clean, to_normalize_clean


# Reads in a CSV of the data
# Normalizes data
# Returns x (features), y (labels), and the whole normalized data set
def read_norm(lg, by_speaker, no_normalize, to_normalize, features):
    lang = lg[:3]
    csv = "../data/lgs/" + lang + "/" + lang + "-all" + "-" + lg[4] + ".csv"
    # Read CSV, replace undefined and 0 with NA
    data = pd.read_csv(csv, na_values=["--undefined--",0])
    zscore = lambda x: (x - x.mean())/x.std() # Define z-score
    # Normalize some features by speaker
    normalized_by_speaker = data[by_speaker+["speaker"]].groupby("speaker").transform(zscore)
    # Normalize some features overall
    normalized = zscore(data[to_normalize])
    # The features that won't be normalized
    notnormalized = data[no_normalize]
    # List of phonation types
    y = data['phonation'].tolist()
    # Returns all the normalized (or not) data to one place
    normalized = pd.concat([normalized, normalized_by_speaker, notnormalized], axis=1)
    x = normalized[features]
    return x, y, normalized


# Replaces undefineds and zeros with the mean
# If the mean is undefined, replaces with 0
def fill_na_zero(x):
    if not np.isnan(x.mean()): # If it's a number
        return x.fillna(x.mean())
    else: # If it's NaN
        return x.fillna(0)

"""
def SVM_rs(x, y, features, lg, xs, ys):
    metrics_all = []
    lgs = ['eng', 'guj', 'hmn', 'cmn', 'maj', 'zap']
    clf = svm.SVC(kernel = 'linear')
    skf = StratifiedKFold(n_splits=5)
    sm = SMOTE(random_state=42)
    x = x.as_matrix()
    y = np.array(y)
    y_pred_all = np.array([])
    y_test_all = np.array([])
    accuracyList = []
    for i in range(len(xs)): # loop through the lgs to test on
        y_pred_all = np.array([])
        y_test_all = np.array([])
        # Indented block happens within the fold
        for train_index, test_index in skf.split(x,y):
            x_train, x_test = x[train_index], x[test_index]
            y_train, y_test = y[train_index], y[test_index]
            # Replace undefined
            x_train = pd.DataFrame(x_train)
            x_test = pd.DataFrame(x_test)
            x_train = x_train.groupby(y_train).transform(fill_na_zero)
            x_test = x_test.groupby(y_test).transform(fill_na_zero)
            # Resample (twice because there are three classes)
            x_res, y_res = sm.fit_sample(x_train, y_train)
            x_res, y_res = sm.fit_sample(x_res, y_res)
            # Fit classifier
            clf.fit(x_res, y_res)
            # test
            x_test = xs[i].groupby(np.array(ys[i])).transform(fill_na_zero)
            y_pred = clf.predict(x_test)
            y_pred_all = np.append(y_pred_all, y_pred)
            y_test_all = np.append(y_test_all, ys[i])
        report = classification_report(y_test_all, y_pred_all)
        prf = clean_report(report, lgs[i])
        acc = round((accuracy_score(y_test_all, y_pred_all)*100),3)
        metrics = [str(round(acc,3))] + prf
        metrics_all.append(metrics)
    return metrics_all
"""

def SVM_imb(x, y, features, lg, xs, ys):
    metrics_all = []
    lgs = ['eng-f']
    clf = svm.SVC(kernel = 'rbf', C = 100, gamma = 0.001)
    x = x.as_matrix()
    y = np.array(y)
    y_pred_all = np.array([])
    y_test_all = np.array([])
    # Replace undefined
    x = pd.DataFrame(x)
    #x = x.groupby(y).transform(fill_na_zero)
    x = x.apply(fill_na_zero)
    # Fit classifier
    clf.fit(x, y)
    for i in range(len(xs)): # loop through the lgs to test on
        # test
        #x_test = xs[i].groupby(np.array(ys[i])).transform(fill_na_zero)
        x_test = xs[i].apply(fill_na_zero)
        y_pred = clf.predict(x_test)
        #y_pred_all = np.append(y_pred_all, y_pred)
        #y_test_all = np.append(y_test_all, ys[i])
        #report = classification_report(y_test_all, y_pred_all)
        report = classification_report(ys[i], y_pred)
        fscore = f1_score(ys[i], y_pred, average='weighted')
        prf = clean_report(report, lgs[i])
        acc = round((accuracy_score(ys[i], y_pred)*100),3)
        metrics = [str(round(acc,3))] + prf
        metrics = metrics + [str(fscore)]
        metrics_all.append(metrics)
        print(fscore)
    return metrics_all



# Takes in a classification report and the language code
# Reorders and returns the report in an easier to manipulate format
def clean_report(report, lg):
    report_data = []
    lines = report.split('\n')
    if lg == 'cmn':
        for line in lines[2:-3]:
            row = []
            row_data = re.split('       |      |     ', line)
            row = row_data[1].strip(), float(row_data[2]), float(row_data[3]), float(row_data[4])
            report_data.append(row)
        cP = str(report_data[0][1])
        cR = str(report_data[0][2])
        cF = str(report_data[0][3])
        mP = str(report_data[1][1])
        mR = str(report_data[1][2])
        mF = str(report_data[1][3])
        prf = ["--", mP, cP, "--", mR, cR, "--", mF, cF]
    elif lg == 'guj':
        for line in lines[2:-3]:
            row = []
            row_data = re.split('       |      ', line)
            row = row_data[1].strip(), float(row_data[2]), float(row_data[3]), float(row_data[4])
            report_data.append(row)
        bP = str(report_data[0][1])
        bR = str(report_data[0][2])
        bF = str(report_data[0][3])
        mP = str(report_data[1][1])
        mR = str(report_data[1][2])
        mF = str(report_data[1][3])
        prf = [bP, mP, "--", bR, mR, "--", bF, mF, "--"]
    else:
        for line in lines[2:-3]:
            row = []
            row_data = re.split('       |      |     ' ,line)
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
        prf = [bP, mP, cP, bR, mR, cR, bF, mF, cF]
    return prf


def main():
    args = parse_args()
    features, by_speaker, no_normalize, to_normalize = get_features(args.features_csv, args.lg)
    x, y, data = read_norm(args.lg, by_speaker, no_normalize, to_normalize, features)
    #x_engf, y_engf, data_engf = read_norm('eng-f', by_speaker, no_normalize, to_normalize, features)
    x_engm, y_engm, data_engm = read_norm('eng-m', by_speaker, no_normalize, to_normalize, features)
    ys = [y_engm]
    xs = [x_engm]
    # Train
    SVM_imb_aprf = SVM_imb(x, y, features, args.lg, xs, ys)
    aprf = pd.DataFrame(SVM_imb_aprf)
    print(aprf)
    #aprf.columns = ['acc', 'pB', 'pM', 'pC', 'rB', 'rM', 'rC', 'fB', 'fM', 'fC', 'f']
    #aprf.to_csv("../data/swap/" + args.lg + "-SVM-gend.csv")

if __name__ == "__main__":
    main()