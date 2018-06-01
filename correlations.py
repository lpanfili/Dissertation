# For a given language:
# Calculates correlations (saves CSV of all, unsorted)
# Runs an SVM and a RF on both imbalanced and resampled data
# Saves accuracy, precision, recall, and F-score for each
# Calculates feature weights for the two SVMs (saves CSVs of all, unsorted)
# Calculates feature importance for the two RFs (saves CSVs of all, unsorted)
# Outputs the top correlations, weights, and importance (saves CSV for rs only)

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
def get_features(features_csv, lg):
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
        # Pick only features marked as 1 and with < 15% udef
        for line in reader:
            if line[lgIndex] == '1':
                if float(line[lgIndex + 1]) < 0.15:
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


# Reads in a CSV of the data
# Normalizes data
# Returns x (features), y (labels), and the whole normalized data set
def read_norm(lg, by_speaker, no_normalize, to_normalize, features):
    csv = "../data/lgs/" + lg + "/" + lg + "-all.csv"
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
    """
    # Scales from 0 to 1
    
    x -= x.min()  # equivalent to df = df - df.min()
    x /= x.max()
    print(x.max())
    print(y)
    """
    return x, y, normalized


# Replaces undefineds and zeros with the mean
# If the mean is undefined, replaces with 0
def fill_na_zero(x):
    if not np.isnan(x.mean()): # If it's a number
        return x.fillna(x.mean())
    else: # If it's NaN
        return x.fillna(0)


# Runs SVM on imbalanced data set
# Returns a list with the accuracy, precision, recall, and F-score
# Returns a Pandas DF with weights
def SVM_imb(x, y, features, lg):
    clf = svm.SVC(kernel = 'linear')
    skf = StratifiedKFold(n_splits=5)
    sm = SMOTE(random_state=42)
    x = x.as_matrix()
    y = np.array(y)
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
        # Fit classifier
        clf.fit(x_train, y_train)
        y_pred = clf.predict(x_test)
        y_pred_all = np.append(y_pred_all, y_pred)
        y_test_all = np.append(y_test_all, y_test)
    fscore = f1_score(y_test_all, y_pred_all, average='weighted')
    print()
    print('f1 score SVM IMB')
    print(fscore)
    report = classification_report(y_test_all, y_pred_all)
    prf = clean_report(report, lg)
    acc = round((accuracy_score(y_test_all, y_pred_all)*100),3)
    metrics = [str(round(acc,3))] + prf
    weights = get_weights(clf.coef_, features, lg)
    return metrics, weights


# Runs SVM on resampled data set
# Returns a list with the accuracy, precision, recall, and F-score
# Returns a Pandas DF with weights
def SVM_rs(x, y, features, lg):
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
        x_train = x_train.groupby(y_train).transform(fill_na_zero)
        x_test = x_test.groupby(y_test).transform(fill_na_zero)
        # Resample (twice because there are three classes)
        x_res, y_res = sm.fit_sample(x_train, y_train)
        x_res, y_res = sm.fit_sample(x_res, y_res)
        # Fit classifier
        clf.fit(x_res, y_res)
        y_pred = clf.predict(x_test)
        y_pred_all = np.append(y_pred_all, y_pred)
        y_test_all = np.append(y_test_all, y_test)
    fscore = f1_score(y_test_all, y_pred_all, average='weighted')
    print()
    print('f1 score SVM RS')
    print(fscore)
    report = classification_report(y_test_all, y_pred_all)
    prf = clean_report(report, lg)
    acc = round((accuracy_score(y_test_all, y_pred_all)*100),3)
    metrics = [str(round(acc,3))] + prf
    weights = get_weights(clf.coef_, features, lg)
    return metrics, weights



# Given normalized data
# Calculates the correlations for each features with each phonation contrast
# Returns a Pandas DF with correlations
def get_correlations(data, y): 
    # Put phonation type back into dataframe
    y = pd.Series(y)
    data['phonation'] = y.values
    # Convert phonation type into three binary features
    data['modal'] = data.apply(lambda row: int(row['phonation'] == 'M'), axis = 1)
    data['creaky'] = data.apply(lambda row: int(row['phonation'] == 'C'), axis = 1)
    data['breathy'] = data.apply(lambda row: int(row['phonation'] == 'B'), axis = 1)
    # Make dfs that exclude one phonation type
    BC = data.drop(data[data.phonation == 'M'].index)
    BM = data.drop(data[data.phonation == 'C'].index)
    CM = data.drop(data[data.phonation == 'B'].index)
    # Make one vs one correlation matrices
    bcCorr = BC.corr()['breathy']
    bmCorr = BM.corr()['breathy']
    cmCorr = CM.corr()['creaky']
    corrMat = pd.concat([bcCorr, bmCorr, cmCorr], axis = 1)
    corrMat.columns.values[0] = 'BC-corr'
    corrMat.columns.values[1] = 'BM-corr'
    corrMat.columns.values[2] = 'CM-corr'
    corrMat = corrMat.round(decimals = 3)
    corrMat = corrMat.drop(['modal', 'breathy', 'creaky'])
    print(corrMat)
    return corrMat


# Takes a dataframe of correlations and the number of top correlations you want
# Sorts the correlations by magnitude for each contrast
# Returns the top x
def sort_corr(corr, x):
    corr['BC-abs'] = corr['BC-corr'].abs()
    corr['BM-abs'] = corr['BM-corr'].abs()
    corr['CM-abs'] = corr['CM-corr'].abs()
    # Sort each column by magnitude and pull out each comparison out as its own dataframe
    corr = corr.sort_values(by = 'BC-abs', ascending = False)
    BC  = corr[['BC-corr']].copy().reset_index()
    corr = corr.sort_values(by = 'BM-abs', ascending = False)
    BM = corr[['BM-corr']].copy().reset_index()
    corr = corr.sort_values(by = 'CM-abs', ascending = False)
    CM = corr[['CM-corr']].copy().reset_index()
    # Put the three into one dataframe
    sortedCorr = pd.concat([BC, BM, CM], axis = 1)
    top = sortedCorr.head(n = x)
    return top



def main():
    args = parse_args()
    path = "../data/lgs/" + args.lg + "/" + args.lg
    features, by_speaker, no_normalize, to_normalize = get_features(args.features_csv, args.lg)
    x, y, data = read_norm(args.lg, by_speaker, no_normalize, to_normalize, features)
    # Get and save correlations
    correlations = get_correlations(data, y)
    #print(correlations)
    #correlations.to_csv(path + "-corr-all.csv")
    

if __name__ == "__main__":
    main()