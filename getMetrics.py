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
from sklearn.metrics import accuracy_score, classification_report
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
                if float(line[lgIndex + 1]) < 15.0:
                    features.append(line[3])
                    if line[1] == '1':
                        no_normalize.append(line[3])
                    if line[2] == '1':
                        by_speaker.append(line[3])
                    if line[1] == "0" and line[2] == "0":
                        to_normalize.append(line[3])
    return features, by_speaker, no_normalize, to_normalize


# Reads in a CSV of the data
# Normalizes data
# Returns x (features), y (labels), and the whole normalized data set
def read_norm(lg, by_speaker, no_normalize, to_normalize, features, feature_dict):
    csv = "../data/lgs/" + lg + "/" + lg + ".csv"
    # Read CSV, replace undefined and 0 with NA
    data = pd.read_csv(csv, na_values=["--undefined--",0])
    # Replace feature names with latex-friendly names
    feature_names = list(data)
    for feat in feature_names:
        if feat in feature_dict:
            data = data.rename(columns = {feat: feature_dict[feat]})
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
    # REPLACE FEATURE NAMES IN NORMALIZED WITH LATEX FRIE
    x = normalized[features]
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
        # Fit classifier
        clf.fit(x_train, y_train)
        y_pred = clf.predict(x_test)
        y_pred_all = np.append(y_pred_all, y_pred)
        y_test_all = np.append(y_test_all, y_test)
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
    report = classification_report(y_test_all, y_pred_all)
    prf = clean_report(report, lg)
    acc = round((accuracy_score(y_test_all, y_pred_all)*100),3)
    metrics = [str(round(acc,3))] + prf
    weights = get_weights(clf.coef_, features, lg)
    return metrics, weights


# Runs RF on imbalanced data set
# Returns a line with the accuracy, precision, recall, and F-score
# Returns a Pandas DF with importance
def RF_imb(x, y, features, lg):
    clf = RandomForestClassifier()
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
        # Fit classifier
        clf.fit(x_train, y_train)
        y_pred = clf.predict(x_test)
        y_pred_all = np.append(y_pred_all, y_pred)
        y_test_all = np.append(y_test_all, y_test)
    report = classification_report(y_test_all, y_pred_all)
    prf = clean_report(report, lg)
    acc = round((accuracy_score(y_test_all, y_pred_all)*100),3)
    metrics = [str(round(acc,3))] + prf
    # Feature importance
    importance = get_importance(clf.feature_importances_, features, lg)
    return metrics, importance


# Runs RF on resampled data set
# Returns a line with the accuracy, precision, recall, and F-score
# Returns a Pandas DF with importance
def RF_rs(x, y, features, lg):
    clf = RandomForestClassifier()
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
    report = classification_report(y_test_all, y_pred_all)
    prf = clean_report(report, lg)
    acc = round((accuracy_score(y_test_all, y_pred_all)*100),3)
    metrics = [str(round(acc,3))] + prf
    # Feature importance
    importance = get_importance(clf.feature_importances_, features, lg)
    return metrics, importance


# Takes in the coef_ from the SVM
# Returns a Pandas DF with the feature weights
def get_weights(coef, features, lg):
    if lg == 'cmn':
        CvM = coef[0]
        CvMWeights = list(zip(features, CvM))
        weights = pd.DataFrame(CvMWeights, columns = ['feat', 'CM-weight'])
        weights = weights.set_index('feat')
    elif lg == 'guj':
        BvM = coef[0]
        BvMWeights = list(zip(features, BvM))
        weights = pd.DataFrame(BvMWeights, columns = ['feat', 'BM-weight'])
        weights = weights.set_index('feat')
    else:
        BvC = coef[0]
        BvM = coef[1]
        CvM = coef[2]
        BvCWeights = list(zip(features, BvC))
        weights = pd.DataFrame(BvCWeights, columns = ['feat', 'BC-weight'])
        weights = weights.set_index('feat')
        weights['BM-weight'] = BvM
        weights['CM-weight'] = CvM
    return weights


# Takes in the importances from the RF
# Returns a Pandas DF with the feature importances
def get_importance(coef, features, lg):
    importance = list(zip(features,coef))
    importance = pd.DataFrame(importance, columns = ['feat', 'importance'])
    importance = importance.set_index('feat')
    # MAYBE NEED TO ROUND OR *100
    return importance


# Takes in a classification report and the language code
# Reorders and returns the report in an easier to manipulate format
def clean_report(report, lg):
    report_data = []
    lines = report.split('\n')
    if lg == 'cmn':
        for line in lines[2:-3]:
            row = []
            row_data = re.split('       |      ', line)
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
            row_data = re.split('       |      ' ,line)
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


# Takes a dataframe of features weights, the number of top feats you want, and lg
# Sorts the features by magnitude for each contrast
# Returns the top x
def sort_weights(weights, x, lg):
    if lg == 'guj':
        weights['BMabs'] = weights['BM-weight'].abs()
        weights = weights.sort_values(by = 'BMabs', ascending = False)
        weights = weights[['BM-weight']].copy().reset_index()
    elif lg == 'cmn':
        weights['CMabs'] = weights['CM-weight'].abs()
        weights = weights.sort_values(by = 'CMabs', ascending = False)
        weights = weights[['CM-weight']].copy().reset_index()
    else:
        weights['BCabs'] = weights['BC-weight'].abs()
        weights['BMabs'] = weights['BM-weight'].abs()
        weights['CMabs'] = weights['CM-weight'].abs()
        weights = weights.sort_values(by = 'BCabs', ascending = False)
        BC = weights[['BC-weight']].copy().reset_index()
        weights = weights.sort_values(by = 'BMabs', ascending = False)
        BM = weights[['BM-weight']].copy().reset_index()
        weights = weights.sort_values(by = 'CMabs', ascending = False)
        CM = weights[['CM-weight']].copy().reset_index()
        weights = pd.concat([BC, BM, CM], axis=1)
    weights = weights.round(decimals = 3)
    top = weights.head(n = x)
    return top


# Takes a dataframe of feature importance and the number of top features you want
# Sorts by importance magnitude
# Returns the top x
def sort_importance(importance, x):
    importance = importance.sort_values(by = 'importance', ascending = False)
    importance = importance.reset_index()
    top = importance.head(n = x)
    return top


# Makes a dictionary mapping the original feature names
# To the LaTeX-friendly feature names
def make_feature_dict(features_csv):
    feature_dict = {}
    with open(features_csv) as f:
        reader = csv.reader(f)
        header = next(reader)
        for line in reader:
            feature = line[0]
            latex_feature = line[3]
            if feature not in feature_dict:
                feature_dict[feature] = ""
            feature_dict[feature] = latex_feature
    return feature_dict


def ablation_list(feature_importance, lg):
    cmn_index = [4,6,8]
    guj_index = [2,6,8]
    others_index = [0,2,4,6,8,10,12]
    to_ablate = []
    if lg == 'cmn':
        index_list = cmn_index
    elif lg == 'guj':
        index_list = guj_index
    else:
        index_list = others_index
    for i in index_list:
        for j in feature_importance[[i]].values.tolist():
            to_ablate.append(j[0])
    return to_ablate


def main():
    args = parse_args()
    path = "../data/lgs/" + args.lg + "/" + args.lg
    feature_dict = make_feature_dict(args.features_csv)
    features, by_speaker, no_normalize, to_normalize = get_features(args.features_csv, args.lg)
    x, y, data = read_norm(args.lg, by_speaker, no_normalize, to_normalize, features, feature_dict)
    # Get and save correlations
    correlations = get_correlations(data, y)
    correlations.to_csv(path + "-corr-all.csv")
    # Run four classifiers (SVM and RF on imb and rs data)
    SVM_imb_aprf, SVM_imb_weights = SVM_imb(x, y, features, args.lg)
    SVM_rs_aprf, SVM_rs_weights = SVM_rs(x, y, features, args.lg)
    RF_imb_aprf, RF_imb_importance = RF_imb(x, y, features, args.lg)
    RF_rs_aprf, RF_rs_importance = RF_imb(x, y, features, args.lg)
    # Combine aprf for each classifier and save
    aprf = pd.DataFrame([SVM_imb_aprf, SVM_rs_aprf, RF_imb_aprf, RF_rs_aprf])
    aprf.columns = ['acc', 'pB', 'pM', 'pC', 'rB', 'rM', 'rC', 'fB', 'fM', 'fC']
    aprf = aprf.rename({0: 'SVM_imb', 1: 'SVM_rs', 2: 'RF_imb', 3: 'RF_rs'})
    aprf.to_csv(path + "-aprf.csv")
    # Save weights and importance
    SVM_imb_weights.to_csv(path + "-weights-imb.csv")
    SVM_rs_weights.to_csv(path + "-weights-rs.csv")
    RF_imb_importance.to_csv(path + "-importance_imb.csv")
    RF_rs_importance.to_csv(path + "-importance_rs.csv")
    # Sort correlations, weights, and importance (resampled data only)
    corr_sorted = sort_corr(correlations, 10)
    weights_sorted = sort_weights(SVM_rs_weights, 10, args.lg)
    importance_sorted = sort_importance(RF_rs_importance, 10)
    feature_importance = pd.concat([corr_sorted, weights_sorted, importance_sorted], axis = 1)
    feature_importance.to_csv(path + "-topvals.csv")
    # Make and print a list of the top features to ablate
    to_ablate = ablation_list(feature_importance, args.lg)
    print(to_ablate)

if __name__ == "__main__":
    main()