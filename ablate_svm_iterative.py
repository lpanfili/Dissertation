# Iteratively ablates categories of features

import pandas as pd
import numpy as np
import sklearn
from sklearn import svm, metrics
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import StratifiedKFold
import argparse
from imblearn.over_sampling import SMOTE
import csv
import re
from random import random

from ablation_err import ablate_category, report_ablation_results

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
    #udef_binary = data[udef].apply(pd.isnull).astype(int) # make binary features from udef
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
    #normalized = pd.concat([normalized, normalized_by_speaker, notnormalized, udef_binary], axis=1)
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



# Runs SVM on resampled data set
# Returns a list with the accuracy, precision, recall, and F-score
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
        #x_test = x_test.groupby(y_test).transform(fill_na_zero)
        x_test = x_test.apply(fill_na_zero)
        # Resample (twice because there are three classes)
        x_res, y_res = sm.fit_sample(x_train, y_train)
        x_res, y_res = sm.fit_sample(x_res, y_res)
        # Fit classifier
        clf.fit(x_res, y_res)
        y_pred = clf.predict(x_test)
        y_pred_all = np.append(y_pred_all, y_pred)
        y_test_all = np.append(y_test_all, y_test)
    fscore = f1_score(y_test_all, y_pred_all, average='weighted')
    report = classification_report(y_test_all, y_pred_all)
    prf = clean_report(report, lg)
    acc = round((accuracy_score(y_test_all, y_pred_all)*100),3)
    metrics = [str(round(acc,3))] + [str(round(fscore,5))] + prf
    print('SVM', fscore)
    return metrics


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

def get_features_by_category(path):
    feature_info = pd.read_csv(path)
    return feature_info.groupby('category')['feature'].apply(list).to_dict()

def ablate_categories(x, y, features, lg, features_by_category):
    train_and_get_f1 = lambda x, y: SVM_rs(x, y, features, lg)[1]
    # Get a list of list of features by category
    category_lists = features_by_category.values()
    # Filter the features in each category based on whether they're
    # actually present in the data
    filtered_lists = []
    for feature_list in category_lists:
        filtered_lists.append([feature for feature in feature_list if feature in x])
    # Convert the lists to tuples and filter out any empty categories
    category_lists = [tuple(f for f in category) for category in filtered_lists if len(category)]
    # Perform the ablation
    results = ablate_category(x, y, category_lists, train_and_get_f1, True)
    # Write the results to a file
    output_file = "ablation_results_{}_all.csv".format(lg)
    report_ablation_results(results).to_csv(output_file, index=False)


def main():
    args = parse_args()
    path = "../data/lgs/" + args.lg + "/" + args.lg
    features, by_speaker, no_normalize, to_normalize = get_features(args.features_csv, args.lg)
    x, y, data = read_norm(args.lg, by_speaker, no_normalize, to_normalize, features)

    features_by_category = get_features_by_category(args.features_csv)
    ablate_categories(x, y, features, args.lg, features_by_category)
    exit()

    #SVM_rs_aprf = SVM_rs(x, y, features, args.lg)
    #print(SVM_rs_aprf)

if __name__ == "__main__":
    main()