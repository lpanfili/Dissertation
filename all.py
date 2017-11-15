# For one language
# RESAMPLES data using SMOTE
# Normalizes features (some by speaker)
# Runs a single-feature model for each

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

# This is a list of all the features to test (their header names)
#features = ['H1A1c_mean','H1A1c_means001','H1A1c_means002','H1A1c_means003','H1A2c_mean','H1A2c_means001','H1A2c_means002','H1A2c_means003','H1A3c_mean','H1A3c_means001','H1A3c_means002','H1A3c_means003','H1H2c_mean','H1H2c_means001','H1H2c_means002','H1H2c_means003','H2H4c_mean','H2H4c_means001','H2H4c_means002','H2H4c_means003','H42Kc_mean','H42Kc_means001','H42Kc_means002','H42Kc_means003','H2KH5Kc_mean','H2KH5Kc_means001','H2KH5Kc_means002','H2KH5Kc_means003','jitter_loc_mean','jitter_loc_1','jitter_loc_2','jitter_loc_3','jitter_loc_abs_mean','jitter_loc_abs_1','jitter_loc_abs_2','jitter_loc_abs_3','jitter_rap_mean','jitter_rap_1','jitter_rap_2','jitter_rap_3','jitter_ppq5_mean','jitter_ppq5_1','jitter_ppq5_2','jitter_ppq5_3','Energy_mean','Energy_means001','Energy_means002','Energy_means003','shimmer_loc_mean','shimmer_loc_1','shimmer_loc_2','shimmer_loc_3','shimmer_local_dB_mean','shimmer_loc_db_1','shimmer_loc_db_2','shimmer_loc_db_3','shimmer_apq3_mean','shimmer_apq3_1','shimmer_apq3_2','shimmer_apq3_3','shimmer_apq5_mean','shimmer_apq5_1','shimmer_apq5_2','shimmer_apq5_3','shimmer_apq11_mean','shimmer_apq11_1','shimmer_apq11_2','shimmer_apq11_3','HNR05_mean','HNR05_means001','HNR05_means002','HNR05_means003','HNR15_mean','HNR15_means001','HNR15_means002','HNR15_means003','HNR25_mean','HNR25_means001','HNR25_means002','HNR25_means003','HNR35_mean','HNR35_means001','HNR35_means002','HNR35_means003','SHR_mean','SHR_means001','SHR_means002','SHR_means003','CPP_mean','CPP_means001','CPP_means002','CPP_means003','pF0_mean','pF0_means001','pF0_means002','pF0_means003','shrF0_mean','shrF0_means001','shrF0_means002','shrF0_means003','sF0_mean','sF0_means001','sF0_means002','sF0_means003','strF0_mean','strF0_means001','strF0_means002','strF0_means003','VoPT','sF1_mean','sF1_means001','sF1_means002','sF1_means003','vowel_dur']
#features = ['H1A1c_means001']
# ENGLISH ONLY FEATURES
features = ['H1A1c_mean','H1A1c_means001','H1A1c_means002','H1A1c_means003','H1A2c_mean','H1A2c_means001','H1A2c_means002','H1A2c_means003','H1A3c_mean','H1A3c_means001','H1A3c_means002','H1A3c_means003','H1H2c_mean','H1H2c_means001','H1H2c_means002','H1H2c_means003','H2H4c_mean','H2H4c_means001','H2H4c_means002','H2H4c_means003','H42Kc_mean','H42Kc_means001','H42Kc_means002','H42Kc_means003','H2KH5Kc_mean','H2KH5Kc_means001','H2KH5Kc_means002','H2KH5Kc_means003','jitter_loc_mean','jitter_loc_1','jitter_loc_2','jitter_loc_3','jitter_loc_abs_mean','jitter_loc_abs_1','jitter_loc_abs_2','jitter_loc_abs_3','jitter_rap_mean','jitter_rap_1','jitter_rap_2','jitter_rap_3','jitter_ppq5_mean','jitter_ppq5_1','jitter_ppq5_2','jitter_ppq5_3','Energy_mean','Energy_means001','Energy_means002','Energy_means003','shimmer_loc_mean','shimmer_loc_1','shimmer_loc_2','shimmer_loc_3','shimmer_local_dB_mean','shimmer_loc_db_1','shimmer_loc_db_2','shimmer_loc_db_3','shimmer_apq3_mean','shimmer_apq3_1','shimmer_apq3_2','shimmer_apq3_3','shimmer_apq5_mean','shimmer_apq5_1','shimmer_apq5_2','shimmer_apq5_3','shimmer_apq11_mean','shimmer_apq11_1','shimmer_apq11_2','shimmer_apq11_3','HNR05_mean','HNR05_means001','HNR05_means002','HNR05_means003','HNR15_mean','HNR15_means001','HNR15_means002','HNR15_means003','HNR25_mean','HNR25_means001','HNR25_means002','HNR25_means003','HNR35_mean','HNR35_means001','HNR35_means002','HNR35_means003','SHR_mean','SHR_means001','SHR_means002','SHR_means003','CPP_mean','CPP_means001','CPP_means002','CPP_means003','pF0_mean','pF0_means001','pF0_means002','pF0_means003','shrF0_mean','shrF0_means001','shrF0_means002','shrF0_means003','sF0_mean','sF0_means001','sF0_means002','sF0_means003','strF0_mean','strF0_means001','strF0_means002','strF0_means003','VoPT','sF1_mean','sF1_means001','sF1_means002','sF1_means003','vowel_dur','pre_is_voiced','fol_is_voiced','pre_is_obs','fol_is_obs','pre_exists','fol_exists','ms_from_utt_end','utt_per','ms_from_word_end','word_per','pre_is_voiced','fol_is_voiced','pre_is_obs','fol_is_obs','pre_exists','fol_exists','ms_from_utt_end','utt_per','ms_from_word_end','word_per']
#features = ['H1A1c_mean','H1A1c_means001']

# Argument should be full path to file with data
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_csv', type=str,
                        help='a CSV file (with headers) containing the input data')
    return parser.parse_args()

# Features that will be normalized by speaker
BY_SPEAKER = ['vowel_dur','Energy_mean','Energy_means001','Energy_means002','Energy_means003','HNR05_mean','HNR05_means001','HNR05_means002','HNR05_means003','HNR15_mean','HNR15_means001','HNR15_means002','HNR15_means003','HNR25_mean','HNR25_means001','HNR25_means002','HNR25_means003','HNR35_mean','HNR35_means001','HNR35_means002','HNR35_means003','strF0_mean','strF0_means001','strF0_means002','strF0_means003','sF0_mean','sF0_means001','sF0_means002','sF0_means003','pF0_mean','pF0_means001','pF0_means002','pF0_means003','shrF0_mean','shrF0_means001','shrF0_means002','shrF0_means003','sF1_mean','sF1_means001','sF1_means002','sF1_means003','pF1_mean','pF1_means001','pF1_means002','pF1_means003']
# Features that will be normalized but not by speaker
#TO_NORMALIZE = ['jitter_ddp_mean','jitter_ddp_1','jitter_ddp_2','jitter_ddp_3','jitter_loc_mean','jitter_loc_1','jitter_loc_2','jitter_loc_3','jitter_loc_abs_mean','jitter_loc_abs_1','jitter_loc_abs_2','jitter_loc_abs_3','jitter_rap_mean','jitter_rap_1','jitter_rap_2','jitter_rap_3','jitter_ppq5_mean','jitter_ppq5_1','jitter_ppq5_2','jitter_ppq5_3','shimmer_loc_mean','shimmer_loc_1','shimmer_loc_2','shimmer_loc_3','shimmer_local_dB_mean','shimmer_loc_db_1','shimmer_loc_db_2','shimmer_loc_db_3','shimmer_apq3_mean','shimmer_apq3_1','shimmer_apq3_2','shimmer_apq3_3','shimmer_apq5_mean','shimmer_apq5_1','shimmer_apq5_2','shimmer_apq5_3','shimmer_apq11_mean','shimmer_apq11_1','shimmer_apq11_2','shimmer_apq11_3','H1H2c_mean','H1H2c_means001','H1H2c_means002','H1H2c_means003','H2H4c_mean','H2H4c_means001','H2H4c_means002','H2H4c_means003','H1A1c_mean','H1A1c_means001','H1A1c_means002','H1A1c_means003','H1A2c_mean','H1A2c_means001','H1A2c_means002','H1A2c_means003','H1A3c_mean','H1A3c_means001','H1A3c_means002','H1A3c_means003','H42Kc_mean','H42Kc_means001','H42Kc_means002','H42Kc_means003','H2KH5Kc_mean','H2KH5Kc_means001','H2KH5Kc_means002','H2KH5Kc_means003','CPP_mean','CPP_means001','CPP_means002','CPP_means003','SHR_mean','SHR_means001','SHR_means002','SHR_means003','VoPT']
#ENG ONLY
TO_NORMALIZE = ['jitter_ddp_mean','jitter_ddp_1','jitter_ddp_2','jitter_ddp_3','jitter_loc_mean','jitter_loc_1','jitter_loc_2','jitter_loc_3','jitter_loc_abs_mean','jitter_loc_abs_1','jitter_loc_abs_2','jitter_loc_abs_3','jitter_rap_mean','jitter_rap_1','jitter_rap_2','jitter_rap_3','jitter_ppq5_mean','jitter_ppq5_1','jitter_ppq5_2','jitter_ppq5_3','shimmer_loc_mean','shimmer_loc_1','shimmer_loc_2','shimmer_loc_3','shimmer_local_dB_mean','shimmer_loc_db_1','shimmer_loc_db_2','shimmer_loc_db_3','shimmer_apq3_mean','shimmer_apq3_1','shimmer_apq3_2','shimmer_apq3_3','shimmer_apq5_mean','shimmer_apq5_1','shimmer_apq5_2','shimmer_apq5_3','shimmer_apq11_mean','shimmer_apq11_1','shimmer_apq11_2','shimmer_apq11_3','H1H2c_mean','H1H2c_means001','H1H2c_means002','H1H2c_means003','H2H4c_mean','H2H4c_means001','H2H4c_means002','H2H4c_means003','H1A1c_mean','H1A1c_means001','H1A1c_means002','H1A1c_means003','H1A2c_mean','H1A2c_means001','H1A2c_means002','H1A2c_means003','H1A3c_mean','H1A3c_means001','H1A3c_means002','H1A3c_means003','H42Kc_mean','H42Kc_means001','H42Kc_means002','H42Kc_means003','H2KH5Kc_mean','H2KH5Kc_means001','H2KH5Kc_means002','H2KH5Kc_means003','CPP_mean','CPP_means001','CPP_means002','CPP_means003','SHR_mean','SHR_means001','SHR_means002','SHR_means003','VoPT','ms_from_utt_end','utt_per','ms_from_word_end','word_per']
NO_NORMALIZE = ['pre_is_voiced','fol_is_voiced','pre_is_obs','fol_is_obs','pre_exists','fol_exists']

# Replaces undefineds and zeros with the mean
# If the mean is undefined, replaces with 0
def fillNaOrZero(x):
    if not np.isnan(x.mean()): # If it's a number
        return x.fillna(x.mean())
    else: # If it's NaN
        return x.fillna(0)

def runClass(x, y, features):
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
        """
        # Resample (twice because there are three classes)
        x_res, y_res = sm.fit_sample(x_train, y_train)
        x_res, y_res = sm.fit_sample(x_res, y_res)
        """
        # Fit classifier
        clf.fit(x_train, y_train)
        y_pred = clf.predict(x_test)
        y_pred_all = np.append(y_pred_all, y_pred)
        y_test_all = np.append(y_test_all, y_test)
    confmat = confusion_matrix(y_test_all, y_pred_all)
    print(classification_report(y_test_all, y_pred_all))
    acc = accuracy_score(y_test_all, y_pred_all)
    acc = round((acc * 100),3)
    # Feature weights
    for x in clf.coef_:
        feat_weights = zip(features, x)
        print(sorted(feat_weights, key=lambda x:x[1], reverse=True)[:5])
    return acc
"""
# IF RESAMPLED
        clf.fit(x_res, y_res)
        y_pred = clf.predict(x_test)
        y_pred_all = np.append(y_pred_all, y_pred)
        y_test_all = np.append(y_test_all, y_test)
    acc = accuracy_score(y_test_all, y_pred_all)
    acc = round((acc * 100),3)
    print(clf.coef_)
    return acc
"""

def main():
    args = parse_args()
    # Replace undefined and 0 with na
    data = pd.read_csv(args.input_csv, na_values=["--undefined--",0])
    data['modal'] = data.apply(lambda row: int(row['phonation'] == 'M'), axis = 1)
    data.corr()['modal'].to_csv('/Users/Laura/Desktop/corr.csv')
    return
    # Combine primary and secondary stress (ENGLISH ONLY)
    # data["vowel_label_no_stress"] = data["vowel_label"].apply(lambda val: val[:2])    
    # Define z-score
    zscore = lambda x: (x - x.mean())/x.std()
    # Normalize some by speaker
    normalized_by_speaker = data[BY_SPEAKER+["speaker"]].groupby("speaker").transform(zscore)
    # Normalize the rest over all the data
    normalized = zscore(data[TO_NORMALIZE])
    notnormalized = data[NO_NORMALIZE]
    # List of phonation
    y = data['phonation'].tolist()
    # Clumps together normalized by speaker and normalized overall
    #normalized = pd.concat([normalized, normalized_by_speaker], axis=1)
    normalized = pd.concat([normalized, normalized_by_speaker, notnormalized], axis=1)
    udefcount = []
    acc_all = []
    # Calculating the percent undefined
    udefDict = {}
    for feature in features:
        nonNullCount = normalized[feature].count()
        fullCount = normalized[feature].fillna(0).count()
        percent = (fullCount-nonNullCount)/fullCount
        udefDict[feature] = percent
        #percent = count * 100
        #udefcount.append(str(round(percent,3)))
    features15 = [x for x in features if udefDict[x] < 0.15]
    x = normalized[features15]
    acc = runClass(x,y,features15)
    acc_all.append(acc)
    # Print everything
    print("ACCURACY")
    for i in acc_all:
       print(i)

if __name__ == "__main__":
    main()