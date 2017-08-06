# For one language
# Normalizes features (some by speaker)
# Runs a single-feature model for each

import pandas as pd
import numpy as np
import sklearn
from sklearn import svm, metrics
from sklearn.svm import SVC
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from sklearn.model_selection import cross_val_predict, StratifiedKFold
import argparse
from imblearn.under_sampling import RandomUnderSampler
from collections import Counter

# This list will contain the output for each single feature model
accuracy = [] 
# This is a list of all the features to test (their header names)
#features = ['H1A1c_mean','H1A1c_means001','H1A1c_means002','H1A1c_means003','H1A2c_mean','H1A2c_means001','H1A2c_means002','H1A2c_means003','H1A3c_mean','H1A3c_means001','H1A3c_means002','H1A3c_means003','H1H2c_mean','H1H2c_means001','H1H2c_means002','H1H2c_means003','H2H4c_mean','H2H4c_means001','H2H4c_means002','H2H4c_means003','H42Kc_mean','H42Kc_means001','H42Kc_means002','H42Kc_means003','H2KH5Kc_mean','H2KH5Kc_means001','H2KH5Kc_means002','H2KH5Kc_means003','jitter_loc_mean','jitter_loc_1','jitter_loc_2','jitter_loc_3','jitter_loc_abs_mean','jitter_loc_abs_1','jitter_loc_abs_2','jitter_loc_abs_3','jitter_rap_mean','jitter_rap_1','jitter_rap_2','jitter_rap_3','jitter_ppq5_mean','jitter_ppq5_1','jitter_ppq5_2','jitter_ppq5_3','Energy_mean','Energy_means001','Energy_means002','Energy_means003','shimmer_loc_mean','shimmer_loc_1','shimmer_loc_2','shimmer_loc_3','shimmer_local_dB_mean','shimmer_loc_db_1','shimmer_loc_db_2','shimmer_loc_db_3','shimmer_apq3_mean','shimmer_apq3_1','shimmer_apq3_2','shimmer_apq3_3','shimmer_apq5_mean','shimmer_apq5_1','shimmer_apq5_2','shimmer_apq5_3','shimmer_apq11_mean','shimmer_apq11_1','shimmer_apq11_2','shimmer_apq11_3','HNR05_mean','HNR05_means001','HNR05_means002','HNR05_means003','HNR15_mean','HNR15_means001','HNR15_means002','HNR15_means003','HNR25_mean','HNR25_means001','HNR25_means002','HNR25_means003','HNR35_mean','HNR35_means001','HNR35_means002','HNR35_means003','SHR_mean','SHR_means001','SHR_means002','SHR_means003','CPP_mean','CPP_means001','CPP_means002','CPP_means003','pF0_mean','pF0_means001','pF0_means002','pF0_means003','shrF0_mean','shrF0_means001','shrF0_means002','shrF0_means003','sF0_mean','sF0_means001','sF0_means002','sF0_means003','strF0_mean','strF0_means001','strF0_means002','strF0_means003','VoPT','sF1_mean','sF1_means001','sF1_means002','sF1_means003','vowel_dur']
#features = ['H1A1c_mean']
# ENGLISH ONLY FEATURES
features = ['H1A1c_mean','H1A1c_means001','H1A1c_means002','H1A1c_means003','H1A2c_mean','H1A2c_means001','H1A2c_means002','H1A2c_means003','H1A3c_mean','H1A3c_means001','H1A3c_means002','H1A3c_means003','H1H2c_mean','H1H2c_means001','H1H2c_means002','H1H2c_means003','H2H4c_mean','H2H4c_means001','H2H4c_means002','H2H4c_means003','H42Kc_mean','H42Kc_means001','H42Kc_means002','H42Kc_means003','H2KH5Kc_mean','H2KH5Kc_means001','H2KH5Kc_means002','H2KH5Kc_means003','jitter_loc_mean','jitter_loc_1','jitter_loc_2','jitter_loc_3','jitter_loc_abs_mean','jitter_loc_abs_1','jitter_loc_abs_2','jitter_loc_abs_3','jitter_rap_mean','jitter_rap_1','jitter_rap_2','jitter_rap_3','jitter_ppq5_mean','jitter_ppq5_1','jitter_ppq5_2','jitter_ppq5_3','Energy_mean','Energy_means001','Energy_means002','Energy_means003','shimmer_loc_mean','shimmer_loc_1','shimmer_loc_2','shimmer_loc_3','shimmer_local_dB_mean','shimmer_loc_db_1','shimmer_loc_db_2','shimmer_loc_db_3','shimmer_apq3_mean','shimmer_apq3_1','shimmer_apq3_2','shimmer_apq3_3','shimmer_apq5_mean','shimmer_apq5_1','shimmer_apq5_2','shimmer_apq5_3','shimmer_apq11_mean','shimmer_apq11_1','shimmer_apq11_2','shimmer_apq11_3','HNR05_mean','HNR05_means001','HNR05_means002','HNR05_means003','HNR15_mean','HNR15_means001','HNR15_means002','HNR15_means003','HNR25_mean','HNR25_means001','HNR25_means002','HNR25_means003','HNR35_mean','HNR35_means001','HNR35_means002','HNR35_means003','SHR_mean','SHR_means001','SHR_means002','SHR_means003','CPP_mean','CPP_means001','CPP_means002','CPP_means003','pF0_mean','pF0_means001','pF0_means002','pF0_means003','shrF0_mean','shrF0_means001','shrF0_means002','shrF0_means003','sF0_mean','sF0_means001','sF0_means002','sF0_means003','strF0_mean','strF0_means001','strF0_means002','strF0_means003','VoPT','sF1_mean','sF1_means001','sF1_means002','sF1_means003','vowel_dur','pre_is_voiced','fol_is_voiced','pre_is_obs','fol_is_obs','pre_exists','fol_exists','ms_from_utt_end','utt_per','ms_from_word_end','word_per']

# Argument should be full path to file with data
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_csv', type=str,
                        help='a CSV file (with headers) containing the input data')
    return parser.parse_args()

# Features that will be normalized by speaker
BY_SPEAKER = ['vowel_dur','Energy_mean','Energy_means001','Energy_means002','Energy_means003','HNR05_mean','HNR05_means001','HNR05_means002','HNR05_means003','HNR15_mean','HNR15_means001','HNR15_means002','HNR15_means003','HNR25_mean','HNR25_means001','HNR25_means002','HNR25_means003','HNR35_mean','HNR35_means001','HNR35_means002','HNR35_means003','strF0_mean','strF0_means001','strF0_means002','strF0_means003','sF0_mean','sF0_means001','sF0_means002','sF0_means003','pF0_mean','pF0_means001','pF0_means002','pF0_means003','shrF0_mean','shrF0_means001','shrF0_means002','shrF0_means003','sF1_mean','sF1_means001','sF1_means002','sF1_means003','pF1_mean','pF1_means001','pF1_means002','pF1_means003']
# Features that will be normalized but not by speaker
TO_NORMALIZE = ['jitter_ddp_mean','jitter_ddp_1','jitter_ddp_2','jitter_ddp_3','jitter_loc_mean','jitter_loc_1','jitter_loc_2','jitter_loc_3','jitter_loc_abs_mean','jitter_loc_abs_1','jitter_loc_abs_2','jitter_loc_abs_3','jitter_rap_mean','jitter_rap_1','jitter_rap_2','jitter_rap_3','jitter_ppq5_mean','jitter_ppq5_1','jitter_ppq5_2','jitter_ppq5_3','shimmer_loc_mean','shimmer_loc_1','shimmer_loc_2','shimmer_loc_3','shimmer_local_dB_mean','shimmer_loc_db_1','shimmer_loc_db_2','shimmer_loc_db_3','shimmer_apq3_mean','shimmer_apq3_1','shimmer_apq3_2','shimmer_apq3_3','shimmer_apq5_mean','shimmer_apq5_1','shimmer_apq5_2','shimmer_apq5_3','shimmer_apq11_mean','shimmer_apq11_1','shimmer_apq11_2','shimmer_apq11_3','H1H2c_mean','H1H2c_means001','H1H2c_means002','H1H2c_means003','H2H4c_mean','H2H4c_means001','H2H4c_means002','H2H4c_means003','H1A1c_mean','H1A1c_means001','H1A1c_means002','H1A1c_means003','H1A2c_mean','H1A2c_means001','H1A2c_means002','H1A2c_means003','H1A3c_mean','H1A3c_means001','H1A3c_means002','H1A3c_means003','H42Kc_mean','H42Kc_means001','H42Kc_means002','H42Kc_means003','H2KH5Kc_mean','H2KH5Kc_means001','H2KH5Kc_means002','H2KH5Kc_means003','CPP_mean','CPP_means001','CPP_means002','CPP_means003','SHR_mean','SHR_means001','SHR_means002','SHR_means003','VoPT','ms_from_utt_end','utt_per','ms_from_word_end','word_per']
NO_NORMALIZE = ['pre_is_voiced','fol_is_voiced','pre_is_obs','fol_is_obs','pre_exists','fol_exists']

def runClass(x, y):
    clf = svm.SVC()
    predictedy = cross_val_predict(clf, x, y, cv = 10)
    acc = accuracy_score(y, predictedy)
    accuracy.append(str(round(acc,3)))
    #p, r, f, s = prfs(y, predictedy)
    #print("Precision: ", p)   
    #print("Recall: ", r)
    #print("F-score: ", f)  

# Calculates precision, recall, fscore, and support
def prfs(trainy, predictedy):
    y_true = np.array(trainy)
    y_pred = np.array(predictedy)
    return precision_recall_fscore_support(y_true, y_pred, average = 'weighted')

def main():
    args = parse_args()
    # Replace undefined and 0 with na
    data = pd.read_csv(args.input_csv, na_values=["--undefined--",0])
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
    normalized = pd.concat([normalized, normalized_by_speaker, notnormalized], axis=1)
    # Replaced undefined and 0 with the mean for the class
    normalized = normalized.groupby(y).transform(lambda x: x.fillna(x.mean()))
    for feature in features:
        x = normalized[[feature]]
        # Next three lines resample, and resampled x and y are used in runClass
        # Must resample twice to cover all three classes
        rus = RandomUnderSampler(random_state=42)
        x_res, y_res = rus.fit_sample(x, y)
        x_res, y_res = rus.fit_sample(x_res, y_res)
        runClass(x_res,y_res)
    print(Counter(y_res))
    print(accuracy)

if __name__ == "__main__":
    main()