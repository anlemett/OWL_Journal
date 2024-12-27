import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings('ignore')

import time
import os
import numpy as np
import pandas as pd
import sys
import math

import scipy.stats as stats
import seaborn as sns
import matplotlib.pyplot as plt

from features import init_blinks_no_head, init_blinks_quantiles
from features import init, init_blinks, blinks
from features import left, right, left_right, left_right_unite

#columns_to_select = init
#columns_to_select = init_blinks
columns_to_select = init_blinks_no_head
#columns_to_select = left_right
#columns_to_select = blinks
#columns_to_select = init_blinks_quantiles

DATA_DIR = os.path.join("..", "..")
DATA_DIR = os.path.join(DATA_DIR, "Data")
ML_DIR = os.path.join(DATA_DIR, "MLInput")
#ML_DIR = os.path.join(DATA_DIR, "MLInput_Journal")
FIG_DIR = os.path.join(".", "Figures")
RANDOM_STATE = 0
CHS = False
BINARY = True
LEFT_RIGHT_AVERAGE = True

#TIME_INTERVAL_DURATION = 300
TIME_INTERVAL_DURATION = 180
#TIME_INTERVAL_DURATION = 60
#TIME_INTERVAL_DURATION = 30
#TIME_INTERVAL_DURATION = 10

def get_percentiles():
    if BINARY:
        print("BINARY")
        return [0.93]
    else:
        print("3 classes")
        return [0.52, 0.93]


def main():
    
    np.random.seed(RANDOM_STATE)
    print(f"RANDOM_STATE: {RANDOM_STATE}")
    print(f"Time interval: {TIME_INTERVAL_DURATION}")
    
    if CHS:
        filename = "ML_features_CHS.csv"
    else:
        filename = "ML_features_" + str(TIME_INTERVAL_DURATION) + ".csv"
    
    full_filename = os.path.join(ML_DIR, filename)
    
    data_df = pd.read_csv(full_filename, sep=' ')
    
    data_df = data_df[['ATCO'] + columns_to_select]
    
    if CHS:
        print("CHS")
        full_filename = os.path.join(ML_DIR, "ML_ET_CH__CH.csv")
        scores_np = np.loadtxt(full_filename, delimiter=" ")
    else:
        print("EEG")
        full_filename = os.path.join(ML_DIR, "ML_ET_EEG_" + str(TIME_INTERVAL_DURATION) + "__EEG.csv")

        scores_df = pd.read_csv(full_filename, sep=' ', header=None)
        scores_np = scores_df.to_numpy()
        
        #scores_np = np.loadtxt(full_filename, delimiter=" ")
    
        scores_np = scores_np[0,:] # Workload
    
    scores = list(scores_np)
    
    data_df['score'] = scores
    
    print(f"Number of slots: {len(data_df.index)}")
    
    data_df = data_df.drop('ATCO', axis=1)
            
    scores = data_df['score'].to_list()
    data_df = data_df.drop('score', axis=1)
    
    if LEFT_RIGHT_AVERAGE:
        for i in range(0,17):
            
            col1 =  left[i]
            col2 = right[i]
            
            data_df[left_right_unite[i]] = (data_df[col1] + data_df[col2])/2
            data_df = data_df.drop([col1, col2], axis=1)
    
    features = data_df.columns
    print(f"Number of features: {len(features)}")
    
    scores = pd.Series(scores)
    
    total_nans = data_df.isna().sum().sum()
    print(f"Total NaNs in the DataFrame: {total_nans}")

    
    ###########################################################################
    # Feature correlation analysis
    
    # Calculate correlation matrix
    correlation_matrix = data_df.corr()
    #print(correlation_matrix)

    # Identify highly correlated pairs (threshold > 0.9)
    high_corr_pairs = correlation_matrix[(correlation_matrix > 0.99) & (correlation_matrix != 1.0)]
    #pd.option_context('display.max_rows', None, 'display.max_columns', None)
    #high_corr_pairs.to_csv("feature_correlation.csv", header=True, index=False)
    upper_triangle = correlation_matrix.where(np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool))

    # Set a threshold for high correlation
    threshold = 0.95

    # Find highly correlated pairs
    high_corr_pairs = [(col, row) for col, row in zip(*np.where(upper_triangle.abs() > threshold))]

    # Convert indices to feature names
    high_corr_feature_pairs = [(data_df.columns[i], data_df.columns[j]) for i, j in high_corr_pairs]

    # Print the list of pairs
    print(high_corr_feature_pairs)
    
    features_to_remove = set()
    for feature_1, feature_2 in high_corr_feature_pairs:
        # remove the second feature
        features_to_remove.add(feature_2)

    # Drop the features from the DataFrame
    data_df_cleaned = data_df.drop(columns=features_to_remove)
    
    #print(f"Number of columns: {len(data_df_cleaned.columns)}")
    #print(data_df_cleaned.columns)

    ###########################################################################
    # # Correlation for each feature with the 'scores'
    '''
    # Create an empty list to store correlation results
    correlations = []

    # Calculate Pearson correlation for each feature with the 'scores'
    for column in data_df.columns:
        corr, p_value = stats.pearsonr(data_df[column], scores)  # Pearson correlation
        correlations.append((column, corr, p_value))

    # Convert to a DataFrame for better visualization
    correlation_df = pd.DataFrame(correlations, columns=['Feature', 'Correlation', 'P-Value'])

    pd.set_option('display.max_rows', None)  # Show all rows
    pd.set_option('display.max_columns', None)  # Show all columns
    # Print the correlation results
    print(correlation_df)

    ###
    ### Correlation analysis with binary scores
    ###
    if CHS:
        binary_scores = [1 if score < 4 else 2 for score in scores]
    else:
        percentile_93 = np.percentile(scores, 93)
        binary_scores = (scores >= percentile_93).astype(int)

    binary_scores = pd.Series(binary_scores)

    # Create an empty list to store correlation results
    correlations = []

    # Calculate Point-Biserial correlation for each feature with the binary 'scores'
    for column in data_df.columns:
        corr, p_value = stats.pointbiserialr(data_df[column], binary_scores)  # Point-Biserial correlation
        correlations.append((column, corr, p_value))

    # Convert the results into a DataFrame for better visualization
    correlation_df = pd.DataFrame(correlations, columns=['Feature', 'Correlation', 'P-Value'])

    # Print the correlation results
    print(correlation_df)
    '''

start_time = time.time()

main()

elapsed_time = time.time() - start_time
print(f"Elapsed time: {elapsed_time:.3f} seconds")
    