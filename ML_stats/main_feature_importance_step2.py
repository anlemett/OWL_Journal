import warnings
warnings.filterwarnings('ignore')

import time
import os
import numpy as np
import pandas as pd
#import sys

import gc

from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold

from get_model import get_model
from get_random_search_params import get_random_search_params
from features import init_blinks, init_blinks_no_head
from features import left, right, left_right_average

columns_to_select = init_blinks
#columns_to_select = init_blinks_no_head

CHS = True
BINARY = False

MODEL = "SVC"
#MODEL = "KNN"

#15 features
if MODEL == "SVC":
    if BINARY: #SVC, binary
        #threshold=0.7
        columns_order = ['Blink Closing Speed Max', 'Pupil Diameter Max', 'Blink Opening Speed Max', 'Fixation Duration Median', 'Pupil Diameter Min', 'Blink Closing Amplitude Mean', 'Blinks Duration Median', 'Pupil Diameter Std', 'Blinks Duration Min', 'Saccades Duration Max', 'Blinks Duration Max', 'Fixation Duration Max', 'Saccades Duration Mean', 'Pupil Diameter Mean', 'Blink Closing Amplitude Max']
    else: #SVC, 3 classes
        #threshold=0.7
        columns_order = ['Blink Closing Speed Max', 'Pupil Diameter Max', 'Blink Opening Speed Max', 'Pupil Diameter Min', 'Fixation Duration Max', 'Fixation Duration Median', 'Pupil Diameter Std', 'Saccades Duration Max', 'Blinks Duration Min', 'Saccades Duration Mean', 'Blinks Duration Max', 'Blink Closing Amplitude Max', 'Blinks Duration Median', 'Blink Closing Amplitude Mean', 'Pupil Diameter Mean']
elif MODEL == "KNN":
    if BINARY: #KNN, binary
        # threshold=0.7
        columns_order = ['Saccades Duration Max', 'Blink Closing Speed Max', 'Blink Closing Amplitude Mean', 'Pupil Diameter Max', 'Blink Opening Speed Max', 'Blinks Duration Median', 'Pupil Diameter Min', 'Fixation Duration Median', 'Blinks Duration Min', 'Fixation Duration Max', 'Blinks Duration Max', 'Pupil Diameter Std', 'Saccades Duration Mean', 'Pupil Diameter Mean', 'Blink Closing Amplitude Max']
    else: #KNN, 3 classes
        # thresold=0.7
        columns_order = ['Pupil Diameter Max', 'Fixation Duration Median', 'Saccades Duration Max', 'Blink Closing Speed Max', 'Blink Opening Speed Max', 'Pupil Diameter Min', 'Blink Closing Amplitude Mean', 'Blinks Duration Median', 'Pupil Diameter Std', 'Fixation Duration Max', 'Blinks Duration Min', 'Saccades Duration Mean', 'Blinks Duration Max', 'Blink Closing Amplitude Max', 'Pupil Diameter Mean']


DATA_DIR = os.path.join("..", "..")
DATA_DIR = os.path.join(DATA_DIR, "Data")
ML_DIR = os.path.join(DATA_DIR, "MLInput")
FIG_DIR = os.path.join(".", "Figures")

RANDOM_STATE = 0
LEFT_RIGHT_AVERAGE = False

N_ITER = 100
N_SPLIT = 10
SCORING = 'f1_macro'

#TIME_INTERVAL_DURATION = 300
TIME_INTERVAL_DURATION = 180
#TIME_INTERVAL_DURATION = 60
#TIME_INTERVAL_DURATION = 30
#TIME_INTERVAL_DURATION = 10
#TIME_INTERVAL_DURATION = 1

class ThresholdLabelTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, percentiles=None):
        """
        Initialize with optional percentiles.
        If percentiles is None, perform no transformation.
        """
        self.percentiles = percentiles if percentiles else []
        self.thresholds = []

    def fit(self, X, y=None):
        # Calculate thresholds based on specified percentiles, if provided
        if self.percentiles:
            self.thresholds = [np.percentile(y, p * 100) for p in self.percentiles]
        return self
    
    def transform(self, X, y=None):
        if y is None:
            return X
        
        if not self.thresholds:
            # If no thresholds are specified, return y unchanged
            return X, y

        # Initialize all labels to the lowest class (1)
        y_transformed = np.ones(y.shape, dtype=int)
        
        if CHS:
            if BINARY:
                y_transformed = [1 if score < 4 else 2 for score in y]
            else:
                y_transformed = [1 if score < 2 else 3 if score > 3 else 2 for score in y]
        else:
            # Apply thresholds to create labels
            for i, threshold in enumerate(self.thresholds):
                y_transformed[y >= threshold] = i + 2  # Increment class label for each threshold

        y_transformed = np.array(y_transformed)
        return X, y_transformed

# Function to perform parameter tuning with RandomizedSearchCV on each training fold
def model_with_tuning_stratified(pipeline, X_train, y_train):
    
    param_dist = get_random_search_params(MODEL)
    
    stratified_kfold = StratifiedKFold(n_splits=N_SPLIT, shuffle=True, random_state=RANDOM_STATE)
    
    # Initialize RandomizedSearchCV with pipeline, parameter distribution, and inner CV
    randomized_search = RandomizedSearchCV(
        pipeline, param_dist, n_iter=N_ITER, cv=stratified_kfold, scoring=SCORING, n_jobs=-1, random_state=RANDOM_STATE
    )
    
    print("before randomized_search.fit")
    # Fit on the training data for this fold
    randomized_search.fit(X_train, y_train)
    print("after randomized_search.fit")
    
    # Return the best estimator found in this fold
    return randomized_search.best_estimator_

    
# Cross-validation function that handles the pipeline
def cross_val_stratified_with_label_transform(pipeline, data_df, scores, cv, features_init):
    
    df = data_df
    
    X = df.to_numpy()
    y = np.array(scores)
    
    pipeline.named_steps['label_transform'].fit(X, y)  # Fit to compute thresholds
    _, y_transformed = pipeline.named_steps['label_transform'].transform(X, y)

    # Set class weights to the classifier
    if MODEL != "KNN":
        pipeline.named_steps['classifier'].set_params(class_weight='balanced')
    
    data_split = list(enumerate(cv.split(X, y_transformed), start=1))
    
    # Calculate performance metrics
    bal_accuracies_num_features = []
    f1_scores_num_features = []
    
    features = features_init
    num_of_features = len(features)
    
    for k in range(0, num_of_features):
      print(f"k = {k}")
      print(f"Number of features left {num_of_features-k}")
      accuracies = []
      bal_accuracies = []
      precisions = []
      recalls = []
      f1_scores = []

      for i, (train_index, test_index) in data_split:
        
        print(f"Iteration {i}")
        
        df = data_df[features]
        X = df.to_numpy()
        
        X_train, X_test = np.array(X)[train_index], np.array(X)[test_index]
        y_train_transformed, y_test_transformed = np.array(y_transformed)[train_index], np.array(y_transformed)[test_index]
        
        print("before model_with_tuning")
        # Get the best model after tuning on the current fold
        best_model = model_with_tuning_stratified(pipeline, X_train, y_train_transformed)
        print("after model_with_tuning")
        
        # Fit the pipeline on transformed y_train
        best_model.fit(X_train, y_train_transformed)
        
        ############################## Predict ################################
    
        y_pred = best_model.predict(X_test)
    
        ############################ Evaluate #################################
                
        # Calculate the metrics
        accuracies.append(accuracy_score(y_test_transformed, y_pred))
        bal_accuracies.append(balanced_accuracy_score(y_test_transformed, y_pred))
        precisions.append(precision_score(y_test_transformed, y_pred, average='macro', zero_division=0))
        recalls.append(recall_score(y_test_transformed, y_pred, average='macro', zero_division=0))
        f1_scores.append(f1_score(y_test_transformed, y_pred, average='macro', zero_division=0))

      print(f"Balanced accuracy: {np.mean(bal_accuracies):.2f} ± {np.std(bal_accuracies):.2f}")
      bal_accuracies_num_features.append(np.mean(bal_accuracies))
      print(f"F1-Score: {np.mean(f1_scores):.2f} ± {np.std(f1_scores):.2f}")
      f1_scores_num_features.append(np.mean(f1_scores))
      
      features = features[1:]
      
      del X
      gc.collect()

    print(len(f1_scores_num_features))
    print(bal_accuracies_num_features)
    print(f1_scores_num_features)
  
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
    
    ###########################################################################
    
    print(f"Number of slots: {len(data_df.index)}")
    
    data_df = data_df.drop('ATCO', axis=1)
    
    scores = data_df['score'].to_list()
    data_df = data_df.drop('score', axis=1)
    
    print(len(data_df.columns))
    if LEFT_RIGHT_AVERAGE:
        for i in range(0,17):
            
            col1 =  left[i]
            col2 = right[i]
            
            data_df[left_right_average[i]] = (data_df[col1] + data_df[col2])/2
            data_df = data_df.drop([col1, col2], axis=1)

    print(len(data_df.columns))

    data_df = data_df[columns_order]    
    features = data_df.columns
    
    print(f"Number of features: {len(features)}")
            
    pipeline = Pipeline([
            # Step 1: Standardize features
            ('scaler', StandardScaler()),
            # Step 2: Apply custom label transformation
            ('label_transform', ThresholdLabelTransformer(get_percentiles())),
            # Step 3: Choose the model
            ('classifier', get_model(MODEL, RANDOM_STATE))
            ])
    
    
    # Initialize the cross-validation splitter
    outer_cv = StratifiedKFold(n_splits=N_SPLIT, shuffle=True, random_state=RANDOM_STATE)
    cross_val_stratified_with_label_transform(pipeline, data_df, scores, outer_cv, features)
    
    print(f"Number of slots: {len(data_df.index)}")


start_time = time.time()

main()

elapsed_time = time.time() - start_time
print(f"Elapsed time: {elapsed_time:.3f} seconds")
    