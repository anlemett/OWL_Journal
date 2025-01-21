import warnings
warnings.filterwarnings('ignore')

import time
import os
import numpy as np
import pandas as pd
import sys

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

#columns_to_select = init_blinks
columns_to_select = init_blinks_no_head

CHS = True
BINARY = False
LEFT_RIGHT_AVERAGE = False

#MODEL = "SVC"
MODEL = "KNN"

'''
#init_blinks, 64 features
if MODEL == "SVC":
    if BINARY: #SVC, binary
        columns_order = ['Fixation Duration Mean', 'Blinks Number', 'Right Blink Closing Amplitude Mean', 'Head Roll Std', 'Right Blink Closing Amplitude Std', 'Left Blink Opening Amplitude Mean', 'Left Blink Closing Speed Std', 'Left Blink Opening Speed Mean', 'Head Roll Median', 'Blinks Duration Std', 'Fixation Duration Std', 'Right Pupil Diameter Median', 'Head Roll Min', 'Right Blink Opening Amplitude Std', 'Left Blink Opening Speed Std', 'Left Blink Opening Amplitude Std', 'Left Blink Closing Speed Mean', 'Left Blink Closing Amplitude Mean', 'Right Pupil Diameter Std', 'Saccades Number', 'Right Blink Closing Speed Std', 'Head Pitch Max', 'Right Blink Opening Speed Std', 'Saccades Duration Mean', 'Right Pupil Diameter Max', 'Right Blink Opening Amplitude Mean', 'Head Heading Max', 'Head Heading Std', 'Head Heading Min', 'Right Blink Opening Amplitude Max', 'Left Pupil Diameter Max', 'Left Pupil Diameter Min', 'Left Blink Closing Speed Max', 'Right Blink Closing Speed Max', 'Left Blink Closing Amplitude Max', 'Head Pitch Std', 'Saccades Duration Std', 'Head Pitch Min', 'Left Blink Closing Amplitude Std', 'Right Blink Opening Speed Mean', 'Right Blink Opening Speed Max', 'Right Blink Closing Speed Mean', 'Head Heading Median', 'Head Roll Max', 'Blinks Duration Median', 'Left Blink Opening Speed Max', 'Right Pupil Diameter Min', 'Saccades Duration Max', 'Left Pupil Diameter Std', 'Blinks Duration Min', 'Fixation Duration Median', 'Blinks Duration Max', 'Left Blink Opening Amplitude Max', 'Head Heading Mean', 'Left Pupil Diameter Mean', 'Saccades Duration Median', 'Blinks Duration Mean', 'Right Blink Closing Amplitude Max', 'Fixation Duration Max', 'Left Pupil Diameter Median', 'Head Roll Mean', 'Right Pupil Diameter Mean', 'Head Pitch Median', 'Head Pitch Mean']
    else: #SVC, 3 classes
        columns_order = ['Right Blink Opening Speed Mean', 'Blinks Number', 'Right Blink Closing Amplitude Mean', 'Left Blink Closing Speed Mean', 'Left Blink Opening Speed Mean', 'Left Blink Closing Amplitude Std', 'Right Pupil Diameter Mean', 'Right Blink Closing Amplitude Std', 'Left Blink Opening Speed Std', 'Left Blink Closing Speed Std', 'Fixation Duration Mean', 'Right Blink Opening Speed Std', 'Left Blink Opening Amplitude Mean', 'Left Blink Closing Amplitude Mean', 'Right Blink Closing Speed Std', 'Left Blink Opening Amplitude Std', 'Fixation Duration Std', 'Head Heading Max', 'Right Pupil Diameter Std', 'Saccades Duration Max', 'Blinks Duration Max', 'Head Roll Mean', 'Head Roll Max', 'Head Heading Min', 'Left Blink Closing Speed Max', 'Right Blink Opening Amplitude Mean', 'Head Heading Median', 'Right Pupil Diameter Max', 'Left Pupil Diameter Max', 'Head Roll Min', 'Right Blink Closing Speed Max', 'Left Pupil Diameter Min', 'Saccades Duration Mean', 'Right Blink Opening Speed Max', 'Blinks Duration Mean', 'Left Blink Opening Amplitude Max', 'Right Blink Closing Speed Mean', 'Head Pitch Max', 'Head Pitch Min', 'Left Blink Opening Speed Max', 'Left Blink Closing Amplitude Max', 'Right Pupil Diameter Min', 'Head Heading Std', 'Blinks Duration Min', 'Left Pupil Diameter Mean', 'Head Roll Std', 'Head Pitch Std', 'Left Pupil Diameter Std', 'Fixation Duration Median', 'Right Blink Closing Amplitude Max', 'Saccades Number', 'Blinks Duration Std', 'Saccades Duration Median', 'Saccades Duration Std', 'Right Blink Opening Amplitude Max', 'Fixation Duration Max', 'Head Heading Mean', 'Head Roll Median', 'Blinks Duration Median', 'Right Blink Opening Amplitude Std', 'Left Pupil Diameter Median', 'Right Pupil Diameter Median', 'Head Pitch Median', 'Head Pitch Mean']
elif MODEL == "KNN":
    if BINARY: #KNN, binary
        columns_order = ['Saccades Duration Max', 'Head Heading Median', 'Right Blink Closing Amplitude Max', 'Head Heading Mean', 'Head Roll Min', 'Right Pupil Diameter Median', 'Fixation Duration Mean', 'Left Blink Closing Amplitude Std', 'Head Pitch Std', 'Left Blink Closing Amplitude Mean', 'Head Roll Mean', 'Head Heading Std', 'Fixation Duration Max', 'Head Heading Min', 'Head Heading Max', 'Right Blink Closing Amplitude Std', 'Fixation Duration Median', 'Left Pupil Diameter Max', 'Right Blink Closing Amplitude Mean', 'Right Blink Opening Amplitude Max', 'Right Pupil Diameter Max', 'Left Blink Opening Amplitude Mean', 'Right Blink Opening Amplitude Mean', 'Left Blink Closing Speed Mean', 'Right Blink Opening Speed Max', 'Left Blink Opening Amplitude Std', 'Left Blink Opening Speed Std', 'Blinks Number', 'Left Blink Closing Speed Std', 'Left Blink Opening Speed Mean', 'Left Pupil Diameter Std', 'Right Blink Opening Speed Mean', 'Left Blink Closing Amplitude Max', 'Right Blink Closing Speed Mean', 'Left Pupil Diameter Mean', 'Blinks Duration Min', 'Blinks Duration Mean', 'Saccades Number', 'Blinks Duration Max', 'Saccades Duration Mean', 'Left Blink Opening Speed Max', 'Head Pitch Max', 'Head Pitch Median', 'Saccades Duration Std', 'Left Blink Closing Speed Max', 'Left Pupil Diameter Min', 'Left Blink Opening Amplitude Max', 'Head Pitch Min', 'Right Pupil Diameter Min', 'Right Blink Opening Amplitude Std', 'Head Roll Median', 'Right Blink Closing Speed Max', 'Fixation Duration Std', 'Right Blink Closing Speed Std', 'Head Roll Max', 'Head Roll Std', 'Blinks Duration Median', 'Right Pupil Diameter Std', 'Blinks Duration Std', 'Right Blink Opening Speed Std', 'Saccades Duration Median', 'Right Pupil Diameter Mean', 'Left Pupil Diameter Median', 'Head Pitch Mean']
    else: #KNN, 3 classes
        columns_order = ['Blinks Duration Max', 'Left Blink Closing Speed Max', 'Left Blink Closing Amplitude Max', 'Right Blink Opening Amplitude Max', 'Left Blink Opening Amplitude Std', 'Head Roll Std', 'Left Blink Opening Speed Std', 'Left Pupil Diameter Max', 'Head Roll Median', 'Saccades Number', 'Left Blink Opening Amplitude Max', 'Right Blink Opening Amplitude Std', 'Left Blink Closing Speed Std', 'Left Blink Closing Amplitude Std', 'Right Blink Closing Speed Max', 'Head Heading Min', 'Left Blink Opening Amplitude Mean', 'Left Blink Closing Speed Mean', 'Left Blink Opening Speed Max', 'Right Blink Opening Speed Max', 'Fixation Duration Max', 'Right Pupil Diameter Max', 'Saccades Duration Mean', 'Right Blink Closing Speed Std', 'Saccades Duration Std', 'Saccades Duration Median', 'Head Heading Median', 'Head Roll Max', 'Head Heading Max', 'Right Blink Closing Amplitude Max', 'Head Roll Mean', 'Saccades Duration Max', 'Blinks Duration Std', 'Fixation Duration Mean', 'Fixation Duration Std', 'Left Blink Opening Speed Mean', 'Left Blink Closing Amplitude Mean', 'Head Roll Min', 'Head Pitch Mean', 'Right Blink Closing Amplitude Mean', 'Right Blink Opening Amplitude Mean', 'Head Heading Std', 'Blinks Duration Min', 'Right Blink Opening Speed Mean', 'Right Blink Closing Amplitude Std', 'Blinks Number', 'Left Pupil Diameter Mean', 'Right Pupil Diameter Std', 'Right Blink Opening Speed Std', 'Right Pupil Diameter Mean', 'Blinks Duration Median', 'Left Pupil Diameter Min', 'Left Pupil Diameter Std', 'Head Pitch Min', 'Head Pitch Max', 'Right Pupil Diameter Min', 'Fixation Duration Median', 'Right Blink Closing Speed Mean', 'Head Pitch Std', 'Left Pupil Diameter Median', 'Head Heading Mean', 'Head Pitch Median', 'Blinks Duration Mean', 'Right Pupil Diameter Median']
elif MODEL == "ETC":
    if BINARY:
        columns_order = ['Right Blink Opening Amplitude Std', 'Right Blink Opening Amplitude Mean', 'Right Blink Closing Amplitude Std', 'Fixation Duration Median', 'Right Blink Opening Speed Std', 'Right Blink Closing Speed Std', 'Head Pitch Mean', 'Right Blink Closing Speed Mean', 'Fixation Duration Mean', 'Right Blink Closing Amplitude Max', 'Right Blink Opening Speed Mean', 'Right Blink Closing Amplitude Mean', 'Left Blink Closing Amplitude Max', 'Saccades Duration Median', 'Saccades Duration Max', 'Left Blink Opening Speed Std', 'Left Blink Opening Amplitude Std', 'Blinks Duration Median', 'Left Blink Closing Speed Std', 'Right Blink Opening Amplitude Max', 'Left Blink Closing Amplitude Std', 'Blinks Duration Mean', 'Head Heading Mean', 'Blinks Number', 'Blinks Duration Std', 'Saccades Number', 'Head Roll Median', 'Right Pupil Diameter Mean', 'Left Blink Closing Amplitude Mean', 'Left Blink Opening Amplitude Mean', 'Right Pupil Diameter Median', 'Left Blink Opening Speed Mean', 'Right Blink Closing Speed Max', 'Right Blink Opening Speed Max', 'Fixation Duration Std', 'Left Blink Closing Speed Max', 'Head Pitch Median', 'Left Blink Opening Speed Max', 'Left Pupil Diameter Max', 'Blinks Duration Min', 'Left Blink Closing Speed Mean', 'Head Roll Min', 'Right Pupil Diameter Min', 'Right Pupil Diameter Max', 'Head Pitch Std', 'Left Pupil Diameter Mean', 'Head Roll Std', 'Head Heading Min', 'Left Pupil Diameter Min', 'Head Heading Max', 'Head Pitch Min', 'Saccades Duration Mean', 'Right Pupil Diameter Std', 'Head Pitch Max', 'Left Blink Opening Amplitude Max', 'Head Roll Max', 'Saccades Duration Std', 'Fixation Duration Max', 'Left Pupil Diameter Std', 'Head Heading Median', 'Left Pupil Diameter Median', 'Head Roll Mean', 'Head Heading Std', 'Blinks Duration Max']
    else: #ETC, 3 classes
        columns_order = ['Right Blink Opening Speed Max', 'Right Pupil Diameter Median', 'Right Blink Opening Amplitude Std', 'Right Blink Closing Speed Max', 'Left Blink Opening Speed Mean', 'Right Pupil Diameter Mean', 'Right Blink Closing Amplitude Mean', 'Left Blink Closing Speed Max', 'Saccades Duration Median', 'Blinks Duration Min', 'Left Blink Opening Amplitude Mean', 'Right Blink Opening Amplitude Mean', 'Right Blink Opening Speed Mean', 'Right Blink Opening Speed Std', 'Right Blink Closing Speed Mean', 'Left Blink Opening Amplitude Std', 'Fixation Duration Median', 'Right Blink Closing Speed Std', 'Left Blink Opening Speed Max', 'Left Pupil Diameter Max', 'Blinks Duration Median', 'Head Heading Max', 'Blinks Number', 'Right Pupil Diameter Max', 'Left Blink Opening Speed Std', 'Head Heading Min', 'Saccades Duration Mean', 'Left Blink Closing Speed Std', 'Head Roll Max', 'Left Blink Closing Speed Mean', 'Right Pupil Diameter Min', 'Fixation Duration Std', 'Head Heading Mean', 'Left Pupil Diameter Mean', 'Left Blink Closing Amplitude Mean', 'Fixation Duration Max', 'Head Roll Median', 'Head Roll Mean', 'Head Pitch Median', 'Left Blink Closing Amplitude Max', 'Head Pitch Min', 'Left Blink Closing Amplitude Std', 'Left Pupil Diameter Min', 'Head Roll Min', 'Head Pitch Mean', 'Left Blink Opening Amplitude Max', 'Right Blink Closing Amplitude Max', 'Saccades Number', 'Right Pupil Diameter Std', 'Saccades Duration Max', 'Head Roll Std', 'Head Pitch Std', 'Head Pitch Max', 'Saccades Duration Std', 'Head Heading Median', 'Blinks Duration Max', 'Left Pupil Diameter Std', 'Head Heading Std', 'Right Blink Opening Amplitude Max', 'Fixation Duration Mean', 'Blinks Duration Mean', 'Left Pupil Diameter Median', 'Right Blink Closing Amplitude Std', 'Blinks Duration Std']
'''
#init_blinks_no_head
if MODEL == "SVC":
    if BINARY: #SVC, binary
        columns_order = ['Left Blink Opening Speed Std', 'Right Blink Closing Amplitude Std', 'Right Pupil Diameter Median', 'Blinks Number', 'Right Blink Closing Speed Std', 'Right Blink Opening Speed Std', 'Left Blink Opening Amplitude Mean', 'Right Blink Opening Amplitude Std', 'Left Blink Closing Amplitude Mean', 'Left Blink Opening Amplitude Std', 'Left Blink Closing Speed Std', 'Left Blink Opening Speed Mean', 'Saccades Number', 'Right Pupil Diameter Std', 'Saccades Duration Mean', 'Right Blink Closing Amplitude Mean', 'Right Pupil Diameter Max', 'Right Blink Opening Amplitude Mean', 'Left Blink Closing Speed Max', 'Right Blink Closing Speed Max', 'Fixation Duration Std', 'Left Blink Opening Speed Max', 'Left Pupil Diameter Max', 'Left Blink Closing Speed Mean', 'Right Blink Opening Amplitude Max', 'Right Blink Opening Speed Max', 'Left Pupil Diameter Min', 'Blinks Duration Median', 'Left Blink Closing Amplitude Max', 'Right Blink Opening Speed Mean', 'Right Pupil Diameter Min', 'Saccades Duration Max', 'Left Pupil Diameter Std', 'Fixation Duration Median', 'Right Blink Closing Speed Mean', 'Fixation Duration Max', 'Blinks Duration Std', 'Blinks Duration Min', 'Left Blink Closing Amplitude Std', 'Saccades Duration Std', 'Blinks Duration Mean', 'Blinks Duration Max', 'Left Blink Opening Amplitude Max', 'Saccades Duration Median', 'Left Pupil Diameter Median', 'Fixation Duration Mean', 'Right Blink Closing Amplitude Max', 'Left Pupil Diameter Mean', 'Right Pupil Diameter Mean']
    else: #SVC, 3 classes
        columns_order = []
elif MODEL == "KNN":
    if BINARY: #KNN, binary
        columns_order = ['Left Blink Opening Amplitude Max', 'Right Pupil Diameter Max', 'Right Blink Closing Amplitude Mean', 'Saccades Duration Mean', 'Right Pupil Diameter Min', 'Left Blink Opening Amplitude Std', 'Saccades Duration Max', 'Saccades Number', 'Saccades Duration Std', 'Left Pupil Diameter Mean', 'Saccades Duration Median', 'Fixation Duration Mean', 'Fixation Duration Std', 'Fixation Duration Median', 'Blinks Number', 'Fixation Duration Max', 'Blinks Duration Mean', 'Blinks Duration Std', 'Blinks Duration Median', 'Blinks Duration Min', 'Right Blink Opening Speed Max', 'Left Pupil Diameter Std', 'Blinks Duration Max', 'Left Blink Closing Speed Max', 'Right Blink Closing Speed Std', 'Left Blink Closing Speed Std', 'Right Blink Opening Speed Mean', 'Left Pupil Diameter Min', 'Right Blink Opening Speed Std', 'Right Blink Opening Amplitude Std', 'Left Blink Opening Amplitude Mean', 'Right Blink Closing Amplitude Max', 'Left Blink Closing Amplitude Mean', 'Left Blink Closing Speed Mean', 'Right Blink Closing Speed Mean', 'Left Blink Closing Amplitude Std', 'Right Blink Opening Amplitude Mean', 'Right Blink Closing Speed Max', 'Left Blink Opening Speed Max', 'Left Blink Opening Speed Mean', 'Left Pupil Diameter Max', 'Right Pupil Diameter Median', 'Right Blink Closing Amplitude Std', 'Right Blink Opening Amplitude Max', 'Right Pupil Diameter Std', 'Left Pupil Diameter Median', 'Right Pupil Diameter Mean', 'Left Blink Opening Speed Std', 'Left Blink Closing Amplitude Max']
    else: #KNN, 3 classes
        columns_order = ['Saccades Duration Median', 'Right Blink Closing Amplitude Mean', 'Right Pupil Diameter Max', 'Saccades Duration Mean', 'Saccades Number', 'Fixation Duration Max', 'Left Blink Closing Amplitude Max', 'Left Blink Opening Amplitude Std', 'Left Blink Closing Amplitude Std', 'Right Blink Closing Amplitude Max', 'Right Blink Closing Amplitude Std', 'Right Blink Opening Amplitude Max', 'Right Blink Opening Amplitude Std', 'Left Blink Opening Speed Max', 'Right Blink Opening Speed Std', 'Saccades Duration Std', 'Left Blink Opening Speed Mean', 'Right Blink Opening Speed Mean', 'Left Blink Closing Speed Max', 'Blinks Duration Mean', 'Saccades Duration Max', 'Left Blink Opening Amplitude Mean', 'Right Blink Closing Speed Mean', 'Blinks Duration Max', 'Left Pupil Diameter Mean', 'Left Pupil Diameter Max', 'Right Blink Opening Speed Max', 'Left Blink Closing Speed Std', 'Fixation Duration Mean', 'Right Pupil Diameter Median', 'Left Blink Opening Speed Std', 'Right Blink Opening Amplitude Mean', 'Left Pupil Diameter Min', 'Right Blink Closing Speed Std', 'Left Blink Closing Amplitude Mean', 'Blinks Duration Min', 'Blinks Number', 'Right Blink Closing Speed Max', 'Fixation Duration Std', 'Right Pupil Diameter Std', 'Right Pupil Diameter Min', 'Fixation Duration Median', 'Left Pupil Diameter Std', 'Left Blink Closing Speed Mean', 'Blinks Duration Std', 'Left Blink Opening Amplitude Max', 'Blinks Duration Median', 'Right Pupil Diameter Mean', 'Left Pupil Diameter Median']

'''
#init_blinks, correlated features dropped (threshold=0.7)
# => 33 features
if MODEL == "SVC":
    if BINARY: #SVC, binary
        columns_order = ['Head Roll Min', 'Head Heading Max', 'Left Blink Closing Speed Max', 'Head Pitch Max', 'Head Heading Min', 'Head Heading Std', 'Right Pupil Diameter Std', 'Right Pupil Diameter Max', 'Right Blink Closing Speed Max', 'Left Pupil Diameter Min', 'Right Blink Opening Speed Max', 'Left Pupil Diameter Max', 'Left Blink Opening Speed Max', 'Head Pitch Min', 'Head Roll Std', 'Blinks Number', 'Right Pupil Diameter Min', 'Head Roll Max', 'Saccades Duration Max', 'Blinks Duration Max', 'Left Blink Closing Amplitude Max', 'Fixation Duration Median', 'Fixation Duration Max', 'Left Pupil Diameter Std', 'Saccades Number', 'Blinks Duration Min', 'Head Pitch Std', 'Saccades Duration Mean', 'Head Heading Mean', 'Head Roll Mean', 'Left Pupil Diameter Mean', 'Blinks Duration Median', 'Head Pitch Mean']
    else: #SVC, 3 classes
        columns_order = []
elif MODEL == "KNN":
    if BINARY: #KNN, binary
        columns_order = ['Blinks Duration Min', 'Head Pitch Min', 'Saccades Number', 'Saccades Duration Max', 'Left Pupil Diameter Max', 'Right Pupil Diameter Max', 'Head Heading Mean', 'Head Roll Min', 'Head Pitch Std', 'Blinks Number', 'Head Heading Min', 'Left Blink Closing Speed Max', 'Left Pupil Diameter Mean', 'Head Heading Max', 'Head Roll Max', 'Left Pupil Diameter Std', 'Left Blink Closing Amplitude Max', 'Fixation Duration Max', 'Left Blink Opening Speed Max', 'Right Pupil Diameter Std', 'Left Pupil Diameter Min', 'Right Blink Opening Speed Max', 'Right Blink Closing Speed Max', 'Head Pitch Max', 'Head Roll Std', 'Right Pupil Diameter Min', 'Fixation Duration Median', 'Blinks Duration Max', 'Head Heading Std', 'Saccades Duration Mean', 'Head Roll Mean', 'Blinks Duration Median', 'Head Pitch Mean']
    else: #KNN, 3 classes
        columns_order = []
elif MODEL == "ETC":
    if BINARY:
        columns_order = ['Saccades Duration Max', 'Saccades Number', 'Blinks Number', 'Blinks Duration Median', 'Right Blink Opening Speed Max', 'Right Blink Closing Speed Max', 'Head Pitch Mean', 'Left Blink Opening Speed Max', 'Fixation Duration Max', 'Head Heading Mean', 'Left Blink Closing Speed Max', 'Head Pitch Max', 'Blinks Duration Min', 'Left Pupil Diameter Max', 'Head Pitch Std', 'Right Pupil Diameter Max', 'Head Heading Min', 'Head Roll Min', 'Right Pupil Diameter Std', 'Head Heading Max', 'Head Roll Max', 'Right Pupil Diameter Min', 'Head Pitch Min', 'Left Pupil Diameter Min', 'Head Heading Std', 'Head Roll Std', 'Head Roll Mean', 'Left Pupil Diameter Std', 'Saccades Duration Mean', 'Fixation Duration Median', 'Left Pupil Diameter Mean', 'Blinks Duration Max', 'Left Blink Closing Amplitude Max']
    else: #ETC, 3 classes
        columns_order = []
'''
'''
#init_blinks_no_head, right/left average, 
#correlated features dropped (threshold=0.7) => 15 features
if MODEL == "SVC":
    if BINARY: #SVC, binary
        columns_order = ['Blink Closing Speed Max', 'Pupil Diameter Max', 'Blink Opening Speed Max', 'Fixation Duration Median', 'Pupil Diameter Min', 'Blink Closing Amplitude Mean', 'Blinks Duration Median', 'Pupil Diameter Std', 'Blinks Duration Min', 'Saccades Duration Max', 'Blinks Duration Max', 'Fixation Duration Max', 'Saccades Duration Mean', 'Pupil Diameter Mean', 'Blink Closing Amplitude Max']
    else: #SVC, 3 classes
        columns_order = ['Blink Closing Speed Max', 'Pupil Diameter Max', 'Blink Opening Speed Max', 'Pupil Diameter Min', 'Fixation Duration Max', 'Fixation Duration Median', 'Pupil Diameter Std', 'Saccades Duration Max', 'Blinks Duration Min', 'Saccades Duration Mean', 'Blinks Duration Max', 'Blink Closing Amplitude Max', 'Blinks Duration Median', 'Blink Closing Amplitude Mean', 'Pupil Diameter Mean']
elif MODEL == "KNN":
    if BINARY: #KNN, binary
        columns_order = ['Saccades Duration Max', 'Blink Closing Speed Max', 'Blink Closing Amplitude Mean', 'Pupil Diameter Max', 'Blink Opening Speed Max', 'Blinks Duration Median', 'Pupil Diameter Min', 'Fixation Duration Median', 'Blinks Duration Min', 'Fixation Duration Max', 'Blinks Duration Max', 'Pupil Diameter Std', 'Saccades Duration Mean', 'Pupil Diameter Mean', 'Blink Closing Amplitude Max']
    else: #KNN, 3 classes
        columns_order = ['Pupil Diameter Max', 'Fixation Duration Median', 'Saccades Duration Max', 'Blink Closing Speed Max', 'Blink Opening Speed Max', 'Pupil Diameter Min', 'Blink Closing Amplitude Mean', 'Blinks Duration Median', 'Pupil Diameter Std', 'Fixation Duration Max', 'Blinks Duration Min', 'Saccades Duration Mean', 'Blinks Duration Max', 'Blink Closing Amplitude Max', 'Pupil Diameter Mean']
elif MODEL == "ETC":
    if BINARY:
        columns_order = ['Blink Closing Speed Max', 'Blink Opening Speed Max', 'Fixation Duration Max', 'Pupil Diameter Max', 'Pupil Diameter Min', 'Saccades Duration Max', 'Blinks Duration Min', 'Pupil Diameter Std', 'Saccades Duration Mean', 'Blinks Duration Median', 'Blink Closing Amplitude Mean', 'Fixation Duration Median', 'Pupil Diameter Mean', 'Blinks Duration Max', 'Blink Closing Amplitude Max']
'''

print(f"Number of features: {len(columns_order)}")

DATA_DIR = os.path.join("..", "..")
DATA_DIR = os.path.join(DATA_DIR, "Data")
ML_DIR = os.path.join(DATA_DIR, "MLInput")
FIG_DIR = os.path.join(".", "Figures")

RANDOM_STATE = 0

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
    print(randomized_search.best_params_)
    print("after randomized_search.fit")
    
    # Return the best estimator found in this fold
    return randomized_search.best_estimator_

    
# Cross-validation function that handles the pipeline
def cross_val_stratified_with_label_transform(pipeline, data_df, scores, cv, features_init):
    
    df = data_df[features_init]
    
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
        
        #print(y_test_transformed)
        
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
    print(features)
            
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
    