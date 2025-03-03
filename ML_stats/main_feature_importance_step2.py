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
from features import init_blinks_no_min_max
from features import init_blinks_no_min_max_no_head
from features import left, right, left_right_average
from features import left_no_min_max, right_no_min_max
from features import left_right_no_min_max
from features import left_right_average_no_min_max
from features import left_right_diff_no_min_max

left = left_no_min_max
right = right_no_min_max
left_right = left_right_no_min_max
left_right_average = left_right_average_no_min_max
left_right_diff = left_right_diff_no_min_max

#columns_to_select = init_blinks
#columns_to_select = init_blinks_no_head
columns_to_select = init_blinks_no_min_max
#columns_to_select = init_blinks_no_min_max_no_head
#columns_to_select = left_right_no_min_max

CHS = True
BINARY = True

LEFT_RIGHT_AVERAGE = False
LEFT_RIGHT_DIFF = False
LEFT_RIGHT_DROP = False

#MODEL = "SVC"
MODEL = "KNN"

#init_blinks_no_min_max
if MODEL == "SVC":
    if BINARY: #SVC, binary
        columns_order = ['Saccades Duration Mean', 'Right Blink Closing Speed Std', 'Left Blink Opening Amplitude Std', 'Blinks Number', 'Head Heading Median', 'Left Blink Closing Amplitude Mean', 'Left Blink Closing Speed Mean', 'Left Blink Opening Amplitude Mean', 'Left Blink Opening Speed Mean', 'Right Pupil Diameter Median', 'Right Blink Opening Speed Std', 'Left Blink Opening Speed Std', 'Right Blink Closing Amplitude Std', 'Left Blink Closing Speed Std', 'Right Blink Closing Amplitude Mean', 'Right Blink Closing Speed Mean', 'Head Roll Median', 'Right Blink Opening Amplitude Mean', 'Right Blink Opening Amplitude Std', 'Left Blink Closing Amplitude Std', 'Blinks Duration Mean', 'Head Roll Std', 'Right Blink Opening Speed Mean', 'Head Pitch Std', 'Fixation Duration Mean', 'Left Pupil Diameter Mean', 'Head Heading Mean', 'Fixation Duration Std', 'Left Pupil Diameter Std', 'Fixation Duration Median', 'Saccades Number', 'Right Pupil Diameter Mean', 'Head Heading Std', 'Head Roll Mean', 'Left Pupil Diameter Median', 'Right Pupil Diameter Std', 'Saccades Duration Std', 'Saccades Duration Median', 'Blinks Duration Median', 'Blinks Duration Std', 'Head Pitch Median', 'Head Pitch Mean']
    else: #SVC, 3 classes
        columns_order = ['Right Blink Opening Speed Std', 'Blinks Number', 'Right Blink Opening Speed Mean', 'Saccades Duration Mean', 'Right Blink Closing Speed Std', 'Left Blink Opening Amplitude Mean', 'Left Blink Opening Speed Mean', 'Left Blink Closing Speed Mean', 'Left Blink Closing Amplitude Mean', 'Left Blink Opening Amplitude Std', 'Left Blink Opening Speed Std', 'Right Blink Opening Amplitude Mean', 'Left Blink Closing Speed Std', 'Right Blink Closing Amplitude Std', 'Head Heading Median', 'Right Pupil Diameter Mean', 'Right Blink Closing Amplitude Mean', 'Right Pupil Diameter Std', 'Head Roll Median', 'Fixation Duration Mean', 'Left Pupil Diameter Mean', 'Head Heading Std', 'Blinks Duration Mean', 'Head Pitch Std', 'Saccades Number', 'Head Roll Std', 'Left Blink Closing Amplitude Std', 'Right Blink Closing Speed Mean', 'Fixation Duration Median', 'Blinks Duration Std', 'Saccades Duration Median', 'Saccades Duration Std', 'Left Pupil Diameter Std', 'Left Pupil Diameter Median', 'Head Heading Mean', 'Fixation Duration Std', 'Head Roll Mean', 'Head Pitch Median', 'Head Pitch Mean', 'Blinks Duration Median', 'Right Blink Opening Amplitude Std', 'Right Pupil Diameter Median']
elif MODEL == "KNN":
    if BINARY: #KNN, binary
        columns_order = ['Head Heading Mean', 'Saccades Number', 'Saccades Duration Mean', 'Saccades Duration Std', 'Fixation Duration Std', 'Head Heading Median', 'Left Pupil Diameter Mean', 'Head Roll Median', 'Head Pitch Std', 'Saccades Duration Median', 'Right Blink Closing Amplitude Std', 'Left Blink Closing Amplitude Std', 'Left Blink Opening Speed Std', 'Fixation Duration Mean', 'Fixation Duration Median', 'Right Blink Closing Amplitude Mean', 'Left Blink Closing Speed Mean', 'Left Blink Opening Speed Mean', 'Head Roll Std', 'Blinks Number', 'Left Pupil Diameter Std', 'Left Blink Closing Speed Std', 'Right Blink Opening Speed Std', 'Right Blink Closing Speed Mean', 'Right Blink Opening Speed Mean', 'Blinks Duration Median', 'Right Blink Closing Speed Std', 'Left Blink Closing Amplitude Mean', 'Left Blink Opening Amplitude Std', 'Right Pupil Diameter Mean', 'Left Blink Opening Amplitude Mean', 'Right Blink Opening Amplitude Mean', 'Head Pitch Median', 'Blinks Duration Std', 'Right Blink Opening Amplitude Std', 'Right Pupil Diameter Std', 'Left Pupil Diameter Median', 'Head Heading Std', 'Blinks Duration Mean', 'Head Roll Mean', 'Right Pupil Diameter Median', 'Head Pitch Mean']
    else: #KNN, 3 classes
        columns_order = ['Left Pupil Diameter Mean', 'Head Heading Mean', 'Head Heading Median', 'Left Blink Opening Amplitude Std', 'Left Blink Closing Amplitude Std', 'Right Blink Opening Amplitude Std', 'Right Blink Closing Speed Mean', 'Head Roll Median', 'Right Pupil Diameter Median', 'Right Blink Opening Speed Std', 'Head Pitch Mean', 'Right Blink Closing Speed Std', 'Blinks Duration Median', 'Right Blink Closing Amplitude Std', 'Saccades Duration Mean', 'Left Blink Opening Speed Std', 'Fixation Duration Mean', 'Left Blink Closing Speed Std', 'Saccades Number', 'Right Blink Opening Speed Mean', 'Left Blink Closing Speed Mean', 'Blinks Duration Mean', 'Left Blink Opening Speed Mean', 'Left Blink Closing Amplitude Mean', 'Right Blink Opening Amplitude Mean', 'Right Pupil Diameter Mean', 'Saccades Duration Std', 'Blinks Number', 'Left Blink Opening Amplitude Mean', 'Saccades Duration Median', 'Right Blink Closing Amplitude Mean', 'Fixation Duration Std', 'Head Roll Std', 'Head Pitch Std', 'Fixation Duration Median', 'Right Pupil Diameter Std', 'Head Roll Mean', 'Head Heading Std', 'Left Pupil Diameter Std', 'Head Pitch Median', 'Left Pupil Diameter Median', 'Blinks Duration Std']

'''
#init_blinks_no_min_max_no_head
if MODEL == "SVC":
    if BINARY: #SVC, binary
        columns_order = ['Right Blink Closing Speed Std', 'Left Blink Opening Speed Std', 'Left Blink Closing Amplitude Mean', 'Left Blink Closing Speed Std', 'Left Blink Closing Speed Mean', 'Left Blink Opening Speed Mean', 'Blinks Number', 'Left Blink Opening Amplitude Mean', 'Left Blink Opening Amplitude Std', 'Right Blink Opening Speed Std', 'Right Blink Closing Amplitude Mean', 'Right Blink Closing Amplitude Std', 'Right Blink Opening Speed Mean', 'Right Pupil Diameter Mean', 'Right Blink Opening Amplitude Mean', 'Right Pupil Diameter Std', 'Left Pupil Diameter Mean', 'Blinks Duration Median', 'Saccades Number', 'Fixation Duration Median', 'Right Blink Closing Speed Mean', 'Saccades Duration Mean', 'Saccades Duration Std', 'Fixation Duration Std', 'Left Pupil Diameter Std', 'Left Blink Closing Amplitude Std', 'Fixation Duration Mean', 'Right Pupil Diameter Median', 'Left Pupil Diameter Median', 'Saccades Duration Median', 'Right Blink Opening Amplitude Std', 'Blinks Duration Mean', 'Blinks Duration Std']
    else: #SVC, 3 classes
        columns_order = ['Right Blink Closing Speed Mean', 'Left Blink Closing Speed Mean', 'Blinks Number', 'Left Blink Closing Amplitude Mean', 'Left Blink Opening Speed Mean', 'Right Blink Closing Speed Std', 'Left Blink Opening Amplitude Mean', 'Left Blink Closing Speed Std', 'Right Blink Closing Amplitude Mean', 'Right Blink Opening Speed Std', 'Left Blink Opening Amplitude Std', 'Right Blink Opening Amplitude Mean', 'Left Pupil Diameter Mean', 'Saccades Duration Mean', 'Right Blink Closing Amplitude Std', 'Fixation Duration Median', 'Left Blink Opening Speed Std', 'Blinks Duration Median', 'Right Pupil Diameter Mean', 'Right Pupil Diameter Std', 'Fixation Duration Std', 'Left Blink Closing Amplitude Std', 'Saccades Number', 'Left Pupil Diameter Std', 'Fixation Duration Mean', 'Saccades Duration Median', 'Saccades Duration Std', 'Right Blink Opening Speed Mean', 'Right Pupil Diameter Median', 'Blinks Duration Mean', 'Left Pupil Diameter Median', 'Right Blink Opening Amplitude Std', 'Blinks Duration Std']
elif MODEL == "KNN":
    if BINARY: #KNN, binary
        columns_order = ['Saccades Number', 'Saccades Duration Mean', 'Saccades Duration Std', 'Left Pupil Diameter Mean', 'Right Blink Closing Speed Std', 'Saccades Duration Median', 'Blinks Duration Mean', 'Right Blink Closing Amplitude Std', 'Right Blink Opening Speed Std', 'Left Blink Opening Amplitude Mean', 'Fixation Duration Mean', 'Right Pupil Diameter Median', 'Left Blink Closing Speed Std', 'Left Blink Opening Speed Std', 'Left Blink Closing Amplitude Std', 'Left Blink Opening Amplitude Std', 'Right Blink Closing Speed Mean', 'Right Blink Opening Speed Mean', 'Blinks Number', 'Right Blink Opening Amplitude Std', 'Left Pupil Diameter Std', 'Left Blink Closing Amplitude Mean', 'Right Blink Opening Amplitude Mean', 'Left Blink Opening Speed Mean', 'Fixation Duration Median', 'Left Blink Closing Speed Mean', 'Right Pupil Diameter Std', 'Right Pupil Diameter Mean', 'Right Blink Closing Amplitude Mean', 'Fixation Duration Std', 'Blinks Duration Median', 'Blinks Duration Std', 'Left Pupil Diameter Median']
    else: #KNN, 3 classes
        columns_order = ['Blinks Duration Mean', 'Blinks Duration Median', 'Fixation Duration Mean', 'Left Pupil Diameter Mean', 'Right Blink Opening Amplitude Std', 'Saccades Number', 'Right Blink Closing Speed Std', 'Saccades Duration Mean', 'Left Blink Closing Amplitude Std', 'Right Blink Opening Amplitude Mean', 'Right Pupil Diameter Median', 'Left Blink Closing Speed Std', 'Left Blink Opening Amplitude Mean', 'Left Blink Opening Speed Std', 'Left Blink Opening Amplitude Std', 'Right Blink Opening Speed Std', 'Left Blink Closing Amplitude Mean', 'Right Blink Opening Speed Mean', 'Left Blink Opening Speed Mean', 'Left Blink Closing Speed Mean', 'Fixation Duration Median', 'Right Blink Closing Amplitude Mean', 'Right Blink Closing Speed Mean', 'Blinks Number', 'Fixation Duration Std', 'Saccades Duration Std', 'Left Pupil Diameter Std', 'Right Blink Closing Amplitude Std', 'Left Pupil Diameter Median', 'Saccades Duration Median', 'Right Pupil Diameter Std', 'Right Pupil Diameter Mean', 'Blinks Duration Std']
'''

'''
#left_right_no_min_max, orig.
if MODEL == "SVC":
    if BINARY: #SVC, binary
        columns_order = ['Left Blink Opening Amplitude Mean', 'Left Blink Closing Speed Mean', 'Right Blink Closing Amplitude Mean', 'Left Blink Closing Amplitude Mean', 'Right Blink Opening Speed Mean', 'Left Blink Opening Speed Mean', 'Right Blink Closing Speed Mean', 'Left Blink Closing Speed Std', 'Right Blink Opening Amplitude Mean', 'Left Blink Opening Amplitude Std', 'Right Blink Closing Speed Std', 'Right Blink Closing Amplitude Std', 'Right Blink Opening Speed Std', 'Left Pupil Diameter Mean', 'Right Pupil Diameter Mean', 'Left Blink Opening Speed Std', 'Left Pupil Diameter Std', 'Right Pupil Diameter Std', 'Right Blink Opening Amplitude Std', 'Left Blink Closing Amplitude Std', 'Right Pupil Diameter Median', 'Left Pupil Diameter Median']
    else: #SVC, 3 classes
        columns_order = ['Left Blink Opening Amplitude Mean', 'Left Blink Closing Speed Mean', 'Right Blink Closing Amplitude Mean', 'Left Blink Closing Amplitude Mean', 'Left Blink Closing Speed Std', 'Right Blink Opening Amplitude Mean', 'Left Blink Opening Speed Mean', 'Right Blink Opening Speed Mean', 'Right Blink Closing Speed Mean', 'Right Blink Opening Amplitude Std', 'Left Blink Opening Amplitude Std', 'Right Blink Opening Speed Std', 'Left Blink Closing Amplitude Std', 'Left Pupil Diameter Mean', 'Right Pupil Diameter Mean', 'Right Blink Closing Speed Std', 'Right Pupil Diameter Std', 'Left Pupil Diameter Std', 'Left Blink Opening Speed Std', 'Right Blink Closing Amplitude Std', 'Right Pupil Diameter Median', 'Left Pupil Diameter Median']
elif MODEL == "KNN":
    if BINARY: #KNN, binary
        columns_order = ['Left Blink Closing Amplitude Mean', 'Right Blink Opening Amplitude Mean', 'Right Blink Closing Amplitude Mean', 'Left Blink Opening Amplitude Mean', 'Left Blink Opening Speed Mean', 'Right Blink Opening Speed Mean', 'Right Blink Closing Speed Mean', 'Left Blink Closing Speed Mean', 'Left Blink Opening Amplitude Std', 'Right Blink Opening Speed Std', 'Left Blink Closing Speed Std', 'Left Blink Closing Amplitude Std', 'Right Blink Opening Amplitude Std', 'Right Blink Closing Speed Std', 'Left Pupil Diameter Mean', 'Right Pupil Diameter Median', 'Left Blink Opening Speed Std', 'Left Pupil Diameter Median', 'Left Pupil Diameter Std', 'Right Pupil Diameter Std', 'Right Pupil Diameter Mean', 'Right Blink Closing Amplitude Std']
    else: #KNN, 3 classes
        columns_order = ['Right Blink Opening Amplitude Mean', 'Left Blink Opening Amplitude Mean', 'Left Blink Closing Amplitude Mean', 'Right Blink Closing Amplitude Mean', 'Right Blink Closing Speed Mean', 'Right Blink Opening Speed Mean', 'Left Blink Opening Speed Mean', 'Left Blink Closing Speed Mean', 'Left Blink Opening Amplitude Std', 'Right Blink Closing Speed Std', 'Left Blink Closing Speed Std', 'Right Blink Opening Amplitude Std', 'Left Blink Closing Amplitude Std', 'Left Pupil Diameter Mean', 'Right Blink Opening Speed Std', 'Right Pupil Diameter Median', 'Left Blink Opening Speed Std', 'Left Pupil Diameter Std', 'Right Pupil Diameter Std', 'Right Pupil Diameter Mean', 'Right Blink Closing Amplitude Std', 'Left Pupil Diameter Median']
'''
'''
#left_right_average_no_min_max, left_right_diff_no_min_max
if MODEL == "SVC":
    if BINARY: #SVC, binary
        columns_order = ['Blink Closing Amplitude Mean', 'Blink Opening Amplitude Mean', 'Blink Opening Speed Mean', 'Blink Closing Speed Mean', 'Blink Opening Speed Mean Diff', 'Blink Closing Amplitude Mean Diff', 'Blink Opening Amplitude Std', 'Blink Closing Speed Std Diff', 'Blink Opening Speed Std', 'Blink Opening Speed Std Diff', 'Blink Closing Speed Mean Diff', 'Blink Opening Amplitude Mean Diff', 'Pupil Diameter Std', 'Pupil Diameter Std Diff', 'Blink Closing Speed Std', 'Blink Closing Amplitude Std Diff', 'Blink Closing Amplitude Std', 'Blink Opening Amplitude Std Diff', 'Pupil Diameter Mean Diff', 'Pupil Diameter Median', 'Pupil Diameter Mean', 'Pupil Diameter Median Diff']
    else: #SVC, 3 classes
        columns_order = ['Blink Closing Amplitude Mean', 'Blink Opening Speed Mean', 'Blink Opening Amplitude Mean', 'Blink Closing Speed Mean Diff', 'Blink Opening Speed Mean Diff', 'Blink Closing Speed Std', 'Blink Closing Speed Std Diff', 'Blink Opening Speed Std Diff', 'Blink Opening Amplitude Mean Diff', 'Blink Opening Speed Std', 'Blink Opening Amplitude Std', 'Blink Closing Amplitude Mean Diff', 'Pupil Diameter Std Diff', 'Blink Closing Speed Mean', 'Pupil Diameter Mean', 'Blink Closing Amplitude Std Diff', 'Blink Opening Amplitude Std Diff', 'Pupil Diameter Std', 'Pupil Diameter Mean Diff', 'Blink Closing Amplitude Std', 'Pupil Diameter Median Diff', 'Pupil Diameter Median']
elif MODEL == "KNN":
    if BINARY: #KNN, binary
        columns_order = ['Blink Closing Amplitude Mean Diff', 'Blink Opening Amplitude Mean Diff', 'Blink Closing Amplitude Std Diff', 'Blink Opening Speed Mean Diff', 'Blink Opening Amplitude Mean', 'Blink Closing Amplitude Mean', 'Blink Closing Speed Mean Diff', 'Blink Closing Speed Mean', 'Blink Opening Speed Mean', 'Blink Opening Amplitude Std', 'Blink Closing Amplitude Std', 'Blink Opening Speed Std Diff', 'Blink Closing Speed Std Diff', 'Blink Closing Speed Std', 'Pupil Diameter Mean', 'Pupil Diameter Median Diff', 'Blink Opening Amplitude Std Diff', 'Pupil Diameter Std Diff', 'Pupil Diameter Std', 'Pupil Diameter Median', 'Blink Opening Speed Std', 'Pupil Diameter Mean Diff']
    else: #KNN, 3 classes
        columns_order = ['Blink Opening Amplitude Mean Diff', 'Blink Opening Amplitude Std Diff', 'Blink Opening Amplitude Mean', 'Blink Opening Amplitude Std', 'Blink Closing Amplitude Mean', 'Blink Closing Amplitude Mean Diff', 'Blink Closing Speed Mean Diff', 'Blink Opening Speed Mean Diff', 'Blink Opening Speed Mean', 'Blink Closing Speed Mean', 'Blink Opening Speed Std Diff', 'Blink Opening Speed Std', 'Blink Closing Speed Std Diff', 'Pupil Diameter Mean Diff', 'Pupil Diameter Mean', 'Blink Closing Amplitude Std', 'Pupil Diameter Std Diff', 'Pupil Diameter Std', 'Blink Closing Amplitude Std Diff', 'Pupil Diameter Median Diff', 'Blink Closing Speed Std', 'Pupil Diameter Median']
'''

'''
#left_right_no_min_max, orig. + left_right_diff
if MODEL == "SVC":
    if BINARY: #SVC, binary
        columns_order = ['Left Blink Closing Amplitude Std', 'Right Blink Opening Amplitude Std', 'Right Blink Closing Amplitude Mean', 'Left Blink Opening Amplitude Mean', 'Left Blink Closing Amplitude Mean', 'Left Blink Closing Speed Mean', 'Left Blink Opening Speed Mean', 'Right Blink Opening Amplitude Mean', 'Left Blink Closing Speed Std', 'Right Blink Closing Speed Mean', 'Right Blink Opening Speed Mean', 'Left Blink Opening Speed Std', 'Blink Opening Speed Mean Diff', 'Left Blink Opening Amplitude Std', 'Blink Closing Amplitude Mean Diff', 'Blink Closing Speed Std Diff', 'Right Pupil Diameter Std', 'Right Blink Opening Speed Std', 'Right Pupil Diameter Median', 'Blink Opening Speed Std Diff', 'Blink Closing Speed Mean Diff', 'Left Pupil Diameter Mean', 'Blink Opening Amplitude Mean Diff', 'Left Pupil Diameter Std', 'Pupil Diameter Std Diff', 'Right Blink Closing Speed Std', 'Blink Closing Amplitude Std Diff', 'Blink Opening Amplitude Std Diff', 'Right Blink Closing Amplitude Std', 'Pupil Diameter Mean Diff', 'Left Pupil Diameter Median', 'Right Pupil Diameter Mean', 'Pupil Diameter Median Diff']
    else: #SVC, 3 classes
        columns_order = ['Left Blink Closing Amplitude Mean', 'Left Blink Opening Amplitude Mean', 'Right Blink Closing Amplitude Mean', 'Left Blink Closing Speed Mean', 'Left Blink Opening Speed Mean', 'Left Blink Closing Speed Std', 'Blink Closing Speed Mean Diff', 'Right Blink Opening Amplitude Mean', 'Right Blink Opening Speed Mean', 'Blink Opening Speed Mean Diff', 'Right Blink Closing Speed Mean', 'Left Blink Closing Amplitude Std', 'Blink Closing Amplitude Mean Diff', 'Left Blink Opening Speed Std', 'Blink Closing Speed Std Diff', 'Right Blink Opening Amplitude Std', 'Blink Opening Speed Std Diff', 'Blink Opening Amplitude Mean Diff', 'Left Blink Opening Amplitude Std', 'Left Pupil Diameter Mean', 'Pupil Diameter Std Diff', 'Right Blink Opening Speed Std', 'Right Pupil Diameter Mean', 'Pupil Diameter Mean Diff', 'Right Pupil Diameter Std', 'Right Pupil Diameter Median', 'Right Blink Closing Speed Std', 'Left Pupil Diameter Std', 'Blink Closing Amplitude Std Diff', 'Blink Opening Amplitude Std Diff', 'Pupil Diameter Median Diff', 'Left Pupil Diameter Median', 'Right Blink Closing Amplitude Std']
elif MODEL == "KNN":
    if BINARY: #KNN, binary
        columns_order = ['Blink Closing Amplitude Mean Diff', 'Left Pupil Diameter Mean', 'Blink Closing Amplitude Std Diff', 'Blink Opening Amplitude Std Diff', 'Right Pupil Diameter Median', 'Blink Opening Amplitude Mean Diff', 'Right Blink Opening Speed Std', 'Left Blink Opening Speed Std', 'Blink Opening Speed Mean Diff', 'Blink Closing Speed Mean Diff', 'Left Blink Opening Amplitude Mean', 'Left Blink Closing Amplitude Mean', 'Right Blink Closing Speed Mean', 'Left Blink Closing Speed Std', 'Left Blink Opening Amplitude Std', 'Pupil Diameter Mean Diff', 'Left Blink Closing Speed Mean', 'Left Blink Opening Speed Mean', 'Right Blink Closing Amplitude Mean', 'Blink Opening Speed Std Diff', 'Right Blink Opening Amplitude Mean', 'Left Blink Closing Amplitude Std', 'Right Blink Opening Amplitude Std', 'Right Blink Opening Speed Mean', 'Right Pupil Diameter Std', 'Right Pupil Diameter Mean', 'Right Blink Closing Amplitude Std', 'Blink Closing Speed Std Diff', 'Pupil Diameter Std Diff', 'Left Pupil Diameter Std', 'Left Pupil Diameter Median', 'Right Blink Closing Speed Std', 'Pupil Diameter Median Diff']
    else: #KNN, 3 classes
        columns_order = ['Blink Closing Amplitude Std Diff', 'Left Blink Closing Speed Std', 'Left Blink Opening Speed Std', 'Right Pupil Diameter Median', 'Right Blink Opening Speed Std', 'Left Pupil Diameter Std', 'Right Blink Opening Amplitude Std', 'Right Blink Closing Amplitude Std', 'Left Blink Closing Speed Mean', 'Right Blink Opening Amplitude Mean', 'Left Blink Closing Amplitude Mean', 'Left Blink Opening Amplitude Mean', 'Blink Opening Amplitude Mean Diff', 'Right Blink Closing Amplitude Mean', 'Left Blink Opening Speed Mean', 'Blink Closing Amplitude Mean Diff', 'Blink Opening Speed Mean Diff', 'Right Blink Opening Speed Mean', 'Left Blink Opening Amplitude Std', 'Blink Opening Speed Std Diff', 'Blink Closing Speed Mean Diff', 'Right Blink Closing Speed Mean', 'Blink Closing Speed Std Diff', 'Pupil Diameter Median Diff', 'Left Pupil Diameter Mean', 'Left Blink Closing Amplitude Std', 'Right Pupil Diameter Mean', 'Pupil Diameter Std Diff', 'Blink Opening Amplitude Std Diff', 'Right Pupil Diameter Std', 'Right Blink Closing Speed Std', 'Left Pupil Diameter Median', 'Pupil Diameter Mean Diff']
'''

'''
#init_blinks_no_min_max + left_right_diff
if MODEL == "SVC":
    if BINARY: #SVC, binary
        columns_order = []
    else: #SVC, 3 classes
        columns_order = []
elif MODEL == "KNN":
    if BINARY: #KNN, binary
        columns_order = ['Saccades Number', 'Saccades Duration Mean', 'Head Heading Median', 'Saccades Duration Std', 'Head Heading Mean', 'Blink Opening Speed Mean Diff', 'Blinks Duration Median', 'Saccades Duration Median', 'Fixation Duration Mean', 'Fixation Duration Std', 'Fixation Duration Median', 'Blinks Duration Mean', 'Head Pitch Mean', 'Blink Closing Amplitude Mean Diff', 'Right Pupil Diameter Std', 'Blinks Number', 'Left Blink Closing Speed Mean', 'Blink Opening Speed Std Diff', 'Head Roll Median', 'Right Blink Closing Speed Std', 'Right Blink Closing Amplitude Mean', 'Pupil Diameter Median Diff', 'Right Blink Closing Speed Mean', 'Head Roll Std', 'Left Blink Closing Amplitude Std', 'Left Pupil Diameter Mean', 'Blink Closing Speed Mean Diff', 'Left Blink Opening Speed Mean', 'Right Blink Opening Speed Std', 'Right Blink Opening Speed Mean', 'Right Blink Opening Amplitude Mean', 'Head Roll Mean', 'Right Blink Closing Amplitude Std', 'Blink Closing Speed Std Diff', 'Left Blink Opening Speed Std', 'Blink Opening Amplitude Mean Diff', 'Blink Closing Amplitude Std Diff', 'Left Blink Opening Amplitude Std', 'Left Blink Closing Amplitude Mean', 'Left Blink Opening Amplitude Mean', 'Right Pupil Diameter Median', 'Left Blink Closing Speed Std', 'Pupil Diameter Mean Diff', 'Head Pitch Std', 'Right Blink Opening Amplitude Std', 'Left Pupil Diameter Median', 'Blinks Duration Std', 'Pupil Diameter Std Diff', 'Head Heading Std', 'Left Pupil Diameter Std', 'Right Pupil Diameter Mean', 'Blink Opening Amplitude Std Diff', 'Head Pitch Median']
    else: #KNN, 3 classes
        columns_order = []
'''

'''
#init_blinks_no_min_max, left_right_average_no_min_max, left_right_diff_no_min_max
if MODEL == "SVC":
    if BINARY: #SVC, binary
        columns_order = []
    else: #SVC, 3 classes
        columns_order = []
elif MODEL == "KNN":
    if BINARY: #KNN, binary
        columns_order = ['Blink Opening Amplitude Mean Diff', 'Blink Opening Speed Std Diff', 'Blink Closing Speed Mean Diff', 'Saccades Number', 'Saccades Duration Mean', 'Head Heading Median', 'Saccades Duration Std', 'Head Pitch Median', 'Saccades Duration Median', 'Head Heading Mean', 'Blink Closing Speed Std Diff', 'Fixation Duration Mean', 'Pupil Diameter Std', 'Blink Opening Speed Std', 'Pupil Diameter Std Diff', 'Blink Closing Speed Std', 'Fixation Duration Std', 'Blinks Number', 'Blink Closing Amplitude Mean', 'Blink Opening Amplitude Std Diff', 'Pupil Diameter Median Diff', 'Blink Opening Amplitude Std', 'Blink Closing Amplitude Std', 'Blink Opening Amplitude Mean', 'Head Roll Mean', 'Blink Closing Amplitude Mean Diff', 'Blinks Duration Std', 'Blinks Duration Median', 'Head Roll Std', 'Blink Opening Speed Mean', 'Pupil Diameter Median', 'Blink Opening Speed Mean Diff', 'Blink Closing Speed Mean', 'Head Pitch Std', 'Pupil Diameter Mean', 'Fixation Duration Median', 'Blink Closing Amplitude Std Diff', 'Head Heading Std', 'Head Roll Median', 'Blinks Duration Mean', 'Pupil Diameter Mean Diff', 'Head Pitch Mean']
    else: #KNN, 3 classes
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

'''
#init_blinks, 64 features
if MODEL == "SVC":
    if BINARY: #SVC, binary
        columns_order = ['Right Pupil Diameter Median', 'Blinks Duration Max', 'Left Blink Opening Speed Std', 'Right Blink Closing Amplitude Std', 'Left Blink Opening Amplitude Std', 'Head Roll Min', 'Left Blink Closing Speed Std', 'Right Blink Closing Amplitude Mean', 'Right Pupil Diameter Std', 'Blinks Number', 'Fixation Duration Mean', 'Right Blink Opening Amplitude Std', 'Saccades Duration Mean', 'Left Blink Opening Amplitude Mean', 'Left Blink Closing Speed Mean', 'Head Roll Median', 'Head Heading Max', 'Head Pitch Max', 'Left Blink Opening Speed Mean', 'Saccades Duration Max', 'Right Blink Closing Speed Std', 'Left Blink Closing Amplitude Mean', 'Fixation Duration Std', 'Head Heading Min', 'Right Pupil Diameter Min', 'Head Roll Max', 'Left Blink Closing Speed Max', 'Head Heading Std', 'Right Blink Closing Speed Max', 'Right Blink Opening Speed Std', 'Right Blink Opening Amplitude Max', 'Left Pupil Diameter Min', 'Right Pupil Diameter Max', 'Left Pupil Diameter Max', 'Blinks Duration Std', 'Left Blink Closing Amplitude Max', 'Right Blink Opening Speed Max', 'Right Blink Opening Amplitude Mean', 'Left Blink Opening Speed Max', 'Blinks Duration Median', 'Head Heading Mean', 'Head Pitch Min', 'Right Blink Opening Speed Mean', 'Fixation Duration Max', 'Head Pitch Std', 'Head Heading Median', 'Right Pupil Diameter Mean', 'Head Roll Std', 'Saccades Number', 'Fixation Duration Median', 'Head Roll Mean', 'Right Blink Closing Amplitude Max', 'Left Pupil Diameter Mean', 'Blinks Duration Min', 'Left Pupil Diameter Median', 'Saccades Duration Std', 'Left Pupil Diameter Std', 'Saccades Duration Median', 'Left Blink Opening Amplitude Max', 'Blinks Duration Mean', 'Right Blink Closing Speed Mean', 'Head Pitch Median', 'Left Blink Closing Amplitude Std', 'Head Pitch Mean']
    else: #SVC, 3 classes
        columns_order = ['Right Blink Opening Speed Mean', 'Blinks Number', 'Right Blink Opening Speed Std', 'Left Blink Opening Speed Mean', 'Left Blink Closing Speed Mean', 'Left Blink Opening Speed Std', 'Left Blink Closing Amplitude Std', 'Left Blink Opening Amplitude Std', 'Right Blink Closing Amplitude Std', 'Left Blink Closing Speed Std', 'Right Blink Closing Speed Std', 'Left Blink Closing Amplitude Mean', 'Head Heading Median', 'Right Blink Closing Amplitude Mean', 'Left Blink Opening Amplitude Mean', 'Fixation Duration Std', 'Right Pupil Diameter Mean', 'Blinks Duration Max', 'Head Heading Max', 'Saccades Duration Max', 'Fixation Duration Mean', 'Head Heading Min', 'Head Roll Max', 'Right Pupil Diameter Min', 'Head Roll Mean', 'Saccades Duration Mean', 'Right Pupil Diameter Max', 'Blinks Duration Min', 'Left Blink Closing Speed Max', 'Right Blink Opening Amplitude Mean', 'Left Pupil Diameter Min', 'Right Blink Closing Speed Max', 'Head Roll Min', 'Blinks Duration Mean', 'Head Heading Std', 'Left Blink Opening Speed Max', 'Right Blink Opening Speed Max', 'Left Pupil Diameter Max', 'Head Pitch Max', 'Left Pupil Diameter Std', 'Left Pupil Diameter Mean', 'Fixation Duration Max', 'Head Pitch Std', 'Head Roll Std', 'Right Blink Opening Amplitude Std', 'Right Blink Opening Amplitude Max', 'Head Pitch Min', 'Left Blink Opening Amplitude Max', 'Head Pitch Median', 'Fixation Duration Median', 'Left Blink Closing Amplitude Max', 'Right Pupil Diameter Std', 'Right Blink Closing Amplitude Max', 'Left Pupil Diameter Median', 'Right Blink Closing Speed Mean', 'Saccades Duration Median', 'Saccades Duration Std', 'Blinks Duration Std', 'Head Heading Mean', 'Blinks Duration Median', 'Head Roll Median', 'Saccades Number', 'Head Pitch Mean', 'Right Pupil Diameter Median']
elif MODEL == "KNN":
    if BINARY: #KNN, binary
        columns_order = ['Blinks Duration Max', 'Right Pupil Diameter Median', 'Head Heading Mean', 'Fixation Duration Max', 'Saccades Duration Max', 'Left Blink Opening Amplitude Mean', 'Head Heading Median', 'Blinks Duration Median', 'Head Heading Min', 'Head Heading Max', 'Right Pupil Diameter Min', 'Right Blink Opening Amplitude Mean', 'Right Blink Opening Speed Max', 'Left Blink Opening Amplitude Std', 'Head Heading Std', 'Left Blink Closing Speed Max', 'Right Blink Closing Amplitude Mean', 'Head Pitch Max', 'Right Blink Closing Amplitude Std', 'Saccades Duration Median', 'Saccades Number', 'Left Blink Closing Amplitude Mean', 'Left Blink Closing Amplitude Std', 'Head Roll Std', 'Saccades Duration Mean', 'Saccades Duration Std', 'Head Pitch Mean', 'Fixation Duration Mean', 'Fixation Duration Std', 'Fixation Duration Median', 'Blinks Number', 'Right Pupil Diameter Max', 'Blinks Duration Mean', 'Head Pitch Min', 'Right Blink Closing Speed Mean', 'Left Blink Opening Amplitude Max', 'Right Blink Opening Amplitude Max', 'Left Blink Opening Speed Mean', 'Head Roll Mean', 'Right Blink Opening Speed Mean', 'Right Blink Closing Speed Std', 'Left Pupil Diameter Std', 'Left Blink Opening Speed Std', 'Left Blink Closing Amplitude Max', 'Left Blink Closing Speed Std', 'Right Blink Opening Speed Std', 'Head Roll Median', 'Head Roll Min', 'Left Blink Opening Speed Max', 'Right Blink Closing Amplitude Max', 'Right Blink Closing Speed Max', 'Left Pupil Diameter Min', 'Left Pupil Diameter Median', 'Right Blink Opening Amplitude Std', 'Left Pupil Diameter Mean', 'Head Pitch Std', 'Left Pupil Diameter Max', 'Head Roll Max', 'Blinks Duration Std', 'Blinks Duration Min', 'Left Blink Closing Speed Mean', 'Right Pupil Diameter Std', 'Right Pupil Diameter Mean', 'Head Pitch Median']
    else: #KNN, 3 classes
        columns_order = ['Head Heading Mean', 'Blinks Duration Max', 'Right Pupil Diameter Max', 'Head Roll Median', 'Left Pupil Diameter Median', 'Left Blink Closing Speed Max', 'Saccades Number', 'Head Heading Max', 'Left Blink Opening Speed Max', 'Right Blink Opening Amplitude Max', 'Left Blink Opening Amplitude Std', 'Left Blink Opening Amplitude Max', 'Right Pupil Diameter Median', 'Left Pupil Diameter Max', 'Right Blink Closing Amplitude Max', 'Left Blink Closing Amplitude Mean', 'Head Pitch Min', 'Left Blink Closing Amplitude Std', 'Left Blink Closing Speed Std', 'Right Blink Closing Speed Std', 'Right Blink Closing Amplitude Std', 'Left Blink Closing Amplitude Max', 'Head Heading Median', 'Saccades Duration Max', 'Right Blink Opening Amplitude Std', 'Right Blink Opening Speed Std', 'Right Blink Opening Speed Mean', 'Head Roll Std', 'Blinks Duration Min', 'Head Pitch Mean', 'Blinks Duration Mean', 'Left Blink Opening Speed Std', 'Saccades Duration Mean', 'Right Blink Closing Amplitude Mean', 'Fixation Duration Mean', 'Head Roll Mean', 'Blinks Number', 'Head Heading Min', 'Left Blink Closing Speed Mean', 'Head Roll Max', 'Right Blink Closing Speed Mean', 'Fixation Duration Median', 'Left Blink Opening Amplitude Mean', 'Right Pupil Diameter Min', 'Head Roll Min', 'Right Blink Closing Speed Max', 'Right Blink Opening Amplitude Mean', 'Left Pupil Diameter Min', 'Right Blink Opening Speed Max', 'Head Pitch Std', 'Fixation Duration Std', 'Head Pitch Max', 'Left Blink Opening Speed Mean', 'Blinks Duration Median', 'Saccades Duration Std', 'Fixation Duration Max', 'Left Pupil Diameter Mean', 'Head Heading Std', 'Left Pupil Diameter Std', 'Saccades Duration Median', 'Right Pupil Diameter Std', 'Blinks Duration Std', 'Head Pitch Median', 'Right Pupil Diameter Mean']
'''
'''
#init_blinks, + left/right diff.
if MODEL == "SVC":
    if BINARY: #SVC, binary
        columns_order = []
    else: #SVC, 3 classes
        columns_order = []
elif MODEL == "KNN":
    if BINARY: #KNN, binary
        columns_order = ['Saccades Number', 'Head Heading Mean', 'Saccades Duration Mean', 'Right Pupil Diameter Min', 'Saccades Duration Std', 'Left Blink Closing Speed Max', 'Saccades Duration Median', 'Right Pupil Diameter Max', 'Left Blink Opening Amplitude Max', 'Blink Opening Speed Std Diff', 'Blink Opening Speed Mean Diff', 'Fixation Duration Mean', 'Left Blink Closing Speed Mean', 'Head Heading Std', 'Blink Closing Speed Mean Diff', 'Left Pupil Diameter Max', 'Head Heading Median', 'Saccades Duration Max', 'Fixation Duration Std', 'Fixation Duration Median', 'Fixation Duration Max', 'Left Pupil Diameter Std', 'Blink Closing Amplitude Std Diff', 'Left Blink Closing Amplitude Max', 'Left Blink Closing Speed Std', 'Blinks Number', 'Blinks Duration Mean', 'Blink Closing Speed Std Diff', 'Blinks Duration Std', 'Head Roll Std', 'Blink Opening Amplitude Mean Diff', 'Left Blink Closing Amplitude Std', 'Left Blink Closing Amplitude Mean', 'Right Blink Opening Speed Max', 'Left Blink Opening Amplitude Mean', 'Blinks Duration Median', 'Left Pupil Diameter Mean', 'Left Blink Opening Speed Mean', 'Left Blink Opening Amplitude Std', 'Right Blink Closing Speed Std', 'Head Pitch Mean', 'Head Roll Mean', 'Head Pitch Std', 'Left Blink Opening Speed Std', 'Right Blink Closing Amplitude Std', 'Right Blink Opening Speed Std', 'Head Pitch Median', 'Head Pitch Max', 'Head Roll Median', 'Right Blink Closing Speed Mean', 'Head Heading Min', 'Head Heading Max', 'Head Roll Min', 'Right Pupil Diameter Mean', 'Right Blink Opening Amplitude Std', 'Right Blink Closing Amplitude Mean', 'Blinks Duration Max', 'Blink Opening Amplitude Std Diff', 'Right Blink Opening Amplitude Max', 'Pupil Diameter Std Diff', 'Right Blink Closing Speed Max', 'Pupil Diameter Mean Diff', 'Right Blink Opening Speed Mean', 'Head Roll Max', 'Left Blink Opening Speed Max', 'Left Pupil Diameter Min', 'Right Blink Opening Amplitude Mean', 'Right Pupil Diameter Median', 'Blink Closing Amplitude Mean Diff', 'Blinks Duration Min', 'Head Pitch Min', 'Pupil Diameter Median Diff', 'Left Pupil Diameter Median', 'Right Pupil Diameter Std', 'Right Blink Closing Amplitude Max']
    else: #KNN, 3 classes
        columns_order = []
'''

'''
#init_blinks_no_head
if MODEL == "SVC":
    if BINARY: #SVC, binary
        columns_order = ['Right Pupil Diameter Median', 'Right Pupil Diameter Mean', 'Right Blink Closing Amplitude Std', 'Left Blink Closing Amplitude Std', 'Left Blink Opening Amplitude Std', 'Left Blink Closing Speed Max', 'Right Pupil Diameter Min', 'Right Blink Closing Speed Max', 'Left Blink Opening Speed Std', 'Right Blink Opening Speed Std', 'Saccades Duration Std', 'Right Blink Closing Speed Std', 'Right Pupil Diameter Std', 'Blinks Number', 'Left Blink Opening Speed Mean', 'Left Blink Closing Speed Std', 'Right Blink Opening Amplitude Std', 'Left Blink Closing Amplitude Mean', 'Left Blink Closing Speed Mean', 'Left Blink Opening Amplitude Mean', 'Left Pupil Diameter Min', 'Left Blink Opening Speed Max', 'Right Blink Closing Amplitude Mean', 'Left Blink Closing Amplitude Max', 'Blinks Duration Mean', 'Right Blink Opening Amplitude Mean', 'Right Blink Opening Amplitude Max', 'Right Blink Opening Speed Max', 'Fixation Duration Std', 'Right Blink Opening Speed Mean', 'Fixation Duration Max', 'Right Blink Closing Speed Mean', 'Fixation Duration Mean', 'Blinks Duration Max', 'Right Pupil Diameter Max', 'Left Pupil Diameter Mean', 'Left Pupil Diameter Max', 'Fixation Duration Median', 'Blinks Duration Median', 'Saccades Number', 'Left Pupil Diameter Std', 'Blinks Duration Min', 'Saccades Duration Mean', 'Left Pupil Diameter Median', 'Saccades Duration Max', 'Saccades Duration Median', 'Blinks Duration Std', 'Left Blink Opening Amplitude Max', 'Right Blink Closing Amplitude Max']
    else: #SVC, 3 classes
        columns_order = ['Right Blink Closing Speed Std', 'Left Blink Closing Speed Std', 'Right Blink Closing Amplitude Std', 'Left Blink Opening Amplitude Std', 'Left Blink Opening Speed Std', 'Right Blink Closing Amplitude Mean', 'Blinks Number', 'Right Blink Opening Speed Std', 'Left Blink Closing Amplitude Std', 'Left Blink Closing Amplitude Mean', 'Left Blink Opening Speed Mean', 'Right Pupil Diameter Mean', 'Left Blink Closing Speed Mean', 'Right Blink Opening Amplitude Std', 'Left Blink Opening Amplitude Mean', 'Right Pupil Diameter Min', 'Right Pupil Diameter Std', 'Left Blink Closing Speed Max', 'Right Blink Closing Speed Max', 'Saccades Duration Mean', 'Left Pupil Diameter Min', 'Fixation Duration Std', 'Left Blink Opening Speed Max', 'Left Blink Closing Amplitude Max', 'Right Blink Opening Speed Max', 'Blinks Duration Max', 'Fixation Duration Max', 'Left Pupil Diameter Mean', 'Blinks Duration Mean', 'Right Blink Opening Speed Mean', 'Left Blink Opening Amplitude Max', 'Saccades Duration Max', 'Fixation Duration Median', 'Right Blink Opening Amplitude Mean', 'Left Pupil Diameter Max', 'Blinks Duration Min', 'Left Pupil Diameter Std', 'Right Pupil Diameter Max', 'Right Blink Opening Amplitude Max', 'Saccades Duration Median', 'Saccades Duration Std', 'Saccades Number', 'Fixation Duration Mean', 'Right Blink Closing Amplitude Max', 'Blinks Duration Std', 'Blinks Duration Median', 'Right Blink Closing Speed Mean', 'Left Pupil Diameter Median', 'Right Pupil Diameter Median']
elif MODEL == "KNN":
    if BINARY: #KNN, binary
        columns_order = ['Saccades Number', 'Left Pupil Diameter Max', 'Right Pupil Diameter Max', 'Blinks Duration Max', 'Saccades Duration Mean', 'Blinks Duration Median', 'Saccades Duration Std', 'Saccades Duration Median', 'Saccades Duration Max', 'Left Blink Closing Amplitude Std', 'Left Blink Opening Amplitude Std', 'Left Blink Opening Speed Std', 'Right Pupil Diameter Min', 'Left Blink Closing Speed Max', 'Left Blink Closing Speed Std', 'Left Blink Opening Speed Max', 'Left Blink Closing Amplitude Max', 'Right Blink Closing Speed Max', 'Left Blink Opening Amplitude Max', 'Right Blink Opening Amplitude Std', 'Fixation Duration Mean', 'Blinks Number', 'Fixation Duration Std', 'Left Pupil Diameter Mean', 'Fixation Duration Median', 'Fixation Duration Max', 'Blinks Duration Mean', 'Right Blink Opening Speed Std', 'Right Blink Opening Amplitude Mean', 'Right Blink Closing Amplitude Std', 'Left Blink Opening Amplitude Mean', 'Right Blink Opening Speed Mean', 'Right Blink Closing Speed Mean', 'Left Blink Closing Speed Mean', 'Left Pupil Diameter Min', 'Left Blink Closing Amplitude Mean', 'Right Pupil Diameter Mean', 'Right Blink Closing Speed Std', 'Right Blink Closing Amplitude Mean', 'Right Blink Closing Amplitude Max', 'Left Pupil Diameter Std', 'Right Blink Opening Speed Max', 'Left Pupil Diameter Median', 'Right Pupil Diameter Std', 'Blinks Duration Std', 'Blinks Duration Min', 'Left Blink Opening Speed Mean', 'Right Pupil Diameter Median', 'Right Blink Opening Amplitude Max']
    else: #KNN, 3 classes
        columns_order = ['Left Blink Opening Amplitude Std', 'Right Blink Opening Amplitude Std', 'Left Pupil Diameter Max', 'Right Pupil Diameter Mean', 'Right Blink Closing Amplitude Std', 'Right Blink Opening Amplitude Max', 'Left Blink Opening Speed Std', 'Left Blink Closing Amplitude Max', 'Right Blink Opening Amplitude Mean', 'Left Blink Closing Speed Std', 'Right Blink Closing Speed Std', 'Saccades Duration Median', 'Left Blink Opening Amplitude Max', 'Right Pupil Diameter Max', 'Fixation Duration Median', 'Left Blink Opening Speed Mean', 'Left Blink Closing Speed Mean', 'Left Blink Closing Amplitude Std', 'Right Blink Closing Speed Max', 'Right Blink Closing Amplitude Mean', 'Left Blink Closing Amplitude Mean', 'Left Blink Opening Speed Max', 'Saccades Duration Mean', 'Left Blink Opening Amplitude Mean', 'Left Blink Closing Speed Max', 'Right Blink Opening Speed Std', 'Fixation Duration Mean', 'Right Blink Opening Speed Mean', 'Right Pupil Diameter Std', 'Blinks Duration Mean', 'Left Pupil Diameter Mean', 'Right Blink Closing Speed Mean', 'Right Pupil Diameter Min', 'Blinks Number', 'Left Pupil Diameter Min', 'Right Blink Opening Speed Max', 'Saccades Number', 'Fixation Duration Max', 'Blinks Duration Median', 'Right Pupil Diameter Median', 'Blinks Duration Max', 'Saccades Duration Max', 'Blinks Duration Min', 'Fixation Duration Std', 'Left Pupil Diameter Std', 'Saccades Duration Std', 'Right Blink Closing Amplitude Max', 'Left Pupil Diameter Median', 'Blinks Duration Std']
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
        #for i in range(0,17):
        for i in range(0,11):
            
            col1 =  left[i]
            col2 = right[i]
            
            data_df[left_right_average[i]] = (data_df[col1] + data_df[col2])/2

    if LEFT_RIGHT_DIFF:
        #for i in range(0,17):
        for i in range(0,11):
            
            col1 =  left[i]
            col2 = right[i]
            
            #data_df[left_right_diff[i]] = abs((data_df[col1] - data_df[col2]))
            data_df[left_right_diff[i]] = (data_df[col1] - data_df[col2])

    if LEFT_RIGHT_DROP:
        #for i in range(0,17):
        for i in range(0,11):
            
            col1 =  left[i]
            col2 = right[i]
            
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
    