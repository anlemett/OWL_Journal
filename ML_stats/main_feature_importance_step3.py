import warnings
warnings.filterwarnings('ignore')

import time
import os
import numpy as np
import pandas as pd
#import sys

import matplotlib.pyplot as plt
import csv
import gc

#from sklearn import preprocessing
from scipy.stats import randint, uniform
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin
#from collections import Counter
from sklearn.model_selection import RandomizedSearchCV, KFold, StratifiedKFold
from sklearn.model_selection import ShuffleSplit, StratifiedShuffleSplit
from permutation import RFEPermutationImportance

from features import init_blinks_no_head
from features import left, right, left_right_unite
from features import CHS_binary_importance

columns_to_select = init_blinks_no_head
columns_order = CHS_binary_importance

DATA_DIR = os.path.join("..", "..")
DATA_DIR = os.path.join(DATA_DIR, "Data")
ML_DIR = os.path.join(DATA_DIR, "MLInput")
#ML_DIR = os.path.join(DATA_DIR, "MLInput_Journal")
FIG_DIR = os.path.join(".", "Figures")

RANDOM_STATE = 0
PLOT = False

CHS = True
BINARY = True

LEFT_RIGHT_AVERAGE = True

MODEL = "SVC"
#MODEL = "RF"
#MODEL = "HGBC"

N_ITER = 100
N_SPLIT = 10
SCORING = 'f1_macro'
#SCORING = 'accuracy'

#TIME_INTERVAL_DURATION = 300
TIME_INTERVAL_DURATION = 180
#TIME_INTERVAL_DURATION = 60
#TIME_INTERVAL_DURATION = 30
#TIME_INTERVAL_DURATION = 10
#TIME_INTERVAL_DURATION = 1

def find_elbow_point(x, y):
    # Create a line between the first and last point
    line = np.array([x, y])
    point1 = line[:, 0]
    point2 = line[:, -1]

    # Calculate the distances
    #print("point1:", point1)
    #print("point2:", point2)
    #print("line:", line)
    distances = np.cross(point2-point1, point1-line.T)/np.linalg.norm(point2-point1)
    elbow_index = np.argmax(np.abs(distances))

    #return x[elbow_index]
    return elbow_index


features = CHS_binary_importance
##############
def main():
    
    bal_accuracies = [0.7891406663141195, 0.7799471179270228, 0.768253569539926, 0.7574603384452671, 0.7582667900581702, 0.7622990481226865, 0.76068614489688, 0.7614925965097832, 0.7498664727657325, 0.75068614489688, 0.7498796932839767, 0.7398532522474881, 0.7398532522474881, 0.7322858276044422, 0.7213987308302485, 0.7257271285034373, 0.7189661554732946, 0.7150145425700688, 0.7332403490216817, 0.7317080909571656, 0.7460364886303543, 0.7141274457958752, 0.7309148598625066, 0.7396113167636171, 0.7913048651507139, 0.7847726070861978, 0.7624338974087784, 0.718213907985193, 0.6773003701745108, 0.6818574828133264, 0.6277723426758329, 0.6496390798519303]
    f1_scores = [0.7496137012957835, 0.7343592743034374, 0.7368939239037041, 0.7250417811244523, 0.7254539366362632, 0.738033233311733, 0.7327955336863429, 0.7379124680364628, 0.7247285425116501, 0.7262728136920668, 0.7258670481881907, 0.7182683267975342, 0.7207365471956759, 0.7208043459388594, 0.7182174656271061, 0.7031395001324412, 0.7092006632349505, 0.6927326720129943, 0.7140144026177538, 0.6951089628587998, 0.6938052732951837, 0.6850626064220601, 0.7057435153016722, 0.6842223468630011, 0.7198322658414614, 0.7060474684701645, 0.6861393106365232, 0.6181430168723145, 0.5698906109879672, 0.5606391175307908, 0.5326128224066251, 0.5655706103217558]

    y = np.array(bal_accuracies[::-1])
    y_ = np.array(f1_scores[::-1])
    x = np.array(range(1, len(y)+1))
        
    elbow_point_idx = find_elbow_point(x, y)
    elbow_point_num = elbow_point_idx + 1
    elbow_point_acc = y[elbow_point_idx]
    corr_f1 = y_[elbow_point_idx]
    print(f"The elbow point for accuracy is at {elbow_point_num} features.")
    print(f"Accuracy at the elbow point: {elbow_point_acc}")
    print(f"Corresponding F1-score: {corr_f1}")
        
    elbow_point_f1_idx = find_elbow_point(x, y_)
    elbow_point_f1_num = elbow_point_f1_idx + 1
    elbow_point_f1 = y_[elbow_point_f1_idx]
    corr_acc = y[elbow_point_f1_idx]
    print(f"The elbow point for F1-score is at {elbow_point_f1_num} features.")
    print(f"F1-score at the elbow point: {elbow_point_f1}")
    print(f"Corresponding accuracy: {corr_acc}")
        
    filename = "chs"
    if BINARY:
        filename = filename + "_binary_"
    else:
        filename = filename + "_3classes_"
    filename = filename + MODEL
            
    # Plot accuracies
    fig0, ax0 = plt.subplots()
    ax0.plot(x, y, marker='o')
            
    ax0.set_xlabel('Number of Features', fontsize=14)
    ax0.set_ylabel('Accuracy', fontsize=14)
    ax0.tick_params(axis='both', which='major', labelsize=12)
    ax0.tick_params(axis='both', which='minor', labelsize=10)
    plt.grid(True)
    acc_filename = filename + "_bal_acc.png"
    full_filename = os.path.join(FIG_DIR, acc_filename)
    #plt.gca().set_aspect('equal', adjustable='box')
    plt.savefig(full_filename, dpi=600)
    plt.show()
            
    # Plot F1-scores
    fig2, ax2 = plt.subplots()
    ax2.plot(x, y_, marker='o')
    ax2.set_xlabel('Number of Features', fontsize=14)
    ax2.set_ylabel('F1-score', fontsize=14)
    ax2.tick_params(axis='both', which='major', labelsize=12)
    ax2.tick_params(axis='both', which='minor', labelsize=10)
    plt.grid(True)
    f1_filename = filename + "_f1.png"
    full_filename = os.path.join(FIG_DIR, f1_filename)
    plt.savefig(full_filename, dpi=600)
    plt.show()
        
    max_acc = max(bal_accuracies)
    max_index = bal_accuracies.index(max_acc)
    #print(f"Max_index: {max_index}")
    number_of_features = max_index + 1
    acc = bal_accuracies[max_index]
    f1 = f1_scores[max_index]
    print("Optimal number of features by maximizing Accuracy")
    print(f"Optimal number of features: {number_of_features}, Accuracy: {acc}, F1-score: {f1}")

    max_f1 = max(f1_scores)
    max_index = f1_scores.index(max_f1)
    #print(f"Max_index: {max_index}")
    number_of_features = max_index + 1
    acc = bal_accuracies[max_index]
    f1 = f1_scores[max_index]
    print("Optimal number of features by maximizing F1-score")
    print(f"Optimal number of features: {number_of_features}, Accuracy: {acc}, F1-score: {f1}, ")
    

##############        

start_time = time.time()

main()

elapsed_time = time.time() - start_time
print(f"Elapsed time: {elapsed_time:.3f} seconds")
    