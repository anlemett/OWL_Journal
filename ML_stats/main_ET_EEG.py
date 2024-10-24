import warnings
warnings.filterwarnings('ignore')

import time
import os
import numpy as np
import pandas as pd
#import sys

from sklearn.model_selection import ShuffleSplit, RandomizedSearchCV
from sklearn import preprocessing
from scipy.stats import randint, uniform
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score

from features import init, init_blinks, init_quantiles, init_blinks_quantiles
from features import init_blinks_without_diam, saccade_fixation_blinks
from features import saccade_fixation_blinks_head, init_blinks_without_head, blinks
from features import blinks_quantiles, blinks_left, blinks_right, left, right
from features import init_blinks_mean, init_blinks_median, init_blinks_sd
from features import init_blinks_min, init_blinks_max, init_blinks_mean_median
from features import init_blinks_number

# TODO: without quantiles

# 3 classes, HGBC, 60 seconds
#columns_to_select = init # acc=0.84, f1=0.77 #58
columns_to_select = init_blinks # acc=0.87, f1=0.82 #64 ! init
                                # 
                                # 
                       # with av.bl.amp&speed + diam, drop orig.: acc=0.84, f1=0.8 #47
                       # with abs(dif), drop orig.: acc=0.79, f1=0.72 #47
                       # with abs(dif), keep orig.: acc=0.85, f1=0.78 #81
#columns_to_select = init_blinks_number # acc=0.48, f1=0.4 #1
#columns_to_select = init_blinks_mean # acc=0.81, f1=0.72 #18
#columns_to_select = init_blinks_median # acc=0.83, f1=0.77 #10
#columns_to_select = init_blinks_sd # acc=0.74, f1=0.64 #18
#columns_to_select = init_blinks_min # acc=0.64, f1=0.52 #8
#columns_to_select = init_blinks_max # acc=0.79, f1=0.65 #18
#columns_to_select = init_blinks_mean_median # acc=0.83, f1=0.71 #26
#columns_to_select = init_quantiles # acc=0.87, f1=0.79 #89
#columns_to_select = init_blinks_quantiles # acc=0.88, f1=0.81. #96
#columns_to_select = init_blinks_without_diam # acc=0.83, f1=0.76 #54
#columns_to_select = saccade_fixation_blinks # acc=0.67, f1=0.47 #15
#columns_to_select = saccade_fixation_blinks_head # acc=0.78, f1=0.73 #30
#columns_to_select = init_blinks_without_head # acc=0.81, f1=0.75 #49
#columns_to_select = blinks # acc=0.76, f1=0.66 #30
                           # with dif: acc=0.77, f1=0.66 #42
                           # with abs(dif): acc=0.78, f1=0.65 #42
                           # with drop the original: acc=0.59, f1=0.47 #18
                           # with av. and drop: acc=76, f1=0.68 #18
#columns_to_select = blinks_quantiles # acc=0.78, f1=0.69 #
  

# binary, RF
#columns_to_select = init # acc=0.95, f1=0.81 #58
#columns_to_select = init_blinks # acc=0.97, f1=0.87 #64
#columns_to_select = init_blinks_number # acc=0.74, f1=0.48 #1
#columns_to_select = init_quantiles # acc=0.95, f1=0.81 #89
#columns_to_select = init_blinks_quantiles # acc=0.97, f1=0.85 #96
#columns_to_select = init_blinks_without_diam # acc=0., f1=0.
#columns_to_select = saccade_fixation_blinks # acc=0., f1=0.


DATA_DIR = os.path.join("..", "..")
DATA_DIR = os.path.join(DATA_DIR, "Data")
ML_DIR = os.path.join(DATA_DIR, "MLInput")
#ML_DIR = os.path.join(DATA_DIR, "MLInput_Journal")
FIG_DIR = os.path.join(".", "Figures")

RANDOM_STATE = 0

BINARY = False
EQUAL_PERCENTILES = False

#MODEL = "LR"
#MODEL = "SVC"
#MODEL = "DT"
#MODEL = "RF"
MODEL = "HGBC"

N_ITER = 100
CV = 5
SCORING = 'f1_macro'
#SCORING = 'accuracy'

TIME_INTERVAL_DURATION = 60 
#TIME_INTERVAL_DURATION = 1 # acc=, f1= for init_blink

np.random.seed(RANDOM_STATE)

def weight_classes(scores):
    
    vals_dict = {}
    for i in scores:
        if i in vals_dict.keys():
            vals_dict[i] += 1
        else:
            vals_dict[i] = 1
    total = sum(vals_dict.values())

    # Formula used:
    # weight = 1 - (no. of samples present / total no. of samples)
    # So more the samples, lower the weight

    weight_dict = {k: (1 - (v / total)) for k, v in vals_dict.items()}
    #print(weight_dict)
        
    return weight_dict

def getEEGThreshold(scores):
    #Split into 2 bins by percentile
    eeg_series = pd.Series(scores)
    if EQUAL_PERCENTILES:
        th = eeg_series.quantile(.5)
    else:
        th = eeg_series.quantile(.93)
    return th

def getEEGThresholds(scores):
    #Split into 3 bins by percentile
    eeg_series = pd.Series(scores)
    if EQUAL_PERCENTILES:
        (th1, th2) = eeg_series.quantile([.33, .66])
    else:
        (th1, th2) = eeg_series.quantile([.52, .93])
    return (th1, th2)


def main():
    
    if TIME_INTERVAL_DURATION == 60:
        filename = "ML_features_1min.csv"
    else:
        filename = "ML_features_1sec.csv"
    
    full_filename = os.path.join(ML_DIR, filename)
    
    data_df = pd.read_csv(full_filename, sep=' ')
    data_df = data_df.drop('ATCO', axis=1)
    
    data_df = data_df[columns_to_select]
    
    '''    
    for i in range(0,17):
        
        col1 =  left[i]
        col2 = right[i]
        
        data_df[str(i)] = abs(data_df[col1] - data_df[col2])
        #data_df[str(i)] = (data_df[col1] + data_df[col2])/2
        #data_df = data_df.drop([col1, col2], axis=1)
    '''
    
    print(len(data_df.columns))
    
    '''
    head_features = [
        'Head Heading Mean', 'Head Pitch Mean', 'Head Roll Mean',
        'Head Heading Std', 'Head Pitch Std', 'Head Roll Std',
        'Head Heading Median', 'Head Pitch Median', 'Head Roll Median',
        'Head Heading Min', 'Head Pitch Min', 'Head Roll Min',
        'Head Heading Max', 'Head Pitch Max', 'Head Roll Max']
    
    for feature in head_features:
        data_df = data_df.drop(columns=[feature])
    
    print(data_df.columns)
    '''
    
    full_filename = os.path.join(ML_DIR, "ML_ET_EEG_" + str(TIME_INTERVAL_DURATION) + "__EEG.csv")

    scores_np = np.loadtxt(full_filename, delimiter=" ")
    
    
    features_np = data_df.to_numpy()
    

    ###########################################################################
    #Shuffle data

    #print(features_np.shape)
    #print(scores_np.shape)
    
    scores_np = scores_np[0,:] # Workload

    zipped = list(zip(features_np, scores_np))

    np.random.shuffle(zipped)

    features_np, scores_np = zip(*zipped)

    scores = list(scores_np)
    
    #print(scores)
    
    #print(type(features_np))
    features_np = np.array(features_np)
    
    # Spit the data into train and test
    rs = ShuffleSplit(n_splits=1, test_size=.1, random_state=RANDOM_STATE)
    
    for i, (train_idx, test_idx) in enumerate(rs.split(features_np)):
        X_train = np.array(features_np)[train_idx.astype(int)]
        y_train = np.array(scores)[train_idx.astype(int)]
        X_test = np.array(features_np)[test_idx.astype(int)]
        y_test = np.array(scores)[test_idx.astype(int)]
    
    #normalize train set
    scaler = preprocessing.MinMaxScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
       
    if  BINARY:
        th = getEEGThreshold(y_train)
        y_train = [1 if score < th else 2 for score in y_train]
    else:
        (th1, th2) = getEEGThresholds(y_train)
        y_train = [1 if score < th1 else 3 if score > th2 else 2 for score in y_train]
    
    print("EEG")
    number_of_classes = len(set(y_train))
    print(f"Number of classes : {number_of_classes}")
    
    weight_dict = weight_classes(y_train)
    
    #normalize test set
    X_test = scaler.transform(X_test)
    
    if  BINARY:
        y_test = [1 if score < th else 2 for score in y_test]
    else:
        y_test = [1 if score < th1 else 3 if score > th2 else 2 for score in y_test]
        
    ################################# Fit #####################################
    
    print(f"Model: {MODEL}")
    print(f"Scoring: {SCORING}, n_iter: {N_ITER}, cv: {CV}")

    if MODEL == "LR":
        
        clf = LogisticRegression(class_weight=weight_dict, solver='liblinear')
        
        param_dist = {
            'C': uniform(loc=0, scale=4),  # Regularization parameter
            'penalty': ['l1', 'l2'],       # Penalty norm
             }
        
        search = RandomizedSearchCV(clf, 
                                param_distributions = param_dist,
                                scoring = SCORING,
                                n_iter=N_ITER,
                                cv=CV,
                                n_jobs=-1,
                                random_state=RANDOM_STATE)
        
        # Fit the search object to the data
        search.fit(X_train, y_train)
        
        # Create a variable for the best model
        best_clf = search.best_estimator_

        # Print the best hyperparameters
        print('Best hyperparameters:',  search.best_params_)
        
    elif MODEL == "SVC":
        
        clf = SVC(class_weight=weight_dict)
        
        param_dist = {
            'C': uniform(loc=0, scale=10),  # Regularization parameter
            'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],  # Kernel type
            'gamma': ['scale', 'auto'],  # Kernel coefficient
            'degree': randint(1, 10)  # Degree of polynomial kernel
            }
        
        search = RandomizedSearchCV(clf, 
                                param_distributions = param_dist,
                                scoring = SCORING,
                                n_iter=N_ITER,
                                cv=CV,
                                n_jobs=-1,
                                random_state=RANDOM_STATE)
        
        # Fit the search object to the data
        search.fit(X_train, y_train)
        
        # Create a variable for the best model
        best_clf = search.best_estimator_

        # Print the best hyperparameters
        print('Best hyperparameters:',  search.best_params_)
        
    elif  MODEL == "DT":
        
        clf = DecisionTreeClassifier(class_weight=weight_dict)

        # Use random search to find the best hyperparameters
        param_dist = {
             'max_depth': randint(1,79),
             }
        
        search = RandomizedSearchCV(clf, 
                                param_distributions = param_dist,
                                scoring = SCORING,
                                n_iter=N_ITER,
                                cv=CV,
                                n_jobs=-1,
                                random_state=RANDOM_STATE)
        
        # Fit the search object to the data
        search.fit(X_train, y_train)
        
        # Create a variable for the best model
        best_clf = search.best_estimator_

        # Print the best hyperparameters
        print('Best hyperparameters:',  search.best_params_)
        
    elif  MODEL == "RF": # 0.96, 0.82; 
        clf = RandomForestClassifier(class_weight=weight_dict,
                                     #bootstrap=True,
                                     max_features=None,
                                     #criterion='entropy',
                                     random_state=RANDOM_STATE)
        
        # Use random search to find the best hyperparameters
        
        param_dist = {
             'n_estimators': randint(50,500),
             'max_depth': randint(1,79),
              #'min_samples_split': randint(2, 40),
              #'min_samples_leaf': randint(1, 40),
              #'max_features': ['auto', 'sqrt', 'log2', None],
              #'criterion': ['gini', 'entropy', 'log_loss']
             }
        
        search = RandomizedSearchCV(clf, 
                                param_distributions = param_dist,
                                scoring = SCORING,
                                n_iter=N_ITER, 
                                cv=CV,
                                n_jobs=-1,
                                random_state=RANDOM_STATE)
        
        # Fit the search object to the data
        search.fit(X_train, y_train)
 
        # Create a variable for the best model
        best_clf = search.best_estimator_

        # Print the best hyperparameters
        print('Best hyperparameters:',  search.best_params_)
        
    elif  MODEL == "HGBC": # 0.95, 0.83; 
        clf = HistGradientBoostingClassifier(class_weight='balanced',
                                             random_state=RANDOM_STATE)

        # Use random search to find the best hyperparameters
        param_dist = {
             'max_depth': randint(1,79),
             }
        
        search = RandomizedSearchCV(clf, 
                                param_distributions = param_dist,
                                scoring = SCORING,
                                n_iter=N_ITER,
                                cv=CV,
                                n_jobs=-1,
                                random_state=RANDOM_STATE)
        
        # Fit the search object to the data
        search.fit(X_train, y_train)
        
        # Create a variable for the best model
        best_clf = search.best_estimator_

        # Print the best hyperparameters
        print('Best hyperparameters:',  search.best_params_)
    
    
    ############################## Predict ####################################
    
    y_pred = best_clf.predict(X_test)

    print("Shape at output after classification:", y_pred.shape)

    ############################ Evaluate #####################################
    
    accuracy = accuracy_score(y_pred=y_pred, y_true=y_test)
    
    f1_macro = f1_score(y_pred=y_pred, y_true=y_test, average='macro')
    
    print("Accuracy:", accuracy)
    print("Macro F1-score:", f1_macro)
    
    
start_time = time.time()

main()

elapsed_time = time.time() - start_time
print(f"Elapsed time: {elapsed_time:.3f} seconds")
    