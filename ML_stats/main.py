import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings('ignore')

import time
import os
import numpy as np
import pandas as pd
import sys
import math

#from sklearn import preprocessing
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import balanced_accuracy_score, jaccard_score, average_precision_score
#from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin
from collections import Counter
from sklearn.model_selection import ShuffleSplit, StratifiedShuffleSplit
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import make_scorer
from sklearn.utils import resample
from sklearn.utils import shuffle
#from sklearn.utils.class_weight import compute_class_weight

from imblearn.ensemble import BalancedRandomForestClassifier, EasyEnsembleClassifier
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import BorderlineSMOTE
from imblearn.over_sampling import ADASYN
from imblearn.over_sampling import KMeansSMOTE
from imblearn.over_sampling import SVMSMOTE
from imblearn.under_sampling import TomekLinks
from imblearn.under_sampling import EditedNearestNeighbours
from imblearn.combine import SMOTEENN
from imblearn.combine import  SMOTETomek
from imblearn.pipeline import Pipeline

from get_model import get_model
from get_random_search_params import get_random_search_params
from get_grid_search_params import get_grid_search_params
from features import init_blinks_no_head, init_blinks_quantiles
from features import init, init_blinks, blinks
from features import left, right, left_right, left_right_average, left_right_diff

#columns_to_select = init
columns_to_select = init_blinks
#columns_to_select = init_blinks_no_head
#columns_to_select = left_right
#columns_to_select = blinks
#columns_to_select = init_blinks_quantiles

DATA_DIR = os.path.join("..", "..")
DATA_DIR = os.path.join(DATA_DIR, "Data")
ML_DIR = os.path.join(DATA_DIR, "MLInput")
#ML_DIR = os.path.join(DATA_DIR, "MLInput_Journal")
FIG_DIR = os.path.join(".", "Figures")
RANDOM_STATE = 0
CHS = True
BINARY = True

LEFT_RIGHT_AVERAGE = False
LEFT_RIGHT_DIFF = False
LEFT_RIGHT_DROP = False

#MODEL = "LR"
#MODEL = "DT"
#MODEL = "KNN"
#MODEL = "SVC"
MODEL = "HGBC"
#MODEL = "XGB"
#MODEL = "NC"
#MODEL = "RNC"
#MODEL = "RF"
#MODEL = "BRF"
#MODEL = "EEC"
#MODEL = "GBC" #slow
#MODEL = "ETC"
#MODEL = "MLPC"
#MODEL = "LDA"
#MODEL = "QDA"
#MODEL = "ABC"
#MODEL = "BC"
#MODEL = "VC"
#MODEL = "SC"
#MODEL = "LP"
#MODEL = "LS"

N_ITER = 100
N_SPLIT = 10
SCORING = 'f1_macro'
#SCORING = make_scorer(average_precision_score)
#minority = make_scorer(recall_score, pos_label=2, zero_division=0)
#minority = make_scorer(precision_score, pos_label=2, zero_division=0)
#minority = make_scorer(f1_score, pos_label=2, zero_division=0)
#minority = make_scorer(jaccard_score, pos_label=2, zero_division=0)
#SCORING = minority


#TIME_INTERVAL_DURATION = 300
TIME_INTERVAL_DURATION = 180
#TIME_INTERVAL_DURATION = 60
#TIME_INTERVAL_DURATION = 30
#TIME_INTERVAL_DURATION = 10

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

def calculate_classwise_accuracies(y_pred, y_true):
    
    print(f"y_pred: {y_pred}")
    print(f"y_true: {y_true}")
    
    y_pred_np = np.array(y_pred)
    y_true_np = np.array(y_true)
    
    # Get the unique classes
    if BINARY:
        classes = [1,2]
    else:
        classes = [1,2,3]

    # Find missing classes
    missing_classes = np.setdiff1d(classes, y_true_np)

    if missing_classes.size > 0:
        print("Missing classes:", missing_classes)
    else:
        print("All classes are present in y_true.")

    #print(classes)

    # Calculate accuracy for each class as a one-vs-all problem
    class_accuracies = []
    for cls in classes:
        # Get the indices of all instances where the class appears in y_true
        class_mask = (y_true_np == cls)
        
        class_accuracy = accuracy_score(y_true_np[class_mask], y_pred_np[class_mask])
        
        # Append the class accuracy
        class_accuracies.append(class_accuracy)

    #print(class_accuracies)
    
    return class_accuracies


# Function to perform parameter tuning with RandomizedSearchCV on each training fold
def model_with_tuning_stratified(pipeline, X_train, y_train):
    
    if (MODEL != "EEC") and (MODEL != "ABC") and (MODEL != "KNN") and (MODEL != "QDA"):
        pipeline.named_steps['classifier'].set_params(class_weight='balanced')
    
    params = get_random_search_params(MODEL)
    
    stratified_kfold = StratifiedKFold(n_splits=N_SPLIT, shuffle=True, random_state=RANDOM_STATE)
    
    random_search = RandomizedSearchCV(pipeline, params, n_iter=N_ITER, cv=stratified_kfold, scoring=SCORING, n_jobs=-1, random_state=RANDOM_STATE)
    
    # Fit on the training data for this fold
    random_search.fit(X_train, y_train)
    
    print(random_search.best_params_)
    
    return random_search.best_estimator_
    '''
    best_params = random_search.best_params_
    
    #print("Best Parameters:", search.best_params_)
    
    params = get_grid_search_params(MODEL, best_params)
    
    grid_search = GridSearchCV(estimator=pipeline,
                               param_grid=params,
                               scoring=SCORING,
                               cv=stratified_kfold,
                               n_jobs=-1
                               )
    
    grid_search.fit(X_train, y_train)
    #final_best_params = grid_search.best_params_
    
    # Return the best estimator found in this fold
    return grid_search.best_estimator_
    '''

# Stratified cross-validation function that handles the pipeline
def cross_val_stratified_with_label_transform(pipeline, X, y, cv):
    accuracies = []
    bal_accuracies = []
    c_w_accuracies = []
    precisions = []
    recalls = []
    f1_scores = []
    
    pipeline.named_steps['label_transform'].fit(X, y)  # Fit to compute thresholds
    _, y_transformed = pipeline.named_steps['label_transform'].transform(X, y)
    
    for i, (train_index, test_index) in enumerate(cv.split(X, y_transformed), start=1):
        
        print(f"Iteration {i}")
        
        X_train, X_test = np.array(X)[train_index], np.array(X)[test_index]
        y_train, y_test = np.array(y_transformed)[train_index], np.array(y_transformed)[test_index]
        
        #print(y_train)
        #print(y_test)
        
        # Set class weights to the classifier
        #pipeline.named_steps['classifier'].set_params(class_weight='balanced')
        
        #print("before model_with_tuning")
        # Get the best model after tuning on the current fold
        best_model = model_with_tuning_stratified(pipeline, X_train, y_train)
        #print("after model_with_tuning")
        
        # Fit the pipeline on transformed y_train
        best_model.fit(X_train, y_train)
        
        # Predict the labels on the transformed test data
        y_pred = best_model.predict(X_test)
        
        print(y_test)
        print(y_pred)
        
        # Calculate the metrics
        accuracies.append(accuracy_score(y_test, y_pred))
        bal_accuracies.append(balanced_accuracy_score(y_test, y_pred))
        c_w_accuracies.append(calculate_classwise_accuracies(y_pred=y_pred, y_true=y_test))
        precisions.append(precision_score(y_test, y_pred, average='macro', zero_division=0))
        recalls.append(recall_score(y_test, y_pred, average='macro', zero_division=0))
        f1_scores.append(f1_score(y_test, y_pred, average='macro', zero_division=0))
    
    
    print(f"Accuracy: {np.mean(accuracies):.2f} ± {np.std(accuracies):.2f}")
    print(f"Balanced accuracy: {np.mean(bal_accuracies):.2f} ± {np.std(bal_accuracies):.2f}")
    class1_values = [sublist[0] for sublist in c_w_accuracies if not math.isnan(sublist[0])]
    print(f"Class1 accuracy: {np.mean(class1_values):.2f} ± {np.std(class1_values):.2f}")
    class2_values = [sublist[1] for sublist in c_w_accuracies if not math.isnan(sublist[1])]
    print(f"Class2 accuracy: {np.mean(class2_values):.2f} ± {np.std(class2_values):.2f}")
    if not BINARY:
        class3_values = [sublist[2] for sublist in c_w_accuracies if not math.isnan(sublist[2])]
        print(f"Class3 accuracy: {np.mean(class3_values):.2f} ± {np.std(class3_values):.2f}")
    print(f"Precision: {np.mean(precisions):.2f} ± {np.std(precisions):.2f}")
    print(f"Recall: {np.mean(recalls):.2f} ± {np.std(recalls):.2f}")
    print(f"F1-Score: {np.mean(f1_scores):.2f} ± {np.std(f1_scores):.2f}")
    
    #print(bal_accuracies)
    #print(f1_scores)
    
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
    
    features = data_df.columns
    print(f"Number of features: {len(features)}")
    
    if LEFT_RIGHT_AVERAGE:
        for i in range(0,17):
            
            col1 =  left[i]
            col2 = right[i]
            
            data_df[left_right_average[i]] = (data_df[col1] + data_df[col2])/2
            #data_df = data_df.drop([col1, col2], axis=1)
    
    if LEFT_RIGHT_DIFF:
        for i in range(0,17):
            
            col1 =  left[i]
            col2 = right[i]
            
            data_df[left_right_diff[i]] = abs((data_df[col1] - data_df[col2]))
            #data_df = data_df.drop([col1, col2], axis=1)
            
    if LEFT_RIGHT_DROP:
        for i in range(0,17):
            
            col1 =  left[i]
            col2 = right[i]
            
            data_df = data_df.drop([col1, col2], axis=1)
    
    features = data_df.columns
    print(f"Number of features: {len(features)}")
        
    X = data_df.to_numpy()
    y = np.array(scores)
    
    zipped = list(zip(X, y))
    
    np.random.shuffle(zipped)
    
    X, y = zip(*zipped)
    
    X = np.array(X)
    y = np.array(y)
    
    pipeline = Pipeline([
                # Feature standartization step
                ('scaler', StandardScaler()),
                # Custom label transformation step
                ('label_transform', ThresholdLabelTransformer(get_percentiles())),
                # Oversampling step
                #('smote', SMOTE(random_state=RANDOM_STATE, k_neighbors=1)),
                #('smote', BorderlineSMOTE(random_state=RANDOM_STATE, k_neighbors=1)),
                #('smote', ADASYN(random_state=RANDOM_STATE, n_neighbors=1)),
                #('smote', SMOTETomek(smote=SMOTE(k_neighbors=1), random_state=RANDOM_STATE)),
                #('smote', SMOTEENN(smote=SMOTE(k_neighbors=1), random_state=RANDOM_STATE)),

                # Model setting step
                ('classifier', get_model(MODEL, RANDOM_STATE))
                ])

    # Initialize the cross-validation splitter
    #outer_cv = KFold(n_splits=N_SPLIT, shuffle=True, random_state=RANDOM_STATE)
    #cross_val_with_label_transform(pipeline, X, y, cv=outer_cv)
    
    outer_cv = StratifiedKFold(n_splits=N_SPLIT, shuffle=True, random_state=RANDOM_STATE)
    cross_val_stratified_with_label_transform(pipeline, X, y, cv=outer_cv)
    
    #hold_out_with_label_transform(pipeline, X, y)
        
    #hold_out_stratified_with_label_transform(pipeline, X, y)


start_time = time.time()

main()

elapsed_time = time.time() - start_time
print(f"Elapsed time: {elapsed_time:.3f} seconds")
    