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
from features import importance_CHS_binary_SVC, importance_CHS_binary_HGBC
from features import importance_CHS_3cl_SVC, importance_CHS_3cl_HGBC

columns_to_select = init_blinks_no_head
#columns_order = importance_CHS_binary_SVC
#columns_order = importance_CHS_binary_HGBC
columns_order = importance_CHS_3cl_SVC
#columns_order = importance_CHS_3cl_HGBC

DATA_DIR = os.path.join("..", "..")
DATA_DIR = os.path.join(DATA_DIR, "Data")
ML_DIR = os.path.join(DATA_DIR, "MLInput")
#ML_DIR = os.path.join(DATA_DIR, "MLInput_Journal")
FIG_DIR = os.path.join(".", "Figures")

RANDOM_STATE = 0
PLOT = False

CHS = True
BINARY = False

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
    distances = np.cross(point2-point1, point1-line.T)/np.linalg.norm(point2-point1)
    elbow_index = np.argmax(np.abs(distances))

    #return x[elbow_index]
    return elbow_index


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
    
    param_dist = get_param_dist()
    
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

    
# Cross-validation function that handles the pipeline and permutation importance
def cross_val_stratified_with_label_transform_and_permutation(pipeline, data_df, scores, cv, features_init):
    
    df = data_df
    
    X = df.to_numpy()
    y = np.array(scores)
    
    pipeline.named_steps['label_transform'].fit(X, y)  # Fit to compute thresholds
    _, y_transformed = pipeline.named_steps['label_transform'].transform(X, y)

    # Set class weights to the classifier
    #pipeline.named_steps['classifier'].set_params(class_weight='balanced')
    
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
      df = data_df[features]
      del X
      gc.collect()
      X = df.to_numpy()

    print(len(f1_scores_num_features))
    print(bal_accuracies_num_features)
    print(f1_scores_num_features)
  
      
##############
# Hold-out function that handles the pipeline and permutation importance
def hold_out_with_label_transform_and_permutation(pipeline, X, y, features):
    
    pipeline.named_steps['label_transform'].fit(X, y)  # Fit to compute thresholds
    _, y_transformed = pipeline.named_steps['label_transform'].transform(X, y)
    
    # Spit the data into train and test
    #ss = ShuffleSplit(n_splits=1, test_size=.1, random_state=RANDOM_STATE)
    sss = StratifiedShuffleSplit(n_splits=1, test_size=.1, random_state=RANDOM_STATE)
    
    #for i, (train_idx, test_idx) in enumerate(sss.split(X)):
    for i, (train_idx, test_idx) in enumerate(sss.split(X,y_transformed)):
        
        X_train = np.array(X)[train_idx.astype(int)]
        y_train = np.array(y_transformed)[train_idx.astype(int)]
        X_test = np.array(X)[test_idx.astype(int)]
        y_test = np.array(y_transformed)[test_idx.astype(int)]
        
        #pipeline.named_steps['label_transform'].fit(X_train, y_train)  # Fit to compute thresholds
        #_, y_train_transformed = pipeline.named_steps['label_transform'].transform(X_train, y_train)
        #_, y_test_transformed = pipeline.named_steps['label_transform'].transform(X_test, y_test)
        
        # Set class weights to the classifier
        pipeline.named_steps['classifier'].set_params(class_weight='balanced')

        # Get the best model after tuning on the current fold
        best_model = model_with_tuning_stratified(pipeline, X_train, y_train)

        #######
        #X_df = pd.DataFrame(X, columns=features)
        
        test_accuracies = []
        acc_num_features_list = []
        test_f1_scores = []
        f1_num_features_list = []
        
        # Perform RFE with Permutation Importance
        rfe = RFEPermutationImportance(best_model, min_features_to_select=1,
                                       n_repeats=5)

        X_train = pd.DataFrame(X_train, columns=features)
        X_test = pd.DataFrame(X_test, columns=features)
        
        rfe.fit(X_train, y_train, X_test, y_test, features)
            
        # Store accuracies for plotting
        for num_features, accuracy in rfe.accuracies_:
            acc_num_features_list.append(num_features)
            test_accuracies.append(accuracy)
    
        # Store f1 for plotting
        for num_features, train_f1_score in rfe.f1_scores_:
            f1_num_features_list.append(num_features)
            test_f1_scores.append(train_f1_score)
    
        # Select the remaining features
        selected_features = list(X_train.columns[rfe.support_])
               
        X_train_selected = X_train[selected_features]

        # Final model training with selected features
        best_model.fit(X_train_selected, y_train)
        
        ############################## Predict ################################
    
        X_test_selected = X_test[selected_features]
        y_pred = best_model.predict(X_test_selected)
    
        ############################ Evaluate #################################
        
        print(selected_features)
    
        accuracy = accuracy_score(y_pred=y_pred, y_true=y_test)
        
        f1_macro = f1_score(y_pred=y_pred, y_true=y_test, average='macro')
    
        print("Accuracy:", accuracy)
        print("Macro F1-score:", f1_macro)
    
        test_accuracies.reverse()
        test_f1_scores.reverse()
        acc_num_features_list.reverse()
        f1_num_features_list.reverse()
        print(test_accuracies)
        print(test_f1_scores)
        
        y = np.array(test_accuracies)
        y_ = np.array(test_f1_scores)
        x = np.array(acc_num_features_list)
        
        filename = 'curve_coordinates_acc.csv'
        with open(filename, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['x', 'y'])  # Write the header
            for i in range(len(x)):
                writer.writerow([x[i], y[i]])
        
        filename = 'curve_coordinates_f1.csv'
        with open(filename, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['x', 'y'])  # Write the header
            for i in range(len(x)):
                writer.writerow([x[i], y_[i]])
        
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
        
        
        if PLOT:
            if CHS:
                filename = SCORING + "_scoring_chs"
            else:
                filename = SCORING + "_scoring_eeg" + str(TIME_INTERVAL_DURATION)
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
            acc_filename = filename + "_acc.png"
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
        
        max_acc = max(test_accuracies)
        max_index = test_accuracies.index(max_acc)
        #print(f"Max_index: {max_index}")
        number_of_features = max_index + 1
        acc = test_accuracies[max_index]
        f1 = test_f1_scores[max_index]
        print("Optimal number of features by maximizing Accuracy")
        print(f"Optimal number of features: {number_of_features}, Accuracy: {acc}, F1-score: {f1}")

        max_f1 = max(test_f1_scores)
        max_index = test_f1_scores.index(max_f1)
        #print(f"Max_index: {max_index}")
        number_of_features = max_index + 1
        acc = test_accuracies[max_index]
        f1 = test_f1_scores[max_index]
        print("Optimal number of features by maximizing F1-score")
        print(f"Optimal number of features: {number_of_features}, Accuracy: {acc}, F1-score: {f1}, ")
    

##############        
   
def get_model():
    print(f"Model: {MODEL}")
    if MODEL == "SVC":
        return SVC(cache_size=10)
    elif MODEL == "RF":
        return RandomForestClassifier(random_state=RANDOM_STATE, max_features=None)
    else:
        return HistGradientBoostingClassifier(random_state=RANDOM_STATE)

def get_param_dist():
    if MODEL == "SVC":
        param_dist = {
            'classifier': [get_model()],
            'classifier__class_weight': ['balanced'],
            'classifier__C': uniform(loc=0, scale=10),  # Regularization parameter
            'classifier__kernel': ['linear', 'rbf', 'sigmoid'],  # Kernel type
            'classifier__gamma': ['scale', 'auto'],  # Kernel coefficient
            'classifier__degree': randint(1, 10)  # Degree of polynomial kernel
            }
    elif  MODEL == "RF":
       param_dist = {
            'classifier': [get_model()],
            'classifier__class_weight': ['balanced'],
            'classifier__n_estimators': randint(50,500),
            'classifier__max_depth': randint(1,79),
             #'min_samples_split': randint(2, 40),
             #'min_samples_leaf': randint(1, 40),
             #'max_features': ['auto', 'sqrt', 'log2', None],
             #'criterion': ['gini', 'entropy', 'log_loss']
            }
    else:
        param_dist = {
             'classifier': [get_model()],
             'classifier__class_weight': ['balanced'],
             'classifier__max_depth': randint(1,79),
             }
    return param_dist

def get_percentiles():
    if BINARY:
        print(f"BINARY")
        return [0.93]
    else:
        print(f"3 classes")
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
    
    if LEFT_RIGHT_AVERAGE:
        for i in range(0,17):
            
            col1 =  left[i]
            col2 = right[i]
            
            data_df[left_right_unite[i]] = (data_df[col1] + data_df[col2])/2
            data_df = data_df.drop([col1, col2], axis=1)

    data_df = data_df[columns_order]    
    features = data_df.columns
    
    print(f"Number of features: {len(features)}")
            
    pipeline = Pipeline([
            # Step 1: Standardize features
            ('scaler', StandardScaler()),
            # Step 2: Apply custom label transformation
            ('label_transform', ThresholdLabelTransformer(get_percentiles())),
            # Step 3: Choose the model
            #('classifier', get_model())
            ('classifier', None)
            ])
    
    
    # Initialize the cross-validation splitter
    outer_cv = StratifiedKFold(n_splits=N_SPLIT, shuffle=True, random_state=RANDOM_STATE)
    cross_val_stratified_with_label_transform_and_permutation(pipeline, data_df, scores, outer_cv, features)
    
    #hold_out_with_label_transform_and_permutation(pipeline, X, y, features)
        
    print(f"Number of slots: {len(data_df.index)}")


start_time = time.time()

main()

elapsed_time = time.time() - start_time
print(f"Elapsed time: {elapsed_time:.3f} seconds")
    