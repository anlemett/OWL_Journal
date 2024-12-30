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
from scipy.stats import randint, uniform
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestCentroid
from sklearn.neighbors import RadiusNeighborsClassifier
from sklearn.semi_supervised import LabelPropagation
from sklearn.semi_supervised import LabelSpreading
#from lightgbm import LGBMClassifier

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

from features import init_blinks_no_head, init_blinks_quantiles
from features import init, init_blinks, blinks
from features import left, right, left_right, left_right_unite

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

#MODEL = "KNN"
#MODEL = "NC"
#MODEL = "RNC"
#MODEL = "SVC"
#MODEL = "RF"
#MODEL = "BRF"
#MODEL = "EEC"
MODEL = "HGBC"
#MODEL = "GBC" #slow
#MODEL = "ETC"
#MODEL = "MLPC"
#MODEL = "LDA"
#MODEL = "QDA" #slow
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
    
    #if MODEL != "EEC":
    #    pipeline.named_steps['classifier'].set_params(class_weight='balanced')
    
    params = get_random_search_params()
    
    stratified_kfold = StratifiedKFold(n_splits=N_SPLIT, shuffle=True, random_state=RANDOM_STATE)
    
    random_search = RandomizedSearchCV(pipeline, params, n_iter=N_ITER, cv=stratified_kfold, scoring=SCORING, n_jobs=-1, random_state=RANDOM_STATE)
    
    # Fit on the training data for this fold
    random_search.fit(X_train, y_train)
    
    best_params = random_search.best_params_
    
    #print("Best Parameters:", search.best_params_)
    
    params = get_grid_search_params(best_params)
    
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
    
    print(bal_accuracies)
    print(f1_scores)


def get_model():
    print(f"Model: {MODEL}")
    if  MODEL == "KNN":
        return KNeighborsClassifier()
    elif MODEL == "NC":
        return NearestCentroid()
    elif MODEL == "RNC":
        return RadiusNeighborsClassifier(outlier_label='most_frequent')
    elif MODEL == "ETC":
        return ExtraTreesClassifier(random_state=RANDOM_STATE)
    #elif MODEL == "LGBM":
    #    return LGBMClassifier(random_state=RANDOM_STATE, verbose=-1)
    elif MODEL == "SVC":
        return SVC()
    elif MODEL == "MLPC":
        return MLPClassifier(max_iter=1000, random_state=RANDOM_STATE)
    elif MODEL == "LDA":
        return LinearDiscriminantAnalysis()
    elif MODEL == "QDA":
        return QuadraticDiscriminantAnalysis()
    elif MODEL == "RF":
        return RandomForestClassifier(random_state=RANDOM_STATE, max_features=None)
    elif MODEL == "BRF":
        return BalancedRandomForestClassifier(
                    bootstrap=False,
                    replacement=True,
                    sampling_strategy='auto',  # Under-sample majority class to match minority
                    random_state=RANDOM_STATE
                    )
    elif MODEL == "EEC":
        return EasyEnsembleClassifier(random_state=RANDOM_STATE)
    elif MODEL == "ABC":
        base_estimator = DecisionTreeClassifier(random_state=RANDOM_STATE)
        adaboost = AdaBoostClassifier(estimator=base_estimator, random_state=RANDOM_STATE)
        return adaboost
    elif MODEL == "BC":
        base_estimator = DecisionTreeClassifier(random_state=RANDOM_STATE)
        bagging = BaggingClassifier(estimator=base_estimator, random_state=RANDOM_STATE)
        return bagging
    elif MODEL == "VC":
        clf1 = LogisticRegression(random_state=RANDOM_STATE, max_iter=500)
        clf2 = RandomForestClassifier(random_state=RANDOM_STATE)
        clf3 = SVC(probability=True, random_state=RANDOM_STATE)
        
        # Initialize VotingClassifier
        voting_clf = VotingClassifier(
            estimators=[('lr', clf1), ('rf', clf2), ('svc', clf3)],
            voting='soft'
            )
        return voting_clf
    elif MODEL == "SC":
        base_estimators = [
            ('svc', SVC(probability=True, random_state=42)),
            ('knn', KNeighborsClassifier()),
            ('hgb', HistGradientBoostingClassifier(random_state=42))
            ]

        # Define meta-classifier
        meta_clf = LogisticRegression(random_state=RANDOM_STATE, max_iter=500)

        # Initialize StackingClassifier
        stacking_clf = StackingClassifier(
            estimators=base_estimators,
            final_estimator=meta_clf
            )
        return stacking_clf
    elif MODEL == "LP":
        return LabelPropagation()
    elif MODEL == "LS":
        return LabelSpreading()
    elif MODEL == "GBC":
        return GradientBoostingClassifier(random_state=RANDOM_STATE)
    else:
        return HistGradientBoostingClassifier(random_state=RANDOM_STATE)

def get_random_search_params():
    if MODEL == "KNN":
        params = {
            'classifier__n_neighbors': randint(3, 15),  # Randomly select n_neighbors between 3 and 15
            'classifier__weights': ['uniform', 'distance'],  # Weight function used in prediction
            'classifier__algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],  # Algorithm for nearest neighbor search
            'classifier__p': [1, 2],  # Power parameter for the Minkowski distance (1=Manhattan, 2=Euclidean)
            }
    elif MODEL == "NC":
        params = {
            'classifier__metric': ['euclidean', 'manhattan', 'minkowski'],  # Distance metrics
            'classifier__shrink_threshold': uniform(0, 1),  # Shrinkage threshold (0 to 1)
            }
    elif MODEL == "RNC":
        params = {
            'classifier__radius': uniform(0.1, 2.0),  # Randomly sample radius between 0.1 and 1.0
            'classifier__weights': ['uniform', 'distance'],  # Weight function used in prediction
            'classifier__algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],  # Algorithm for nearest neighbor search
            'classifier__leaf_size': [10, 20, 30, 40, 50],  # Size of the leaf for the tree-based algorithms
            }
    elif MODEL == "ETC":
        params = {
            'classifier__n_estimators': randint(50, 300),  # Randomly search between 50 and 300 trees
            'classifier__max_features': ['auto', 'sqrt', 'log2', None],  # Try different options for max features
            'classifier__max_depth': randint(5, 50),  # Random search between depth of 5 and 50
            'classifier__min_samples_split': randint(2, 20),  # Random search for min samples for splitting
            'classifier__min_samples_leaf': randint(1, 20),  # Random search for min samples for leaf nodes
            }
    elif MODEL == "LGBM":
            params = {
                'classifier__num_leaves': np.arange(20, 150, 10),  # Range of number of leaves
                'classifier__max_depth': np.arange(3, 15),         # Range of max depth
                'classifier__learning_rate': [0.01, 0.05, 0.1, 0.2, 0.3],  # Learning rates to try
                'classifier__n_estimators': [100, 200, 500],      # Number of trees
                'classifier__min_data_in_leaf': [10, 20, 30],     # Minimum samples per leaf
                'classifier__lambda_l1': [0, 0.1, 1, 10],         # L1 regularization strength
                'classifier__lambda_l2': [0, 0.1, 1, 10],         # L2 regularization strength
                'classifier__feature_fraction': [0.7, 0.8, 0.9, 1.0],  # Feature sampling ratio
                'classifier__bagging_fraction': [0.7, 0.8, 0.9, 1.0],  # Row sampling ratio
                'classifier__bagging_freq': [0, 5, 10],           # Frequency of row sampling
                'classifier__subsample_for_bin': [200000, 300000, 400000],  # Number of samples used to construct histograms
                }
    elif MODEL == "SVC":
            params = {
                #'classifier__C': uniform(loc=0, scale=1000),  # Regularization parameter
                #'classifier__kernel': ['sigmoid', 'linear', 'rbf'],  # Kernel type
                #'classifier__gamma': ['scale','auto', 0.001, 0.1],  # Kernel coefficient
                'classifier__C': [0.001, 0.01, 1, 5, 10, 100],
                'classifier__kernel': ['linear', 'rbf'],
                'classifier__gamma': [0.001, 0.01, 0.1, 1, 10, 100, 'scale', 'auto'],
                }
    elif MODEL == "MLPC":
        params = {
            'classifier__hidden_layer_sizes': [(50,), (100,), (150,), (200,)],  # Vary the number of neurons in hidden layers
            'classifier__activation': ['relu', 'tanh', 'logistic'],  # Different activation functions
            'classifier__solver': ['adam', 'sgd'],  # Solvers to use
            'classifier__alpha': uniform(0.0001, 0.1),  # Regularization strength
            'classifier__learning_rate': ['constant', 'invscaling', 'adaptive'],  # Learning rate schedules
            'classifier__learning_rate_init': uniform(0.0001, 0.1),  # Initial learning rate
            }
    elif MODEL == "LDA":
        params = {
            'classifier__solver': ['svd', 'lsqr', 'eigen'],  # Solvers for LDA
            'classifier__shrinkage': uniform(0.0, 1.0)       # Shrinkage parameter for 'lsqr' and 'eigen' solvers
            }
    elif MODEL == "QDA":
        params = {
            'classifier__reg_param': uniform(0, 1),  # Regularization parameter (0 to 1)
            'classifier__tol': uniform(1e-5, 1e-3),  # Tolerance for stopping criterion
            }
    elif  (MODEL == "RF") or (MODEL == "BRF"):
            params = {
                'classifier__n_estimators': randint(50,500),
                'classifier__max_depth': randint(1,79),
                'classifier__max_features': ['sqrt', 'log2', None],
                'classifier__min_samples_leaf': randint(1, 40),
                'classifier__min_samples_split': randint(2, 40)
                #'classifier__n_estimators': [50, 100, 200, 300],
                #'classifier__max_depth': [None, 10, 20, 30, 40],
                #'classifier__max_features': ['sqrt', 'log2', None],
                #'classifier__min_samples_leaf': [1, 2, 4],
                #'classifier__min_samples_split': [2, 5, 10]
                }
    elif MODEL == "ABC":
            params = {
                'classifier__n_estimators': randint(50, 200),  # Number of weak classifiers
                'classifier__learning_rate': uniform(0.01, 1.0),  # Shrinks the contribution of each classifier
                'classifier__estimator__max_depth': randint(1, 10),  # Depth of trees in base estimator
                'classifier__estimator__min_samples_split': randint(2, 10),  # Minimum samples to split an internal node
                }
    elif MODEL == "BC":
        params = {
            'classifier__n_estimators': randint(10, 100),  # Number of base estimators
            'classifier__max_samples': uniform(0.5, 0.5),  # Fraction of samples for training each base estimator
            'classifier__max_features': uniform(0.5, 0.5),  # Fraction of features for training each base estimator
            'classifier__estimator__max_depth': randint(1, 10),  # Depth of trees in base estimator
            'classifier__estimator__min_samples_split': randint(2, 10),  # Minimum samples required to split
            }
    elif MODEL == "GBC":
        params = {
            'classifier__n_estimators': [50, 100, 200, 300, 400],  # Number of boosting stages
            'classifier__learning_rate': uniform(0.01, 0.2),  # Step size shrinking to prevent overfitting (uniform distribution)
            'classifier__max_depth': [3, 5, 7, 9, 10],  # Maximum depth of the individual trees
            'classifier__min_samples_split': [2, 5, 10, 20],  # Minimum number of samples required to split an internal node
            'classifier__min_samples_leaf': [1, 2, 4, 8]  # Minimum number of samples required to be at a leaf node
            }
    elif MODEL == "VC":
        params = {
            'classifier__lr__C': uniform(0.01, 10),  # Logistic Regression: Regularization parameter
            'classifier__rf__n_estimators': randint(50, 200),  # Random Forest: Number of trees
            'classifier__rf__max_depth': randint(3, 20),  # Random Forest: Maximum tree depth
            'classifier__svc__C': uniform(0.01, 10),  # SVC: Regularization parameter
            }
    elif MODEL == "SC":
        params = {
            'classifier__svc__C': uniform(0.01, 10),  # SVC: Regularization parameter
            'classifier__svc__gamma': uniform(0.01, 1),  # SVC: Kernel coefficient
            'classifier__knn__n_neighbors': randint(3, 50),  # KNN: Number of neighbors
            'classifier__knn__weights': ['uniform', 'distance'],  # KNN: Weight function
            'classifier__hgb__learning_rate': uniform(0.01, 0.2),  # HGB: Learning rate
            'classifier__hgb__max_iter': randint(50, 200),  # HGB: Number of boosting iterations
            }
    elif MODEL == "LP":
        params = {
            'classifier__kernel': ['rbf', 'knn'],  # Kernel type
            'classifier__gamma': uniform(0.01, 1.0)  # Only used for 'rbf' kernel
            }
    elif MODEL == "LS":
        params = {
            'classifier__kernel': ['rbf', 'knn'],  # Kernel type
            'classifier__gamma': uniform(0.01, 1.0),  # Only used for 'rbf' kernel
            'classifier__alpha': uniform(0.01, 0.5)  # Regularization parameter
            }
    else: #HGBC
            params = {
                'classifier__learning_rate': np.logspace(-3, 0, 10),   # Learning rate between 0.001 and 1
                'classifier__max_iter': [None, 50, 100, 200, 300],           # Number of boosting iterations
                #'classifier__max_depth': randint(1,79),
                'classifier__max_depth': [None, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],              # Maximum depth of trees
                'classifier__min_samples_leaf': [10, 20, 30],          # Minimum number of samples per leaf
                'classifier__l2_regularization': np.logspace(-4, 0, 10)  # L2 regularization strength
                }

    return params

def get_grid_search_params(best_params):
    if MODEL == 'KNN':
        params = {
            'classifier__n_neighbors': [best_params['classifier__n_neighbors'] - 2, best_params['classifier__n_neighbors'], best_params['classifier__n_neighbors'] + 2],  # Nearby values for n_neighbors
            'classifier__weights': [best_params['classifier__weights'], 'uniform' if best_params['classifier__weights'] == 'distance' else 'distance'],  # Toggle weights
            'classifier__algorithm': [best_params['classifier__algorithm'], 'auto'],  # Keep best algorithm with an additional option
            'classifier__p': [best_params['classifier__p'], 1 if best_params['classifier__p'] == 2 else 2],  # Toggle p between 1 and 2
            }
    elif MODEL == "NC":
        params = {
            'classifier__metric': [best_params['classifier__metric']],  # Keep the best metric
            'classifier__shrink_threshold': [
                max(0, best_params['classifier__shrink_threshold'] - 0.1),
                best_params['classifier__shrink_threshold'],
                min(1, best_params['classifier__shrink_threshold'] + 0.1),
                ],
            }
    elif MODEL == "RNC":
        params = {
            'classifier__radius': [best_params['classifier__radius'] * 0.5, best_params['classifier__radius'], best_params['classifier__radius'] * 1.5],  # Nearby values for radius
            'classifier__weights': [best_params['classifier__weights'], 'uniform' if best_params['classifier__weights'] == 'distance' else 'distance'],  # Toggle weights
            'classifier__algorithm': [best_params['classifier__algorithm'], 'auto'],  # Keep best algorithm with an additional option
            'classifier__leaf_size': [best_params['classifier__leaf_size'] - 10, best_params['classifier__leaf_size'], best_params['classifier__leaf_size'] + 10]
            }
    elif MODEL == "ETC":
        params = {
            'classifier__n_estimators': [best_params['classifier__n_estimators'] - 50, best_params['classifier__n_estimators'], best_params['classifier__n_estimators'] + 50],
            'classifier__max_features': [best_params['classifier__max_features']],  # Keep the best max features found by Random Search
            'classifier__max_depth': [best_params['classifier__max_depth'] - 5, best_params['classifier__max_depth'], best_params['classifier__max_depth'] + 5],
            'classifier__min_samples_split': [best_params['classifier__min_samples_split'] - 2, best_params['classifier__min_samples_split'], best_params['classifier__min_samples_split'] + 2],
            'classifier__min_samples_leaf': [best_params['classifier__min_samples_leaf'] - 2, best_params['classifier__min_samples_leaf'], best_params['classifier__min_samples_leaf'] + 2],
            }
    elif MODEL == 'SVC':
            params = {
                'classifier__C': [best_params['classifier__C'] * 0.1, best_params['classifier__C'], best_params['classifier__C'] * 10],
                'classifier__kernel': [best_params['classifier__kernel']],  # Keep the same kernel as found in Random Search
                #'classifier__gamma': [best_params['classifier__gamma'] * 0.1, best_params['classifier__gamma'], best_params['classifier__gamma'] * 10]
                'classifier__gamma': (
                        [best_params['classifier__gamma'] * 0.1,
                         best_params['classifier__gamma'],
                         best_params['classifier__gamma'] * 10
                        ]
                if isinstance(best_params['classifier__gamma'], (int, float))
                else [best_params['classifier__gamma']]  # Preserve 'scale' or 'auto'
                )
                }
    elif MODEL == "MLPC":
        params = {
            'classifier__hidden_layer_sizes': [(best_params['classifier__hidden_layer_sizes'][0] - 25,), 
                           best_params['classifier__hidden_layer_sizes'], 
                           (best_params['classifier__hidden_layer_sizes'][0] + 25,)], 
            'classifier__activation': [best_params['classifier__activation']],  # Keep the best activation function found by Random Search
            'classifier__solver': [best_params['classifier__solver']],  # Keep the best solver found by Random Search
            'classifier__alpha': [best_params['classifier__alpha'] * 0.5, best_params['classifier__alpha'], best_params['classifier__alpha'] * 1.5],
            'classifier__learning_rate': [best_params['classifier__learning_rate']],  # Keep the best learning rate found by Random Search
            'classifier__learning_rate_init': [best_params['classifier__learning_rate_init'] * 0.5, 
                           best_params['classifier__learning_rate_init'], 
                           best_params['classifier__learning_rate_init'] * 1.5],
            }
    elif MODEL == "LDA":
        params = {
            'classifier__solver': [best_params['classifier__solver']],  # Fix the solver to the best found
            'classifier__shrinkage': [
                max(0, best_params['classifier__shrinkage'] - 0.1),  # Reduce shrinkage slightly
                best_params['classifier__shrinkage'],
                min(1, best_params['classifier__shrinkage'] + 0.1)   # Increase shrinkage slightly
                ] if best_params['classifier__solver'] in ['lsqr', 'eigen'] else ['auto']  # 'auto' if shrinkage isn't used
            }
    elif MODEL == "QDA":
        params = {
            'classifier__reg_param': [
                max(0, best_params['classifier__reg_param'] - 0.1),
                best_params['classifier__reg_param'],
                min(1, best_params['classifier__reg_param'] + 0.1),
                ],
            'classifier__tol': [
                max(1e-6, best_params['classifier__tol'] - 1e-5),
                best_params['classifier__tol'],
                best_params['classifier__tol'] + 1e-5,
                ],
            }
    elif MODEL == 'RF':
            params = {
                'classifier__n_estimators': [best_params['classifier__n_estimators'] - 50, best_params['classifier__n_estimators'], best_params['classifier__n_estimators'] + 50],
                'classifier__max_depth': [best_params['classifier__max_depth'] - 10, best_params['max_depth'], best_params['classifier__max_depth'] + 10] if best_params['classifier__max_depth'] else [None],
                'classifier__max_features': [best_params['classifier__max_features']],
                'classifier__min_samples_leaf': [best_params['classifier__min_samples_leaf'] - 1, best_params['classifier__min_samples_leaf'], best_params['classifier__min_samples_leaf'] + 1],
                #'classifier__min_samples_split': [best_params['min_samples_split'] - 1, best_params['min_samples_split'], best_params['min_samples_split'] + 1]
                'classifier__min_samples_split': [best_params['classifier__min_samples_split']]
                }
    elif MODEL == "ABC":
        params = {
            'classifier__n_estimators': [
                max(50, best_params['classifier__n_estimators'] - 20), 
                best_params['classifier__n_estimators'], 
                best_params['classifier__n_estimators'] + 20
                ],
            'classifier__learning_rate': [
                max(0.01, best_params['classifier__learning_rate'] - 0.1), 
                best_params['classifier__learning_rate'], 
                best_params['classifier__learning_rate'] + 0.1
                ],
            'classifier__base_estimator__max_depth': [
                max(1, best_params['classifier__estimator__max_depth'] - 1), 
                best_params['classifier__estimator__max_depth'], 
                best_params['classifier__estimator__max_depth'] + 1
                ],
            'classifier__base_estimator__min_samples_split': [
                max(2, best_params['classifier__estimator__min_samples_split'] - 1), 
                best_params['classifier__estimator__min_samples_split'], 
                best_params['classifier__estimator__min_samples_split'] + 1
                ]
            }
    elif MODEL == "BC":
        params = {
            'classifier__n_estimators': [
                max(10, best_params['classifier__n_estimators'] - 10), 
                best_params['classifier__n_estimators'], 
                best_params['classifier__n_estimators'] + 10
                ],
            
            'classifier__max_samples': [
                max(0.1, best_params['classifier__max_samples'] - 0.1), 
                best_params['classifier__max_samples'], 
                min(1.0, best_params['classifier__max_samples'] + 0.1)
                ],
            
            'classifier__max_features': [
                max(0.1, best_params['classifier__max_features'] - 0.1), 
                best_params['classifier__max_features'], 
                min(1.0, best_params['classifier__max_features'] + 0.1)
                ],
            
            'classifier__estimator__max_depth': [
                max(1, best_params['classifier__estimator__max_depth'] - 1), 
                best_params['classifier__estimator__max_depth'], 
                best_params['classifier__estimator__max_depth'] + 1
                ],
            
            'classifier__estimator__min_samples_split': [
                max(2, best_params['classifier__estimator__min_samples_split'] - 1), 
                best_params['classifier__estimator__min_samples_split'], 
                best_params['classifier__estimator__min_samples_split'] + 1
                ]
        }

    elif MODEL == "GBC":
        params = {
            'classifier__n_estimators': [best_params['classifier__n_estimators'] - 50, best_params['classifier__n_estimators'], best_params['classifier__n_estimators'] + 50],
            'classifier__learning_rate': [best_params['classifier__learning_rate'] * 0.5, best_params['classifier__learning_rate'], best_params['classifier__learning_rate'] * 1.5],
            'classifier__max_depth': [best_params['classifier__max_depth'] - 1, best_params['classifier__max_depth'], best_params['classifier__max_depth'] + 1],
            'classifier__min_samples_split': [best_params['classifier__min_samples_split'] - 2, best_params['classifier__min_samples_split'], best_params['classifier__min_samples_split'] + 2],
            'classifier__min_samples_leaf': [best_params['classifier__min_samples_leaf'] - 1, best_params['classifier__min_samples_leaf'], best_params['classifier__min_samples_leaf'] + 1]
            }
    elif MODEL == "VC":
        params = {
            'classifier__lr__C': [
                max(0.01, best_params['classifier__lr__C'] - 0.5),
                best_params['classifier__lr__C'],
                best_params['classifier__lr__C'] + 0.5,
                ],
            'classifier__rf__n_estimators': [
                max(50, best_params['classifier__rf__n_estimators'] - 50),
                best_params['classifier__rf__n_estimators'],
                best_params['classifier__rf__n_estimators'] + 50,
                ],
            'classifier__rf__max_depth': [
                max(3, best_params['classifier__rf__max_depth'] - 2),
                best_params['classifier__rf__max_depth'],
                best_params['classifier__rf__max_depth'] + 2,
                ],
            'classifier__svc__C': [
                max(0.01, best_params['classifier__svc__C'] - 0.5),
                best_params['classifier__svc__C'],
                best_params['classifier__svc__C'] + 0.5,
                ],
            }
    elif MODEL == "SC":
        params = {
            'classifier__svc__C': [
                max(0.01, best_params['classifier__svc__C'] - 0.5),
                best_params['classifier__svc__C'],
                best_params['classifier__svc__C'] + 0.5,
                ],
            'classifier__svc__gamma': [
                max(0.01, best_params['classifier__svc__gamma'] - 0.1),
                best_params['classifier__svc__gamma'],
                best_params['classifier__svc__gamma'] + 0.1,
                ],
            'classifier__knn__n_neighbors': [
                max(3, best_params['classifier__knn__n_neighbors'] - 2),
                best_params['classifier__knn__n_neighbors'],
                best_params['classifier__knn__n_neighbors'] + 2,
                ],
            'classifier__hgb__learning_rate': [
                max(0.01, best_params['classifier__hgb__learning_rate'] - 0.05),
                best_params['classifier__hgb__learning_rate'],
                best_params['classifier__hgb__learning_rate'] + 0.05,
                ],
            'classifier__hgb__max_iter': [
                max(50, best_params['classifier__hgb__max_iter'] - 20),
                best_params['classifier__hgb__max_iter'],
                best_params['classifier__hgb__max_iter'] + 20,
                ],
            }
    elif MODEL == "LP":
        params = {
            'classifier__kernel': [best_params['classifier__kernel']],
            'classifier__gamma': [
                max(0.01, best_params['classifier__gamma'] - 0.1),
                best_params['classifier__gamma'],
                best_params['classifier__gamma'] + 0.1
                ] if 'classifier__gamma' in best_params else [None]
            }
    elif MODEL == "LS":
        params = {
            'classifier__kernel': [best_params['classifier__kernel']],
            'classifier__gamma': [
                max(0.01, best_params['classifier__gamma'] - 0.1),
                best_params['classifier__gamma'],
                best_params['classifier__gamma'] + 0.1
                ] if 'classifier__gamma' in best_params else [None],
            'classifier__alpha': [
                max(0.01, best_params['classifier__alpha'] - 0.05),
                best_params['classifier__alpha'],
                best_params['classifier__alpha'] + 0.05
                ]
            }
    else: #HGBC
            print(best_params)
            params = {
                'classifier__learning_rate':
                    [best_params['classifier__learning_rate'] * 0.5,
                     best_params['classifier__learning_rate'],
                     best_params['classifier__learning_rate'] * 1.5],
                'classifier__max_iter':
                    [best_params['classifier__max_iter'] - 50, 
                     best_params['classifier__max_iter'],
                     best_params['classifier__max_iter'] + 50],
                'classifier__max_depth':
                    [best_params['classifier__max_depth']]
                    if best_params['classifier__max_depth'] is None else
                       [best_params['classifier__max_depth'] - 5,
                       best_params['classifier__max_depth'],
                       best_params['classifier__max_depth'] + 5], 
                'classifier__min_samples_leaf': [best_params['classifier__min_samples_leaf'] - 5, best_params['classifier__min_samples_leaf'], best_params['classifier__min_samples_leaf'] + 5],
                #'classifier__min_samples_leaf': [best_params['classifier__min_samples_leaf']], # keep the same
                'classifier__l2_regularization': [best_params['classifier__l2_regularization'] * 0.5, best_params['classifier__l2_regularization'], best_params['classifier__l2_regularization'] * 1.5]
                #'classifier__l2_regularization': [best_params['classifier__l2_regularization']] # keep the same
                }
    
    # Ensure parameter values are valid
    params = {
        key: [
            val for val in values if val is None or isinstance(val, (str)) or isinstance(val, (tuple)) or val > 0
            ]
        for key, values in params.items()
        }
    
    return params
   
    
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
    
    if LEFT_RIGHT_AVERAGE:
        for i in range(0,17):
            
            col1 =  left[i]
            col2 = right[i]
            
            data_df[left_right_unite[i]] = (data_df[col1] + data_df[col2])/2
            data_df = data_df.drop([col1, col2], axis=1)
    
    if LEFT_RIGHT_DIFF:
        for i in range(0,17):
            
            col1 =  left[i]
            col2 = right[i]
            
            data_df[left_right_unite[i]] = abs((data_df[col1] - data_df[col2]))
            data_df = data_df.drop([col1, col2], axis=1)
            
    if LEFT_RIGHT_DROP:
        for i in range(0,17):
            
            col1 =  left[i]
            col2 = right[i]
            
            data_df = data_df.drop([col1, col2], axis=1)
    
            
    scores = data_df['score'].to_list()
    data_df = data_df.drop('score', axis=1)
    
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
                ('classifier', get_model())
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
    