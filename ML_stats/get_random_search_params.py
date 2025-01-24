import numpy as np
from scipy.stats import randint, uniform
import sys

def get_random_search_params(model):
    if model == "LR":
        params = {
                'classifier__C': [0.01, 0.1, 1, 10, 100],                   # Regularization strength
                'classifier__penalty': ['None', 'l1', 'l2', 'elasticnet'],  # Type of regularization
                #'classifier__solver': ['lbfgs', 'liblinear', 'saga'],       # Optimization solvers
                'classifier__solver': ['liblinear'],       # Optimization solvers
                'classifier__max_iter': [1000]             # Number of iterations
             }
    elif model == "DT":
        params = {
             'classifier__criterion': ['gini', 'entropy', 'log_loss'],  # Splitting criteria
             'classifier__max_depth': [None, randint(1, 10)],           # Maximum depth of the tree
             'classifier__min_samples_split': [2, 5, 10],               # Minimum samples required to split a node
             'classifier__min_samples_leaf': [1, 2, 4],                 # Minimum samples required at a leaf node
             'classifier__max_features': [None, 'sqrt', 'log2'],        # Number of features to consider for best split
             }
    elif model == "KNN":
        params = {
            'classifier__n_neighbors': randint(2, 15),       # Randomly select n_neighbors between 2 and 15
            'classifier__weights': ['uniform', 'distance'],  # Weight function used in prediction
            'classifier__algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],  # Algorithm for nearest neighbor search
            'classifier__metric': ['manhattan', 'canberra'],
            }

    elif model == "SVC":
            '''
            #with_min_max
            params = {
                'classifier__C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],   # Regularization parameter
                'classifier__kernel': ['rbf'],
                'classifier__gamma': [0.001, 0.01, 0.1, 1, 10, 100, 'scale', 'auto'],
                }
            '''
            #no_min_max
            params = {
                #'classifier__C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],   # Regularization parameter
                'classifier__C': [10],
                'classifier__kernel': ['rbf'],
                'classifier__gamma': ['scale'],
                }
    elif model == "HGBC":
            params = {
                'classifier__learning_rate': np.logspace(-3, 0, 10),   # Learning rate between 0.001 and 1
                'classifier__max_iter': [None, 100],                   # Number of boosting iterations
                'classifier__max_depth': randint(1, 100),              # Maximum depth of trees
                'classifier__min_samples_leaf': [10, 20, 30],          # Minimum number of samples per leaf
                #'classifier__l2_regularization': np.logspace(-4, 0, 10)  # L2 regularization strength
                }
    elif model == "ETC":
        #with_min_max
        '''
        params = {
            'classifier__n_estimators': randint(1, 100),                 # Randomly search between 1 and 100 trees
            'classifier__max_features': ['auto', 'sqrt', 'log2', None],  # Try different options for max features
            'classifier__min_samples_split': randint(2, 20),             # Random search for min samples for splitting
            'classifier__min_samples_leaf': randint(1, 20),              # Random search for min samples for leaf nodes
            }
        '''
        #no_min_max
        params = {
            'classifier__n_estimators': randint(1, 100),                 # Randomly search between 1 and 100 trees
            'classifier__max_features': ['sqrt', 'log2'],                # Try different options for max features
            'classifier__max_depth': randint(1, 10),                     # Maximum depth of trees
            'classifier__min_samples_split': randint(2, 20),             # Random search for min samples for splitting
            'classifier__min_samples_leaf': randint(1, 20),              # Random search for min samples for leaf nodes
            }
    elif model == "LDA":
        params = {
            'classifier__solver': ['svd', 'lsqr', 'eigen'],  # Solvers for LDA
            'classifier__shrinkage': uniform(0.0, 1.0)       # Shrinkage parameter for 'lsqr' and 'eigen' solvers
            }
    elif  (model == "RF"):
            params = {
                'classifier__n_estimators': randint(50, 500),
                'classifier__max_features': ['sqrt', 'log2', None],
                'classifier__min_samples_leaf': randint(1, 40),
                'classifier__min_samples_split': randint(2, 40)
                }
    elif model == "ABC":
            params = {
                'classifier__n_estimators': randint(50, 200),                # Number of weak classifiers
                'classifier__learning_rate': uniform(0.01, 1.0),             # Shrinks the contribution of each classifier
                'classifier__estimator__min_samples_split': randint(2, 10),  # Minimum samples to split an internal node
                }
    elif model == "BC":
        params = {
            'classifier__n_estimators': randint(10, 100),                # Number of base estimators
            'classifier__max_samples': uniform(0.5, 0.5),                # Fraction of samples for training each base estimator
            'classifier__max_features': uniform(0.5, 0.5),               # Fraction of features for training each base estimator
            'classifier__estimator__min_samples_split': randint(2, 10),  # Minimum samples required to split
            }
    else:
        print("Model not supported")
        sys.exit(0)

    return params
