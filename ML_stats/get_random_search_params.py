import numpy as np
from scipy.stats import randint, uniform
import sys

def get_random_search_params(model):
    if model == "LR":
        params = {
                'classifier__C': [0.01, 0.1, 1, 10, 100],               # Regularization strength
                'classifier__penalty': ['l1', 'l2', 'elasticnet', 'none'],  # Type of regularization
                #'classifier__solver': ['lbfgs', 'liblinear', 'saga'],       # Optimization solvers
                'classifier__solver': ['liblinear'],       # Optimization solvers
                'classifier__max_iter': [1000]                    # Number of iterations
             }
    elif model == "DT":
        params = {
             'classifier__criterion': ['gini', 'entropy', 'log_loss'],    # Splitting criteria
             'classifier__max_depth': [None, 2, 3, 5, 10],               # Maximum depth of the tree
             'classifier__min_samples_split': [2, 5, 10],                # Minimum samples required to split a node
             'classifier__min_samples_leaf': [1, 2, 4],                  # Minimum samples required at a leaf node
             'classifier__max_features': [None, 'sqrt', 'log2'],         # Number of features to consider for best split
             }
    elif model == "KNN":
        params = {
            'classifier__n_neighbors': randint(2, 15),  # Randomly select n_neighbors between 3 and 15
            'classifier__weights': ['uniform', 'distance'],  # Weight function used in prediction
            'classifier__algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],  # Algorithm for nearest neighbor search
            'classifier__metric': ['manhattan', 'canberra'],
            }
    elif model == "SVC":
            params = {
                # f1=0.71 3cl.&binary
                #'classifier__C': uniform(loc=0, scale=100),  # Regularization parameter
                'classifier__C': np.logspace(-3, 3, 100), # Regularization parameter
                'classifier__kernel': ['sigmoid', 'linear', 'rbf'],  # Kernel type
                #'classifier__gamma': [0.001, 0.01, 0.1, 1, 10, 100, 'scale', 'auto'],
                'classifier__gamma': ['scale', 'auto'] + list(np.logspace(-3, 2, 10)),
                }
    elif model == "HGBC":
            params = {
                #'classifier__max_depth': randint(1,79),
                'classifier__learning_rate': np.logspace(-3, 0, 10),   # Learning rate between 0.001 and 1
                'classifier__max_iter': [None, 100],           # Number of boosting iterations
                'classifier__max_depth': [None, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],              # Maximum depth of trees
                'classifier__min_samples_leaf': [10, 20, 30],          # Minimum number of samples per leaf
                #'classifier__l2_regularization': np.logspace(-4, 0, 10)  # L2 regularization strength
                }
    elif model == "RC":
        params = {
            'classifier__alpha': np.logspace(-4, 2, 100),  # Regularization strength
            'classifier__solver': ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga', 'lbfgs']  # Solvers
            }
    elif model == "XGB":
        params = {
            'classifier__n_estimators': np.arange(50, 200, 50),
            'classifier__learning_rate': np.logspace(-3, 0, 10),
            'classifier__max_depth': np.arange(3, 10, 2),
            'classifier__subsample': np.linspace(0.6, 1.0, 5),
            'classifier__colsample_bytree': np.linspace(0.6, 1.0, 5)
            }
    elif model == "NC":
        params = {
            'classifier__metric': ['euclidean', 'manhattan', 'minkowski'],  # Distance metrics
            'classifier__shrink_threshold': uniform(0, 1),  # Shrinkage threshold (0 to 1)
            }
    elif model == "RNC":
        params = {
            'classifier__radius': uniform(0.1, 2.0),  # Randomly sample radius between 0.1 and 1.0
            'classifier__weights': ['uniform', 'distance'],  # Weight function used in prediction
            'classifier__algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],  # Algorithm for nearest neighbor search
            'classifier__leaf_size': [10, 20, 30, 40, 50],  # Size of the leaf for the tree-based algorithms
            }
    elif model == "ETC":
        params = {
            'classifier__n_estimators': randint(50, 300),  # Randomly search between 50 and 300 trees
            'classifier__max_features': ['auto', 'sqrt', 'log2', None],  # Try different options for max features
            'classifier__max_depth': randint(5, 50),  # Random search between depth of 5 and 50
            'classifier__min_samples_split': randint(2, 20),  # Random search for min samples for splitting
            'classifier__min_samples_leaf': randint(1, 20),  # Random search for min samples for leaf nodes
            }
    elif model == "LGBM":
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
    elif model == "MLPC":
        params = {
            'classifier__hidden_layer_sizes': [(50,), (100,), (150,), (200,)],  # Vary the number of neurons in hidden layers
            'classifier__activation': ['relu', 'tanh', 'logistic'],  # Different activation functions
            'classifier__solver': ['adam', 'sgd'],  # Solvers to use
            'classifier__alpha': uniform(0.0001, 0.1),  # Regularization strength
            'classifier__learning_rate': ['constant', 'invscaling', 'adaptive'],  # Learning rate schedules
            'classifier__learning_rate_init': uniform(0.0001, 0.1),  # Initial learning rate
            }
    elif model == "LDA":
        params = {
            'classifier__solver': ['svd', 'lsqr', 'eigen'],  # Solvers for LDA
            'classifier__shrinkage': uniform(0.0, 1.0)       # Shrinkage parameter for 'lsqr' and 'eigen' solvers
            }
    elif model == "QDA":
        params = {
            'classifier__reg_param': uniform(0, 1),  # Regularization parameter (0 to 1)
            'classifier__tol': uniform(1e-5, 1e-3),  # Tolerance for stopping criterion
            }
    elif  (model == "RF") or (model == "BRF"):
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
    elif model == "ABC":
            params = {
                'classifier__n_estimators': randint(50, 200),  # Number of weak classifiers
                'classifier__learning_rate': uniform(0.01, 1.0),  # Shrinks the contribution of each classifier
                'classifier__estimator__max_depth': randint(1, 10),  # Depth of trees in base estimator
                'classifier__estimator__min_samples_split': randint(2, 10),  # Minimum samples to split an internal node
                }
    elif model == "BC":
        params = {
            'classifier__n_estimators': randint(10, 100),  # Number of base estimators
            'classifier__max_samples': uniform(0.5, 0.5),  # Fraction of samples for training each base estimator
            'classifier__max_features': uniform(0.5, 0.5),  # Fraction of features for training each base estimator
            'classifier__estimator__max_depth': randint(1, 10),  # Depth of trees in base estimator
            'classifier__estimator__min_samples_split': randint(2, 10),  # Minimum samples required to split
            }
    elif model == "GBC":
        params = {
            'classifier__n_estimators': [50, 100, 200, 300, 400],  # Number of boosting stages
            'classifier__learning_rate': uniform(0.01, 0.2),  # Step size shrinking to prevent overfitting (uniform distribution)
            'classifier__max_depth': [3, 5, 7, 9, 10],  # Maximum depth of the individual trees
            'classifier__min_samples_split': [2, 5, 10, 20],  # Minimum number of samples required to split an internal node
            'classifier__min_samples_leaf': [1, 2, 4, 8]  # Minimum number of samples required to be at a leaf node
            }
    elif model == "VC":
        params = {
            'classifier__svc__C': [0.001, 0.01, 1, 5, 10, 100],  # SVC: Regularization parameter
            'classifier__svc__kernel': ['linear', 'rbf'], # SVC: Kernel
            'classifier__svc__gamma': [0.001, 0.01, 0.1, 1, 10, 100, 'scale', 'auto'],  # SVC: Kernel coefficient
           
            'classifier__knn__n_neighbors': randint(3, 50),  # KNN: Number of neighbors
            'classifier__knn__weights': ['uniform', 'distance'],  # KNN: Weight function
            'classifier__knn__algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],  # KNN: Algorithm for nearest neighbor search
            'classifier__knn__p': [1, 2],  # KNN: Power parameter for the Minkowski distance (1=Manhattan, 2=Euclidean)

            'classifier__hgb__learning_rate': np.logspace(-3, 0, 10),  # HGB: Learning rate
            'classifier__hgb__max_iter': [None, 50, 100, 200, 300], # HGB: Number of boosting iterations
            'classifier__hgb__max_depth': [None, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],              # HGB: Maximum depth of trees
            'classifier__hgb__min_samples_leaf': [10, 20, 30],          # HGB: Minimum number of samples per leaf
            'classifier__hgb__l2_regularization': np.logspace(-4, 0, 10)  # HGB: L2 regularization strength
            }
    elif model == "SC":
        params = {
            'classifier__svc__C': [0.001, 0.01, 1, 5, 10, 100],  # SVC: Regularization parameter
            'classifier__svc__kernel': ['linear', 'rbf'], # SVC: Kernel
            'classifier__svc__gamma': [0.001, 0.01, 0.1, 1, 10, 100, 'scale', 'auto'],  # SVC: Kernel coefficient
           
            'classifier__knn__n_neighbors': randint(3, 50),  # KNN: Number of neighbors
            'classifier__knn__weights': ['uniform', 'distance'],  # KNN: Weight function
            'classifier__knn__algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],  # KNN: Algorithm for nearest neighbor search
            'classifier__knn__p': [1, 2],  # KNN: Power parameter for the Minkowski distance (1=Manhattan, 2=Euclidean)

            'classifier__hgb__learning_rate': np.logspace(-3, 0, 10),  # HGB: Learning rate
            'classifier__hgb__max_iter': [None, 50, 100, 200, 300], # HGB: Number of boosting iterations
            'classifier__hgb__max_depth': [None, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],              # HGB: Maximum depth of trees
            'classifier__hgb__min_samples_leaf': [10, 20, 30],          # HGB: Minimum number of samples per leaf
            'classifier__hgb__l2_regularization': np.logspace(-4, 0, 10)  # HGB: L2 regularization strength
            }
    elif model == "LP":
        params = {
            'classifier__kernel': ['rbf', 'knn'],  # Kernel type
            'classifier__gamma': uniform(0.01, 1.0)  # Only used for 'rbf' kernel
            }
    elif model == "LS":
        params = {
            'classifier__kernel': ['rbf', 'knn'],  # Kernel type
            'classifier__gamma': uniform(0.01, 1.0),  # Only used for 'rbf' kernel
            'classifier__alpha': uniform(0.01, 0.5)  # Regularization parameter
            }
    else:
        print("Model not supported")
        sys.exit(0)

    return params
