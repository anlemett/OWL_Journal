import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

#from sklearn.model_selection import train_test_split, RandomizedSearchCV
#from sklearn import preprocessing
#from scipy.stats import randint
#from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import balanced_accuracy_score, f1_score

from sklearn.base import clone
from sklearn.inspection import permutation_importance

SCORING = 'f1_macro'

RANDOM_STATE = 0

# Function to calculate permutation importance using sklearn
def calculate_permutation_importance(model, X_val, y_val, n_repeats):

    importances = []
    result = permutation_importance(model, X_val, y_val, scoring=SCORING, n_repeats=n_repeats, random_state=RANDOM_STATE)
    
    average_importances = result.importances_mean #importances_mean - means over n_repeats
        
    return average_importances #importances for all features from X_val
'''
# Function to calculate permutation importance using sklearn
def calculate_permutation_importance(model, X_val, y_val, n_repeats):

    importances = []

    for seed in range(5):  # Aggregate over 5 seeds
        result = permutation_importance(model, X_val, y_val, SCORING, n_repeats, seed)
        importances.append(result.importances_mean) #importances_mean - means over n_repeats

    average_importances = np.mean(importances, axis=0) #means over seeds
        
    return average_importances #importances for all features from X_val
'''

# Custom RFE class with permutation importance
class RFEPermutationImportance:
    def __init__(self, estimator, n_features_to_select=None, step=1, min_features_to_select=1, n_repeats=20):
        self.estimator = estimator
        self.n_features_to_select = n_features_to_select
        self.step = step
        self.min_features_to_select = min_features_to_select
        self.importances = []
        self.least_important_feature = ""
        self.least_importance = 0
        self.n_repeats = n_repeats
        self.bal_accuracies_ = []
        self.f1_scores_ = []
        self.features_by_importance_ = []

    def fit(self, X, y, X_test, y_test, features_lst):
        self.estimator_ = clone(self.estimator)
        
        features = list(features_lst)
        
        X_df = pd.DataFrame(X, columns=features)
        X_test_df = pd.DataFrame(X_test, columns=features)
        
        self.estimator_.fit(X_df[features], y)
        
        test_bal_accuracy = balanced_accuracy_score(y_test, self.estimator_.predict(X_test_df[features]))
        self.bal_accuracies_.append((len(features), test_bal_accuracy))
        
        test_f1_score = f1_score(y_test, self.estimator_.predict(X_test_df[features]), average='macro')
        self.f1_scores_.append((len(features), test_f1_score))
        
        while len(features) > self.min_features_to_select:
            self.estimator_.fit(X_df[features], y)
            self.importances = calculate_permutation_importance(self.estimator_, X_df[features], y, self.n_repeats)
            
            # Identify least important feature
            least_important_feature_index = np.argmin(self.importances)
            self.least_importance = self.importances[least_important_feature_index]
            self.least_important_feature = features[least_important_feature_index]
            
            # Remove the least important feature
            features.remove(self.least_important_feature)
            self.features_by_importance_.append(self.least_important_feature)
            
            print(f'Removed feature: {self.least_important_feature}')
            print(f'Remaining features: {len(features)}')
            
            # Evaluate and store test accuracy and F1-score without removed feature
            
            self.estimator_.fit(X_df[features], y)
            
            test_bal_accuracy = balanced_accuracy_score(y_test, self.estimator_.predict(X_test_df[features]))
            self.bal_accuracies_.append((len(features), test_bal_accuracy))
            
            test_f1_score = f1_score(y_test, self.estimator_.predict(X_test_df[features]), average='macro')
            self.f1_scores_.append((len(features), test_f1_score))
                    
        self.support_ = np.isin(X_df.columns, features)
        self.ranking_ = np.ones(len(X_df.columns), dtype=int)
        self.ranking_[~self.support_] = len(X_df.columns) - np.sum(self.support_) + 1
        
        return self

    def transform(self, X):
        return X.loc[:, self.support_]
