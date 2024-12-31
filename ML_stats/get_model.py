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
from imblearn.ensemble import BalancedRandomForestClassifier, EasyEnsembleClassifier
#from lightgbm import LGBMClassifier
#from xgboost import XGBClassifier


def get_model(model, random_state):
    print(f"Model: {model}")
    if  model == "KNN":
        return KNeighborsClassifier()
    #elif model == "XGB":
    #    return XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=random_state)
    elif model == "NC":
        return NearestCentroid()
    elif model == "RNC":
        return RadiusNeighborsClassifier(outlier_label='most_frequent')
    elif model == "ETC":
        return ExtraTreesClassifier(random_state=random_state)
    #elif model == "LGBM":
    #    return LGBMClassifier(random_state=RANDOM_STATE, verbose=-1)
    elif model == "SVC":
        return SVC()
    elif model == "MLPC":
        return MLPClassifier(max_iter=1000, random_state=random_state)
    elif model == "LDA":
        return LinearDiscriminantAnalysis()
    elif model == "QDA":
        return QuadraticDiscriminantAnalysis()
    elif model == "RF":
        return RandomForestClassifier(random_state=random_state, max_features=None)
    elif model == "BRF":
        return BalancedRandomForestClassifier(
                    bootstrap=False,
                    replacement=True,
                    sampling_strategy='auto',  # Under-sample majority class to match minority
                    random_state=random_state
                    )
    elif model == "EEC":
        return EasyEnsembleClassifier(random_state=random_state)
    elif model == "ABC":
        base_estimator = DecisionTreeClassifier(random_state=random_state)
        adaboost = AdaBoostClassifier(estimator=base_estimator, random_state=random_state)
        return adaboost
    elif model == "BC":
        base_estimator = DecisionTreeClassifier(random_state=random_state)
        bagging = BaggingClassifier(estimator=base_estimator, random_state=random_state)
        return bagging
    elif model == "VC":
        # Initialize VotingClassifier
        voting_clf = VotingClassifier(
            estimators=[
                ('svc', SVC(probability=True, random_state=random_state)),
                ('knn', KNeighborsClassifier()),
                ('hgb', HistGradientBoostingClassifier(random_state=random_state))
                ],
            voting='soft'
            )
        return voting_clf
    elif model == "SC":
        base_estimators = [
            ('svc', SVC(probability=True, random_state=random_state)),
            ('knn', KNeighborsClassifier()),
            ('hgb', HistGradientBoostingClassifier(random_state=random_state))
            ]

        # Define meta-classifier
        meta_clf = KNeighborsClassifier()

        # Initialize StackingClassifier
        stacking_clf = StackingClassifier(
            estimators=base_estimators,
            final_estimator=meta_clf
            )
        return stacking_clf
    elif model == "LP":
        return LabelPropagation()
    elif model == "LS":
        return LabelSpreading()
    elif model == "GBC":
        return GradientBoostingClassifier(random_state=random_state)
    else:
        return HistGradientBoostingClassifier(random_state=random_state)
