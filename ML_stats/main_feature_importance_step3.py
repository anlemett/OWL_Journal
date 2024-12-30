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
#columns_order = mportance_CHS_binary_HGBC
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
    #print("point1:", point1)
    #print("point2:", point2)
    #print("line:", line)
    distances = np.cross(point2-point1, point1-line.T)/np.linalg.norm(point2-point1)
    elbow_index = np.argmax(np.abs(distances))

    #return x[elbow_index]
    return elbow_index


features = columns_order
##############
def main():

    #CHS, binary, SVC
    if MODEL == "SVC":
        if BINARY:
            bal_accuracies = [0.7891406663141195, 0.7799471179270228, 0.768253569539926, 0.7574603384452671, 0.7582667900581702, 0.7622990481226865, 0.76068614489688, 0.7614925965097832, 0.7498664727657325, 0.75068614489688, 0.7498796932839767, 0.7398532522474881, 0.7398532522474881, 0.7322858276044422, 0.7213987308302485, 0.7257271285034373, 0.7189661554732946, 0.7150145425700688, 0.7332403490216817, 0.7317080909571656, 0.7460364886303543, 0.7141274457958752, 0.7309148598625066, 0.7396113167636171, 0.7913048651507139, 0.7847726070861978, 0.7624338974087784, 0.718213907985193, 0.6773003701745108, 0.6818574828133264, 0.6277723426758329, 0.6496390798519303]
            f1_scores = [0.7496137012957835, 0.7343592743034374, 0.7368939239037041, 0.7250417811244523, 0.7254539366362632, 0.738033233311733, 0.7327955336863429, 0.7379124680364628, 0.7247285425116501, 0.7262728136920668, 0.7258670481881907, 0.7182683267975342, 0.7207365471956759, 0.7208043459388594, 0.7182174656271061, 0.7031395001324412, 0.7092006632349505, 0.6927326720129943, 0.7140144026177538, 0.6951089628587998, 0.6938052732951837, 0.6850626064220601, 0.7057435153016722, 0.6842223468630011, 0.7198322658414614, 0.7060474684701645, 0.6861393106365232, 0.6181430168723145, 0.5698906109879672, 0.5606391175307908, 0.5326128224066251, 0.5655706103217558]
        else:
            bal_accuracies = [0.6917009025832556, 0.6924535740222015, 0.6993863471314452, 0.7112750285299306, 0.6851343500363108, 0.6838494657122108, 0.6889438738458347, 0.6970391119410728, 0.7030075733997303, 0.6874151882975412, 0.6958247743541862, 0.709174188193796, 0.7192021993982778, 0.6962169312169312, 0.7000285299304908, 0.7091041601825916, 0.714486461251167, 0.713728083826123, 0.7241347650171178, 0.7390305010893247, 0.7320318497769478, 0.7367755991285403, 0.750785351177508, 0.7502251270878721, 0.7375635439360929, 0.7201882975412388, 0.7141238717709306, 0.7122668326589896, 0.6595995435211122, 0.6271023965141612, 0.5591985683162155, 0.5239765535843968]
            f1_scores = [0.6863544788991758, 0.6944432989231072, 0.6998934617976637, 0.7091405078532974, 0.6846126791072968, 0.6816701737881432, 0.6893824238590336, 0.6946892641004737, 0.6967058068027553, 0.6832617868847979, 0.690518337103593, 0.7051833162610035, 0.7126631723327296, 0.6952166735074402, 0.7046214331210264, 0.7144847177398395, 0.7192378593393889, 0.7167445836436767, 0.7252186745002707, 0.7340051346711544, 0.7346677628976227, 0.7237524256335977, 0.7410557000448181, 0.7351226968789217, 0.7128213832398189, 0.6960090480958874, 0.6862496574786785, 0.6808613333482455, 0.6131772890082513, 0.5699221788158926, 0.49531332514901016, 0.47592758434570365]

    #CHS, binary, HGBC
    else:
        if BINARY:
            bal_accuracies = [0.7891406663141195, 0.768253569539926, 0.7766406663141195, 0.7790600211528291, 0.7706729243786357, 0.76068614489688, 0.7714793759915389, 0.7498664727657325, 0.7606729243786357, 0.7907667900581703, 0.7499471179270227, 0.757527763088313, 0.7442080909571656, 0.7142213114754098, 0.7291274457958752, 0.7366538868323639, 0.7291406663141194, 0.7374471179270228, 0.7192622950819672, 0.7542755156002114, 0.7434690639873083, 0.7334016393442622, 0.7534148598625067, 0.7505658381808565, 0.7453239026969858, 0.7316010047593866, 0.737971972501322, 0.729074563722898, 0.7593429402432575, 0.716090692755156, 0.6737652035959809, 0.6496390798519303]
            f1_scores = [0.7496137012957835, 0.7317619816629373, 0.7412339052776981, 0.75032763835423, 0.740564064076149, 0.729565353286476, 0.7417207413365345, 0.7273403555273722, 0.7348571642599678, 0.7435780362785381, 0.7155070470969754, 0.7131164398192831, 0.7037109379990792, 0.6838102215936944, 0.6951472013512532, 0.7059361001806774, 0.6992640006965013, 0.7071206953022022, 0.685483591631441, 0.7118914056150584, 0.69999103776828, 0.6983284152449161, 0.7075946083832201, 0.6815147522545674, 0.6977665452336843, 0.6620451051588581, 0.6348975686610248, 0.6182852238623866, 0.6286625731952348, 0.604491016408913, 0.5759018495075662, 0.5655706103217558]
        else:
            bal_accuracies = [0.7891406663141195, 0.7791406663141196, 0.7799471179270228, 0.7598664727657324, 0.7522990481226864, 0.7599603384452671, 0.7615732416710734, 0.7615732416710734, 0.7607667900581703, 0.7816538868323638, 0.7615732416710734, 0.7515732416710735, 0.7566406663141195, 0.7458342147012164, 0.7450277630883131, 0.7550277630883131, 0.7258342147012163, 0.7667213114754099, 0.7325145425700688, 0.7618019566367003, 0.7626216287678477, 0.7550409836065575, 0.7642345319936542, 0.7626216287678478, 0.760995505023797, 0.7554177683765204, 0.7527432575356954, 0.7270716552088843, 0.7629997355896351, 0.7031491274457959, 0.6329230565838181, 0.6311237440507667]
            f1_scores = [0.7496137012957835, 0.7410013847839333, 0.7441482746106028, 0.7297129888510678, 0.7306000856252617, 0.7234057636122653, 0.726517142465043, 0.7275641556832161, 0.7210767948012826, 0.7328013495166802, 0.7249540511947007, 0.7128103047992077, 0.7141387619070547, 0.7034082990944114, 0.7138310268778509, 0.7193897180945142, 0.7016842152332713, 0.7287326905966541, 0.7056946921381818, 0.7167910958417366, 0.7176678705320748, 0.7182620908096464, 0.7248198258245729, 0.7193949168366089, 0.7167996886017625, 0.6911601567829534, 0.6553889615446813, 0.6305429986012999, 0.6477784080516313, 0.5535900057227934, 0.48569478952344164, 0.4651139321084923]
            
    y = np.array(bal_accuracies[::-1])
    y_ = np.array(f1_scores[::-1])
    x = np.array(range(1, len(y)+1))
        
    elbow_point_idx = find_elbow_point(x, y)
    elbow_point_num = elbow_point_idx + 1
    elbow_point_acc = y[elbow_point_idx]
    corr_f1 = y_[elbow_point_idx]
    print(f"The elbow point for bal.accuracy is at {elbow_point_num} features.")
    print(f"Bal.accuracy at the elbow point: {elbow_point_acc}")
    print(f"Corresponding F1-score: {corr_f1}")
        
    elbow_point_f1_idx = find_elbow_point(x, y_)
    elbow_point_f1_num = elbow_point_f1_idx + 1
    elbow_point_f1 = y_[elbow_point_f1_idx]
    corr_acc = y[elbow_point_f1_idx]
    print(f"The elbow point for F1-score is at {elbow_point_f1_num} features.")
    print(f"F1-score at the elbow point: {elbow_point_f1}")
    print(f"Corresponding bal.accuracy: {corr_acc}")
        
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
    ax0.set_ylabel('Balanced accuracy', fontsize=14)
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
    plt.tight_layout()
    plt.savefig(full_filename, dpi=600)
    plt.show()
        
    max_acc = max(bal_accuracies)
    max_index = bal_accuracies.index(max_acc)
    #print(f"Max_index: {max_index}")
    number_of_features = max_index + 1
    acc = bal_accuracies[max_index]
    f1 = f1_scores[max_index]
    print("Optimal number of features by maximizing Bal.Accuracy")
    print(f"Optimal number of features: {number_of_features}, Bal.Accuracy: {acc}, F1-score: {f1}")

    max_f1 = max(f1_scores)
    max_index = f1_scores.index(max_f1)
    #print(f"Max_index: {max_index}")
    number_of_features = max_index + 1
    acc = bal_accuracies[max_index]
    f1 = f1_scores[max_index]
    print("Optimal number of features by maximizing F1-score")
    print(f"Optimal number of features: {number_of_features}, Bal.Accuracy: {acc}, F1-score: {f1}, ")
    

##############        

start_time = time.time()

main()

elapsed_time = time.time() - start_time
print(f"Elapsed time: {elapsed_time:.3f} seconds")
    