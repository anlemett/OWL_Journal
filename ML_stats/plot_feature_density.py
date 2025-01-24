import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings('ignore')

import time
import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from features import diameter_left_right, amplitude_left_right, speed_left_right
from features import saccade_duration, fixation_duration, blink_duration
from features import head_position
columns_to_select = diameter_left_right
#columns_to_select = amplitude_left_right
xlabel = "Value [m]"
#columns_to_select = speed_left_right
#xlabel = "Value [m/sec]"
#columns_to_select = saccade_duration
#columns_to_select = fixation_duration
#columns_to_select = blink_duration
#xlabel = "Value [1/250 sec]"
#columns_to_select = head_position
#xlabel = "Value [rad]"
#columns_to_select = ["Saccades Number", "Blinks Number"]
#xlabel = "number"

DATA_DIR = os.path.join("..", "..")
DATA_DIR = os.path.join(DATA_DIR, "Data")
ML_DIR = os.path.join(DATA_DIR, "MLInput")
#ML_DIR = os.path.join(DATA_DIR, "MLInput_Journal")
FIG_DIR = os.path.join(".", "Figures")

filename = "ML_features_CHS.csv"
full_filename = os.path.join(ML_DIR, filename)
    
df = pd.read_csv(full_filename, sep=' ')
df = df[columns_to_select]

# Plot density for multiple features on one plot
plt.figure(figsize=(10, 6))
for column in df.columns:
    sns.kdeplot(df[column], label=column, fill=True, alpha=0.3)  # `fill` adds shaded density curves

#plt.title("Density Plot of Selected Features")
plt.xlabel(xlabel, fontsize=16)
plt.ylabel("Density", fontsize=16)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)

plt.legend(title="Features", fontsize=16)

plt.tight_layout()
plt.show()