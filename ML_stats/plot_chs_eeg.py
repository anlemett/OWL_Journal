import warnings
warnings.filterwarnings('ignore')

import time
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

current_dir = os.getcwd()
DATA_DIR = os.path.join(current_dir, '..', '..', 'Data')
EEG_DIR = os.path.join(DATA_DIR, "EEG2")
ML_DIR = os.path.join(DATA_DIR, "MLInput")
FIG_DIR = os.path.join(".", "Figures")

full_filename = os.path.join(ML_DIR, "ML_CH.csv")
chs_df =  pd.read_csv(full_filename, sep=' ')

full_filename = os.path.join(EEG_DIR, "EEG_all_180.csv")
eeg_df = pd.read_csv(full_filename, sep=' ')

#TODO: interpolation for EEG
threshold = eeg_df['WorkloadMean'].quantile(0.93)
eeg_df['Workload'] = eeg_df['WorkloadMean'].apply(lambda x: 2 if x > threshold else 1)


for atco_num in range(1, 19):
    chs_atco = chs_df[chs_df['ATCO']==atco_num]
    eeg_atco = eeg_df[eeg_df['ATCO']==atco_num]
    
    chs_atco['Workload'] = chs_atco['score'].apply(lambda x: 2 if x > 3 else 1)
    
    #threshold = eeg_atco['WorkloadMean'].quantile(0.93)
    #eeg_atco['Workload'] = eeg_atco['WorkloadMean'].apply(lambda x: 2 if x > threshold else 1)
    
    for run_num in range(1,4):

        chs_run = chs_atco[chs_atco['Run']==run_num]
        eeg_run = eeg_atco[eeg_atco['Run']==run_num]

        chs_np = np.array(chs_run['Workload'])
        eeg_np = np.array(eeg_run['Workload'])

        print(len(chs_np))
        print(len(eeg_np))
        
        # Find the minimum length
        min_length = min(len(chs_np), len(eeg_np))

        # Truncate both arrays to the same length
        chs_np = chs_np[:min_length]
        eeg_np = eeg_np[:min_length]

        x = np.arange(1, len(chs_np) + 1)

        # Plot the values
        plt.figure(figsize=(8, 5))

        plt.plot(x, chs_np, marker='o', linestyle='-', color='b', label='CHS')

        # Plot the second array
        plt.plot(x, eeg_np, marker='s', linestyle='--', color='r', label='EEG')

        plt.yticks([1, 2])
        
        # Add labels, title, and legend
        plt.xlabel('Time slot')
        plt.ylabel('Workload')
        plt.title(f"ATCO {atco_num}, run {run_num}")
        plt.legend()
        plt.grid(True)

        filename = f"ATCO{atco_num}_run{run_num}.png"
        full_filename = os.path.join(FIG_DIR, filename)
        plt.savefig(full_filename, dpi=600)
        # Show the plot
        plt.show()