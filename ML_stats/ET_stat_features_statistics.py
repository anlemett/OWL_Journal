import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings('ignore')

import time
import os
import numpy as np
import pandas as pd

from features import init, init_blinks, blinks
columns_to_select = init_blinks

DATA_DIR = os.path.join("..", "..")
DATA_DIR = os.path.join(DATA_DIR, "Data")
ML_DIR = os.path.join(DATA_DIR, "MLInput")
#ML_DIR = os.path.join(DATA_DIR, "MLInput_Journal")
FIG_DIR = os.path.join(".", "Figures")

def main():

    filename = "ML_features_CHS.csv"
    full_filename = os.path.join(ML_DIR, filename)
    
    df = pd.read_csv(full_filename, sep=' ')
    
    column_means = df.mean()
    column_stds = df.std()

    # Create a DataFrame for output
    stats_df = pd.DataFrame({
        'Mean': column_means,
        'Standard Deviation': column_stds
        })

    # Output to a text file
    with open('ET_stat_features_statistics.txt', 'w') as file:
        file.write("Column Statistics:\n")
        file.write(stats_df.to_string())  # Write the DataFrame as a string

    # Alternatively, output to a CSV file
    stats_df.to_csv('ET_stat_features_statistics.csv')
    
    zero_std_columns = column_stds[column_stds == 0].index.tolist()

    # Output to a text file
    with open('zero_std_columns.txt', 'w') as file:
        file.write("Columns with standard deviation = 0:\n")
        for col in zero_std_columns:
            file.write(f"{col}\n")
    
main()