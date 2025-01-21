# Proportion of EEG high/low workload in light, moderate and heavy scenarios
import os
import numpy as np
import pandas as pd

DATA_DIR = os.path.join("..", "..")
DATA_DIR = os.path.join(DATA_DIR, "Data")
CHS_DIR = os.path.join(DATA_DIR, "MLInput")

scenario_dict = {
    1 :[1, 2, 3],
    2: [3, 2, 1],
    3: [3, 2, 1],
    4: [1, 2, 3],
    5: [1, 3, 2],
    6: [3, 1, 2],
    7: [2, 3, 1],
    8: [1, 3, 2],
    9: [2, 1, 3],
    10: [3, 1, 2],
    11: [2, 3, 1],
    12: [2, 1, 3],
    13: [2, 3, 1],
    14: [1, 3, 2],
    15: [1, 3, 2],
    16: [2, 3, 1],
    17: [3, 1, 2],
    18: [3, 2, 1]
    }

def main():
    print("CHS")
    full_filename = os.path.join(CHS_DIR, "ML_CH.csv")
    df = pd.read_csv(full_filename, sep=' ', dtype={'ATCO': 'int', 'Run': 'int',
                                                        'TimeInterval': 'int'})
    #"ATCO", "Run", "timeInterval", "score"
    df['WL'] = df['score'].apply(lambda x: 1 if x < 2 else 3 if x > 3 else 2)
  
    df['scenario'] = df.apply(lambda row: scenario_dict[row['ATCO']][int(row['Run'] - 1)], axis=1)
    
    df.to_csv("scores_scenario_3classes.csv", sep= ' ', header=True, index=False)
    
    # Calculate the percentage of high WL (WL = 3) for each scenario
    result = df.groupby('scenario')['WL'].apply(lambda x: (x == 3).sum() / len(x) * 100)

    high_df = result.reset_index(name='percent_high_WL')

    # Calculate the percentage of high WL (WL = 2) for each scenario
    result = df.groupby('scenario')['WL'].apply(lambda x: (x == 2).sum() / len(x) * 100)

    medium_df = result.reset_index(name='percent_medium_WL')

    # Calculate the percentage of low WL (WL = 1) for each scenario
    result = df.groupby('scenario')['WL'].apply(lambda x: (x == 1).sum() / len(x) * 100)

    low_df = result.reset_index(name='percent_low_WL')
    
    result_df = pd.merge(low_df, medium_df, on='scenario', how='inner')
    result_df = pd.merge(result_df, high_df, on='scenario', how='inner')

    result_df = result_df.round(2)
    
    # Adjust the medium column to ensure that the sum is 100%
    result_df['percent_medium_WL'] = result_df['percent_medium_WL'] + \
        (100 - result_df[['percent_low_WL', 'percent_medium_WL', 'percent_high_WL']].sum(axis=1))
    
    # Print the result
    print(result_df)

main()