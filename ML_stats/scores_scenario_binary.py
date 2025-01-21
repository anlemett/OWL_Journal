# Proportion of EEG high/low workload in light, moderate and heavy scenarios
import os
import numpy as np
import pandas as pd

DATA_DIR = os.path.join("..", "..")
DATA_DIR = os.path.join(DATA_DIR, "Data")
EEG_DIR = os.path.join(DATA_DIR, "EEG2")
CHS_DIR = os.path.join(DATA_DIR, "MLInput")
CHS = False
TIME_INTERVAL_DURATION = 180

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
    if CHS:
        print("CHS")
        full_filename = os.path.join(CHS_DIR, "ML_CH.csv")
        df = pd.read_csv(full_filename, sep=' ', dtype={'ATCO': 'int', 'Run': 'int',
                                                        'TimeInterval': 'int'})
        #"ATCO", "Run", "timeInterval", "score"
        df['WL'] = df['score'].apply(lambda x: 1 if x < 4 else 2)
    else:
        print("EEG")
        full_filename = os.path.join(EEG_DIR, "EEG_all_" + str(TIME_INTERVAL_DURATION) + ".csv")

        df = pd.read_csv(full_filename, sep=' ', dtype={'ATCO': 'int', 'Run': 'int',
                                                        'TimeInterval': 'int'})
        print(df.head(1))
        df = df[["ATCO", "Run", "timeInterval", "WorkloadMean"]]
        df = df.rename(columns={'WorkloadMean': 'score'})
        df['score'] = df['score'].interpolate(method='linear')
        
        df['score_norm'] = df.groupby('ATCO')['score'].transform(lambda x: (x - x.min()) / (x.max() - x.min()))

        #threshold = df['score'].quantile(0.93)
        #df['WL'] = df['score'].apply(lambda x: 1 if x < threshold else 2)
        threshold = df['score_norm'].quantile(0.93)
        df['WL'] = df['score_norm'].apply(lambda x: 1 if x < threshold else 2)
    
    df['scenario'] = df.apply(lambda row: scenario_dict[row['ATCO']][int(row['Run'] - 1)], axis=1)
    
    df.to_csv("scores_scenario_binary.csv", sep= ' ', header=True, index=False)
    
    # Calculate the percentage of high WL (WL = 2) for each scenario
    result = df.groupby('scenario')['WL'].apply(lambda x: (x == 2).sum() / len(x) * 100)

    high_df = result.reset_index(name='percent_high_WL')

    # Calculate the percentage of low WL (WL = 1) for each scenario
    result = df.groupby('scenario')['WL'].apply(lambda x: (x == 1).sum() / len(x) * 100)

    low_df = result.reset_index(name='percent_low_WL')
    
    result_df = pd.merge(low_df, high_df, on='scenario')

    print(result_df)


main()