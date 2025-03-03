import warnings
warnings.filterwarnings('ignore')

import os

import pandas as pd

DATA_DIR = os.path.join("..", "..")
DATA_DIR = os.path.join(DATA_DIR, "Data")
INPUT_DIR = os.path.join(DATA_DIR, "EyeTracking1")
OUTPUT_DIR = os.path.join(DATA_DIR, "EyeTracking2")

metrics_list = ['Saccade', 'Fixation', 'Blink',
                'PupilDiameter', 'LeftPupilDiameter', 'RightPupilDiameter',
                'LeftBlinkClosingAmplitude', 'LeftBlinkOpeningAmplitude',
                'LeftBlinkClosingSpeed', 'LeftBlinkOpeningSpeed',
                'RightBlinkClosingAmplitude', 'RightBlinkOpeningAmplitude',
                'RightBlinkClosingSpeed', 'RightBlinkOpeningSpeed',
                'HeadHeading', 'HeadPitch',	'HeadRoll']

def getValuesPerSecond(timestamps_dict, unix_timestamp):
    return timestamps_dict[unix_timestamp]

filenames = [
             "D1r1", "D1r2", "D1r3",
             "D1r4", "D1r5", "D1r6",
             "D2r1", "D2r2",
             "D2r4", "D2r5", "D2r6",
             "D3r1", "D3r2", "D3r3",
             "D3r4", "D3r5", "D3r6",
             "D4r1", "D4r2", "D4r3",
             "D4r4", "D4r5", "D4r6",
             "D5r1", "D5r2", "D5r3",
             "D5r4", "D5r5", "D5r6",
             "D6r1", "D6r2", "D6r3",
             "D6r4", "D6r5", "D6r6",
             "D7r1", "D7r2", "D7r3",
             "D7r4", "D7r5", "D7r6",
             "D8r1", "D8r2", "D8r3",
             "D8r4", "D8r5", "D8r6",
             "D9r1", "D9r2", "D9r3",
             "D9r4", "D9r5", "D9r6"
             ]

print("Start")
for filename in filenames:
    
    print(filename)
    full_filename = os.path.join(INPUT_DIR, "ET_" + filename +  ".csv")
    df = pd.read_csv(full_filename, sep=' ')

    timestamps_lst = df['UnixTimestamp'].to_list()
    timestamps_dict = {i:timestamps_lst.count(i) for i in timestamps_lst}
    #Results: most of the values are 250 (sometimes 249)
    #         first and last might be anything (<=250)
    #         some seconds are missig and thus the neighbor values are also anything <=250


    df['ValuesPerSecond'] = df.apply(lambda row: getValuesPerSecond(timestamps_dict,
                                                                    row['UnixTimestamp']), axis=1)


    columns = ['UnixTimestamp'] + ['ValuesPerSecond'] + metrics_list
    df = df[columns]

    full_filename = os.path.join(OUTPUT_DIR, "ET_" + filename +  ".csv")
    df.to_csv(full_filename, sep=' ', encoding='utf-8', index = False, header = True)
