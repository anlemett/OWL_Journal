import warnings
warnings.filterwarnings('ignore')

import os
import pandas as pd
import numpy as np
import math
import statistics
#from sklearn import preprocessing

DATA_DIR = os.path.join("..", "..")
DATA_DIR = os.path.join(DATA_DIR, "Data")
ET_DIR = os.path.join(DATA_DIR, "EyeTracking3")
CH_DIR = os.path.join(DATA_DIR, "CH1")
OUTPUT_DIR = os.path.join(DATA_DIR, "EyeTracking4")

#TIME_INTERVAL_DURATION = 180  #sec
TIME_INTERVAL_DURATION = 60  #sec

filenames = [["D1r1_MO", "D1r2_MO", "D1r3_MO"],
             ["D1r4_EI", "D1r5_EI", "D1r6_EI"],
             ["D2r1_KV", "D2r2_KV"           ],
             ["D2r4_UO", "D2r5_UO", "D2r6_UO"],
             ["D3r1_KB", "D3r2_KB", "D3r3_KB"],
             ["D3r4_PF", "D3r5_PF", "D3r6_PF"],
             ["D4r1_AL", "D4r2_AL", "D4r3_AL"],
             ["D4r4_IH", "D4r5_IH", "D4r6_IH"],
             ["D5r1_RI", "D5r2_RI", "D5r3_RI"],
             ["D5r4_JO", "D5r5_JO", "D5r6_JO"],
             ["D6r1_AE", "D6r2_AE", "D6r3_AE"],
             ["D6r4_HC", "D6r5_HC", "D6r6_HC"],
             ["D7r1_LS", "D7r2_LS", "D7r3_LS"],
             ["D7r4_ML", "D7r5_ML", "D7r6_ML"],
             ["D8r1_AP", "D8r2_AP", "D8r3_AP"],
             ["D8r4_AK", "D8r5_AK", "D8r6_AK"],
             ["D9r1_RE", "D9r2_RE", "D9r3_RE"],
             ["D9r4_SV", "D9r5_SV", "D9r6_SV"]
             ]

#filenames = [["D6r1_AE"]]

new_features = ['SaccadesNumber',
                'SaccadesDurationMean', 'SaccadesDurationStd', 'SaccadesDurationMedian',
                'SaccadesDurationQuantile25', 'SaccadesDurationQuantile75',
                'SaccadesDurationMin', 'SaccadesDurationMax',
                'FixationDurationMean', 'FixationDurationStd', 'FixationDurationMedian',
                'FixationDurationQuantile25', 'FixationDurationQuantile75',
                'FixationDurationMin', 'FixationDurationMax',
                'BlinksNumber',
                'BlinksDurationMean', 'BlinksDurationStd', 'BlinksDurationMedian',
                'BlinksDurationQuantile25', 'BlinksDurationQuantile75',
                'BlinksDurationMin', 'BlinksDurationMax',
                'PupilDiameter', 'LeftPupilDiameter', 'RightPupilDiameter',
                'LeftBlinkClosingAmplitude', 'LeftBlinkOpeningAmplitude',
                'LeftBlinkClosingSpeed', 'LeftBlinkOpeningSpeed',
                'RightBlinkClosingAmplitude', 'RightBlinkOpeningAmplitude',
                'RightBlinkClosingSpeed', 'RightBlinkOpeningSpeed',
                'HeadHeading', 'HeadPitch', 'HeadRoll']


def getTimeInterval(timestamp, ch_first_timestamp, ch_last_timestamp):

    if timestamp < ch_first_timestamp:
        return 0
    if timestamp >= ch_last_timestamp:
        return -1
    return math.trunc((timestamp - ch_first_timestamp)/TIME_INTERVAL_DURATION) + 1

def timeSynch(timestamp, atco_num, run):
    
    #11/21/2023    -13
    #11/24/2023    -10
    #11/28/2023    -8
    #11/29/2023    -7
    #11/30/2023    -6
    #12/5/2023     -1
    #12/6/2023     0
    #12/7/2023     1      until run 44
    #12/7/2023     -24    run 45+
    #12/15/2023    -34
    
    new_timestamp = timestamp
    
    if atco_num==1 or atco_num==2:
        new_timestamp = new_timestamp - 13
    elif atco_num==3 or atco_num==4:
        new_timestamp = new_timestamp - 10
    elif atco_num==5 or atco_num==6:
        new_timestamp = new_timestamp - 8
    elif atco_num==7 or atco_num==8:
        new_timestamp = new_timestamp - 7
    elif atco_num==9 or atco_num==10:
        new_timestamp = new_timestamp - 6
    elif atco_num==11 or atco_num==12:
        new_timestamp = new_timestamp - 1
    elif atco_num==13 or atco_num==14:
        new_timestamp = new_timestamp - 0
    elif atco_num==15:
        if run < 3:
            new_timestamp = new_timestamp + 1
        else:
            new_timestamp = new_timestamp - 24
    elif atco_num==16:
        new_timestamp = new_timestamp - 24
    else:
        new_timestamp = new_timestamp - 34

    return new_timestamp


TI_df = pd.DataFrame()

atco_num = 0

for atco in filenames:
    
    atco_num = atco_num + 1
    
    atco_df = pd.DataFrame()
    run = 1
    for filename in atco:
        print(filename)
        full_filename = os.path.join(ET_DIR, 'ET_' + filename +  ".csv")
        df = pd.read_csv(full_filename, sep=' ')
        
        #negative_count = df['LeftBlinkOpeningAmplitude'].lt(0).sum()
        #print(negative_count)
        
        # adjust ET timestamps (time synchronization)
        df['UnixTimestamp'] = df.apply(lambda row: timeSynch(row['UnixTimestamp'],
                                                             atco_num,
                                                             run),
                                       axis=1)
                       
        first_timestamp = df['UnixTimestamp'].tolist()[0]
                      
        full_filename = os.path.join(CH_DIR, filename + ".csv")
        scores_df = pd.read_csv(full_filename, sep=' ')
        
        ch_timestamps = scores_df['timestamp'].tolist()
        ch_first_timestamp = ch_timestamps[0]
        
        dif = first_timestamp - ch_first_timestamp
        if dif>0:
            ch_first_timestamp = first_timestamp
            
        number_of_ch_timestamps = len(ch_timestamps)
        ch_last_timestamp = ch_first_timestamp + 180*(number_of_ch_timestamps-1)
        
        df['timeInterval'] = df.apply(lambda row: getTimeInterval(row['UnixTimestamp'],
                                                                  ch_first_timestamp,
                                                                  ch_last_timestamp
                                                                  ),
                                      axis=1) 
                       
        df = df[df['timeInterval']!=0]
        df = df[df['timeInterval']!=-1]
        
        timeIntervals = set(df['timeInterval'].tolist())
        number_of_time_intervals = len(timeIntervals)
        
        print("Number of time intervals: ")
        print(number_of_time_intervals)
                        
        SaccadesNumber = []
        SaccadesDurationMean = []
        SaccadesDurationStd = []
        SaccadesDurationMedian = []
        SaccadesDurationQuantile25 = []
        SaccadesDurationQuantile75 = []
        SaccadesDurationMin = []
        SaccadesDurationMax = []

        FixationDurationMean = []
        FixationDurationStd = []
        FixationDurationMedian = []
        FixationDurationQuantile25 = []
        FixationDurationQuantile75 = []
        FixationDurationMin = []
        FixationDurationMax = []
        
        BlinksNumber = []
        BlinksDurationMean = []
        BlinksDurationStd = []
        BlinksDurationMedian = []
        BlinksDurationQuantile25 = []
        BlinksDurationQuantile75 = []
        BlinksDurationMin = []
        BlinksDurationMax = []


        # Saccade and Fixation can not be 0 simultaneously
        df.loc[(df['Saccade'] == 0) & (df['Fixation'] == 0), ['Saccade', 'Fixation']] = np.nan
        
        # Percent of missing Saccade values
        nan_percentage = (df['Saccade'].isna().sum() / len(df['Saccade'])) * 100
        print(f"Percentage of NaNs in Saccade: {nan_percentage:.2f}%")

        # Percent of missing Fixation values
        nan_percentage = (df['Fixation'].isna().sum() / len(df['Fixation'])) * 100
        print(f"Percentage of NaNs in Fixation: {nan_percentage:.2f}%")
        
        # Percent of missing Blink values
        nan_percentage = (df['Blink'].isna().sum() / len(df['Blink'])) * 100
        print(f"Percentage of NaNs in Blink: {nan_percentage:.2f}%")


        #Add Saccade number, total duration and duration stats per period
        for ti in range(1, number_of_time_intervals+1):
            ti_df = df[df['timeInterval']==ti]
            
            if ti_df.empty: # should not be the case
                continue
            
            nona_ti_saccade_df = ti_df.dropna(subset=['Saccade'])
            
            if nona_ti_saccade_df.empty: # incomplete data (data loss)
                # set to NA, then use linear interpolation over the column
                saccades_number = np.nan
                saccades_duration_mean = np.nan
                saccades_duration_std = np.nan
                saccades_duration_median = np.nan
                saccades_duration_quantile25 = np.nan
                saccades_duration_quantile75 = np.nan
                saccades_duration_min = np.nan
                saccades_duration_max = np.nan
            else:
                ti_saccades_df = nona_ti_saccade_df[nona_ti_saccade_df['Saccade']!=0]
                saccades_set = set(ti_saccades_df['Saccade'].tolist())
                saccades_number = len(saccades_set)
                saccades_duration = []
                for saccade in saccades_set:
                    saccade_df = ti_df[ti_df['Saccade']==saccade]
                    if not saccade_df.empty:
                        saccades_duration.append(len(saccade_df.index))
                    
                    saccades_duration_mean = statistics.mean(saccades_duration)
                    saccades_duration_std = statistics.stdev(saccades_duration) if len(saccades_duration)>1 else 0
                    saccades_duration_median = statistics.median(saccades_duration)
                    first_el = saccades_duration[0]
                    quantiles = statistics.quantiles(saccades_duration) if len(saccades_duration)>1 else [first_el]*3
                    saccades_duration_quantile25 = quantiles[0]
                    saccades_duration_quantile75 = quantiles[2]
                    saccades_duration_min = min(saccades_duration)
                    saccades_duration_max = max(saccades_duration)


            nona_ti_fixation_df = ti_df.dropna(subset=['Fixation'])

            if nona_ti_fixation_df.empty: # incomplete data (data loss)
                # set to NA, then use linear interpolation over the column
                fixation_number = np.nan
                fixation_duration_mean = np.nan
                fixation_duration_std = np.nan
                fixation_duration_median = np.nan
                fixation_duration_quantile25 = np.nan
                fixation_duration_quantile75 = np.nan
                fixation_duration_min = np.nan
                fixation_duration_max = np.nan

            else:
                ti_fixation_df = nona_ti_fixation_df[nona_ti_fixation_df['Fixation']!=0]
                fixation_set = set(ti_fixation_df['Fixation'].tolist())
                fixation_duration = []
                for fixation in fixation_set:
                    fixation_df = ti_df[ti_df['Fixation']==fixation]
                    if not fixation_df.empty:
                        fixation_duration.append(len(fixation_df.index))
                    
                fixation_duration_mean = statistics.mean(fixation_duration)
                fixation_duration_std = statistics.stdev(fixation_duration) if len(fixation_duration)>1 else 0
                fixation_duration_median = statistics.median(fixation_duration)
                first_el = fixation_duration[0]
                quantiles = statistics.quantiles(fixation_duration) if len(fixation_duration)>1 else [first_el]*3
                fixation_duration_quantile25 = quantiles[0]
                fixation_duration_quantile75 = quantiles[2]
                fixation_duration_min = min(fixation_duration)
                fixation_duration_max = max(fixation_duration)


            nona_ti_blink_df = ti_df.dropna(subset=['Blink'])

            if nona_ti_blink_df.empty: # possible if time interval is small
                blinks_number = 0
                blinks_duration_mean = 0
                blinks_duration_std = 0
                blinks_duration_median = 0
                blinks_duration_quantile25 = 0
                blinks_duration_quantile75 = 0
                blinks_duration_min = 0
                blinks_duration_max = 0

            else:
                ti_blinks_df = nona_ti_blink_df[nona_ti_blink_df['Blink']!=0]
                
                if ti_blinks_df.empty: # possible
                    blinks_number = 0
                    blinks_duration_mean = 0
                    blinks_duration_std = 0
                    blinks_duration_median = 0
                    blinks_duration_quantile25 = 0
                    blinks_duration_quantile75 = 0
                    blinks_duration_min = 0
                    blinks_duration_max = 0
                else:
                    blinks_set = set(ti_blinks_df['Blink'].tolist())
                    blinks_number = len(blinks_set)
                    blinks_duration = []
                    
                    for blink in blinks_set:
                        blink_df = ti_df[ti_df['Blink']==blink]
                        if not blink_df.empty:
                            blinks_duration.append(len(blink_df.index))
                    
                    blinks_duration_mean = statistics.mean(blinks_duration)
                    blinks_duration_std = statistics.stdev(blinks_duration) if len(blinks_duration)>1 else 0
                    blinks_duration_median = statistics.median(blinks_duration)
                    first_el = blinks_duration[0]
                    quantiles = statistics.quantiles(blinks_duration) if len(blinks_duration)>1 else [first_el]*3
                    blinks_duration_quantile25 = quantiles[0]
                    blinks_duration_quantile75 = quantiles[2]
                    blinks_duration_min = min(blinks_duration)
                    blinks_duration_max = max(blinks_duration)
            
            SaccadesNumber.extend([saccades_number]*TIME_INTERVAL_DURATION*250)
            SaccadesDurationMean.extend([saccades_duration_mean]*TIME_INTERVAL_DURATION*250)
            SaccadesDurationStd.extend([saccades_duration_std]*TIME_INTERVAL_DURATION*250)
            SaccadesDurationMedian.extend([saccades_duration_median]*TIME_INTERVAL_DURATION*250)
            SaccadesDurationQuantile25.extend([saccades_duration_quantile25]*TIME_INTERVAL_DURATION*250)
            SaccadesDurationQuantile75.extend([saccades_duration_quantile25]*TIME_INTERVAL_DURATION*250)
            SaccadesDurationMin.extend([saccades_duration_min]*TIME_INTERVAL_DURATION*250)
            SaccadesDurationMax.extend([saccades_duration_max]*TIME_INTERVAL_DURATION*250)
            
            FixationDurationMean.extend([fixation_duration_mean]*TIME_INTERVAL_DURATION*250)
            FixationDurationStd.extend([fixation_duration_std]*TIME_INTERVAL_DURATION*250)
            FixationDurationMedian.extend([fixation_duration_median]*TIME_INTERVAL_DURATION*250)
            FixationDurationQuantile25.extend([fixation_duration_quantile25]*TIME_INTERVAL_DURATION*250)
            FixationDurationQuantile75.extend([fixation_duration_quantile25]*TIME_INTERVAL_DURATION*250)
            FixationDurationMin.extend([fixation_duration_min]*TIME_INTERVAL_DURATION*250)
            FixationDurationMax.extend([fixation_duration_max]*TIME_INTERVAL_DURATION*250)
            
            BlinksNumber.extend([blinks_number]*TIME_INTERVAL_DURATION*250)
            BlinksDurationMean.extend([blinks_duration_mean]*TIME_INTERVAL_DURATION*250)
            BlinksDurationStd.extend([blinks_duration_std]*TIME_INTERVAL_DURATION*250)
            BlinksDurationMedian.extend([blinks_duration_median]*TIME_INTERVAL_DURATION*250)
            BlinksDurationQuantile25.extend([blinks_duration_quantile25]*TIME_INTERVAL_DURATION*250)
            BlinksDurationQuantile75.extend([blinks_duration_quantile25]*TIME_INTERVAL_DURATION*250)
            BlinksDurationMin.extend([blinks_duration_min]*TIME_INTERVAL_DURATION*250)
            BlinksDurationMax.extend([blinks_duration_max]*TIME_INTERVAL_DURATION*250)
        
        
        df['SaccadesNumber'] = SaccadesNumber
        df['SaccadesDurationMean'] = SaccadesDurationMean
        df['SaccadesDurationStd'] = SaccadesDurationStd
        df['SaccadesDurationMedian'] = SaccadesDurationMedian
        df['SaccadesDurationQuantile25'] = SaccadesDurationQuantile25
        df['SaccadesDurationQuantile75'] = SaccadesDurationQuantile75
        df['SaccadesDurationMin'] = SaccadesDurationMin
        df['SaccadesDurationMax'] = SaccadesDurationMax
        
        df['FixationDurationMean'] = FixationDurationMean
        df['FixationDurationStd'] = FixationDurationStd
        df['FixationDurationMedian'] = FixationDurationMedian
        df['FixationDurationQuantile25'] = FixationDurationQuantile25
        df['FixationDurationQuantile75'] = FixationDurationQuantile75
        df['FixationDurationMin'] = FixationDurationMin
        df['FixationDurationMax'] = FixationDurationMax
        
        df['BlinksNumber'] = BlinksNumber
        df['BlinksDurationMean'] = BlinksDurationMean
        df['BlinksDurationStd'] = BlinksDurationStd
        df['BlinksDurationMedian'] = BlinksDurationMedian
        df['BlinksDurationQuantile25'] = BlinksDurationQuantile25
        df['BlinksDurationQuantile75'] = BlinksDurationQuantile75
        df['BlinksDurationMin'] = BlinksDurationMin
        df['BlinksDurationMax'] = BlinksDurationMax

        df = df.drop('Saccade', axis=1)
        df = df.drop('Fixation', axis=1)
        df = df.drop('Blink', axis=1)
        
        # Fill NaN values: linear interpolation of respective columns
        # (stat. summary features of Saccade, Fixation & Blinks)
        # All other columns are processed on the previous step (so, no NaNs)
        df.interpolate(method='linear', limit_direction='both', axis=0, inplace=True)
        
        row_num = len(df.index)
        #df['ATCO'] = [filename[-2:]] * row_num
        df['ATCO'] = [atco_num] * row_num
        df['Run'] = [run] * row_num
        run = run + 1    

        columns = ['ATCO'] + ['Run'] + ['timeInterval'] + ['UnixTimestamp'] + \
            ['SamplePerSecond'] + new_features
        df = df[columns]
        
        atco_df = pd.concat([atco_df, df], ignore_index=True)
    
    #####################################
    # Normalization per ATCO 
    # might cause data leakage
    '''
    scaler = preprocessing.MinMaxScaler()

    for feature in new_features:
        feature_lst = atco_df[feature].tolist()
        scaled_feature_lst = scaler.fit_transform(np.asarray(feature_lst).reshape(-1, 1))
        atco_df = atco_df.drop(feature, axis = 1)
        atco_df[feature] = scaled_feature_lst
    '''
    #####################################
    
    TI_df = pd.concat([TI_df, atco_df], ignore_index=True)

#print(TI_df.isnull().any().any())
#nan_count = TI_df.isna().sum()
#print(nan_count)

pd.set_option('display.max_columns', None)
#print(TI_df.head(1))

#negative_count = TI_df['LeftBlinkOpeningAmplitude'].lt(0).sum()
#print(negative_count)

full_filename = os.path.join(OUTPUT_DIR, "ET_all_" + str(TIME_INTERVAL_DURATION) + ".csv")
TI_df.to_csv(full_filename, sep=' ', encoding='utf-8', index = False, header = True)
