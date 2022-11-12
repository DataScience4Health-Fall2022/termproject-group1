import pandas as pd
import numpy as np
import pickle
from matplotlib import pyplot as plt
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE
from scipy.signal import find_peaks

import sys

''' The NN50 measure (variant 1), defined as the number
of pairs of adjacent RR- intervals where the first RR-
interval exceeds the second RR- interval by more than
50 ms.

The NN50 measure (variant 2), defined as the number
of pairs of adjacent RR-intervals where the second RR-
interval exceeds the first RR interval by more than 50
ms.
## Each second has 80 samples so 50ms corresponds to 4 samples.
'''
def count_nn50( intervals ):
    var_1_count = 0
    var_2_count = 0

    for i in range(len(intervals) - 1):
        if ( intervals[i] - intervals[i+1] > 4 ):
            var_1_count += 1
        if ( intervals[i+1] - intervals[i] > 4 ):
            var_1_count += 1

    pNN50_1 = var_1_count / len(intervals)
    pNN50_2 = var_2_count / len(intervals)

    return (var_1_count, var_2_count, pNN50_1, pNN50_2)

'''
The RMSD measures, defined as the square root of the
mean of the sum of the squares of differences between
adjacent RR- intervals.
'''
def rmsd( intervals ):
    deltas = np.diff(intervals)

    if ( len(deltas) == 0 ):
        print("WHY NO DELTAS!")
        exit(0)

    return np.sqrt( np.sum( np.mean(deltas **2) ))

def extract_ecg_features( col, ecg_data ):
    features = {}
    ## find the peak indices
    indices = find_peaks(ecg_data, threshold=2.2)[0]

    if ( len(ecg_data) == 0 ):
        print("WHY ECG 0???")
        exit(0)

    intervals = np.diff(indices)

    # Probably because all nans but need fake data
    if ( len(intervals) < 3 ):
        features[col + '_indices'] = []
        features[col + '_intervals'] = []
        features[col + '_std'] = np.nan
        features[col + '_NN50_1'] = np.nan
        features[col + '_NN50_2'] = np.nan
        features[col + '_pNN50_1'] = np.nan
        features[col + '_pNN50_2'] = np.nan
        features[col + '_median'] = np.nan
        features[col + '_rmsd'] = np.nan
        features[col + '_iqr'] = np.nan

        return features

    

    features[col + '_indices'] = list(indices)
    features[col + '_intervals'] = list(intervals)
    features[col + '_std'] = np.std(intervals)
    NN50_1, NN50_2, pNN50_1, pNN50_2 = count_nn50(intervals)

    features[col + '_NN50_1'] = NN50_1
    features[col + '_NN50_2'] = NN50_2
    features[col + '_pNN50_1'] = pNN50_1
    features[col + '_pNN50_2'] = pNN50_2
    features[col + '_median'] = np.median(intervals)

    features[col + '_rmsd'] = rmsd(intervals)

    q75, q25 = np.percentile(intervals, [75 ,25])
    iqr = q75 - q25
    
    features[col + '_iqr'] = iqr

    return features

def do_sliding_window( dataset, windowsize, overlap=.5 ):
    patient_dict = {}

    ## Get the list of patients
    patients = np.unique(dataset['patient'])

    ## Loop through each patient
    for patient_id in patients:
        print('\tPatient id: ', patient_id)
        ## Isolate the patient data
        patient_data = dataset[ dataset['patient'] == patient_id]

        ## Rebase the index to 0. Drop the index column
        patient_data = patient_data.reset_index(drop=True)

        ## Loop through each entry to do the window with 50% overlap
        for i in range(0, len(patient_data) - windowsize, int(windowsize * overlap) ):

            ## Loop through each column in the window
            for col in patient_data.columns:
                ## Skip some
                if col in ['signal_pleth', 'PI(%)', 'event', 'timestamp_datetime', 'PSG_Flow', 'PSG_Snore', 'PSG_Thorax', 'PSG_Abdomen', 'PSG_Position', 'HR(bpm)', 'SpO2(%)', 'RR(rpm)', 'PVCs(/min)']:
                    continue

                ## Add this column to the patient dictionary
                if col not in patient_dict:
                    patient_dict[col] = []

                ## Get the data for this specific column
                patient_col_data = patient_data[ col ]
                patient_col_data = patient_col_data[i: i + windowsize]

                ## If the column is just the patient id then just append it, mean doesn't make sense
                if col == 'patient':
                    patient_dict[col].append(patient_col_data[i])
                elif col == 'anomaly':
                    ## If any anomaly occurs in the window then it is considered true
                    if any(val == True for val in patient_col_data):
                        patient_dict[col].append( True )
                    else:
                        patient_dict[col].append( False )
                elif col in ['signal_ecg_i', 'signal_ecg_ii', 'signal_ecg_iii']:
                    patient_dict[col].append(np.mean(patient_col_data).tolist())

                    stuff = list(patient_col_data)

                    flatter = np.array(stuff).flatten( )
                    if len(flatter) == 0:
                        print('WHY flatter 0')
                        print(stuff)
                        exit(0)

                    features = extract_ecg_features(col, flatter)

                    for key in features:
                        if key not in patient_dict:
                            patient_dict[key] = []
                        
                        patient_dict[key].append(features[key])

                else:
                    print("WHY NO HANDLE COL: %s" %(col))

    ## Convert the dict to a pandas dataframe
    return pd.DataFrame.from_dict(patient_dict)

def remove_nans(data):
    rows_to_delete = []
    ## loop through each entry and if there is a nan remove it

    for index in range(0, len(data)):
        if np.isnan(data['PI(%)'][index]):
            rows_to_delete.append(index)
            continue

        if np.isnan(data['RR(rpm)'][index]) or np.isnan(data['HR(bpm)'][index]) or np.isnan(data['PVCs(/min)'][index]):
           rows_to_delete.append(index)
           continue

        if np.isnan(data['SpO2(%)'][index]): #15635
            rows_to_delete.append(index)
            continue

        if np.isnan(np.sum(data['signal_pleth'][index])): # 16029
            rows_to_delete.append(index)
            continue

        if np.isnan(np.sum(data['signal_ecg_i'][index])): #99
            rows_to_delete.append(index)
            continue

        if np.isnan(np.sum(data['signal_ecg_ii'][index])): #408
            rows_to_delete.append(index)
            continue

        if np.isnan(np.sum(data['signal_ecg_iii'][index])):#2542
            rows_to_delete.append(index)
            continue

        if np.isnan(np.sum(data['PSG_Abdomen'][index])):
            rows_to_delete.append(index)
            continue

        if np.isnan(np.sum(data['PSG_Flow'][index])):
            rows_to_delete.append(index)
            continue

        if np.isnan(np.sum(data['PSG_Position'][index])):
            rows_to_delete.append(index)
            continue

        if np.isnan(np.sum(data['PSG_Snore'][index])):
            rows_to_delete.append(index)
            continue

        if np.isnan(np.sum(data['PSG_Thorax'][index])):
            rows_to_delete.append(index)
            continue

    print('\tNaNs Total: %d Removed: %d Percentage: %f' %(len(data), len(rows_to_delete), len(rows_to_delete)/len(data)))
    data = data.drop(data.index[rows_to_delete])
    data = data.reset_index()
    
    return data

if __name__ == '__main__':
    f = open('./dataset_OSAS.pickle', 'rb')
    data = pickle.load(f)
    f.close()

    for i in [20]:
        print('Generating window size: ', i)
        a = do_sliding_window(data, i)

        a.to_pickle(f'./dataset_OSAS-window-{i}.pkl')
