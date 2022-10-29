import pandas as pd
import numpy as np
import pickle
from matplotlib import pyplot as plt
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE

import sys

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
                    if True in patient_col_data:
                        patient_dict[col].append( True )
                    else:
                        patient_dict[col].append( False )
                elif col == 'event':
                    patient_col_data = patient_col_data.tolist()

                    ## If there is an event then select it: 
                    ##    ['APNEA-CENTRAL' 'APNEA-MIXED' 'APNEA-OBSTRUCTIVE' 'HYPOPNEA' 'NONE']
                    if 'APNEA-CENTRAL' in patient_col_data:
                        patient_dict[col].append('APNEA-CENTRAL')
                    elif 'APNEA-MIXED' in patient_col_data:
                        patient_dict[col].append('APNEA-MIXED')
                    elif 'APNEA-OBSTRUCTIVE' in patient_col_data:
                        patient_dict[col].append('APNEA-OBSTRUCTIVE')
                    elif 'HYPOPNEA' in patient_col_data:
                        patient_dict[col].append('HYPOPNEA')
                    else:
                        patient_dict[col].append('NONE')
                elif col == 'timestamp_datetime':
                    ## Just take the time stamp of the first entry
                    patient_dict[col].append( patient_col_data[i] )
                elif col in ['HR(bpm)', 'SpO2(%)', 'PI(%)', 'RR(rpm)', 'PVCs(/min)']:
                    ## Easy mean across the 5 floats
                    patient_dict[col].append(np.mean(patient_col_data))
                else:
                    patient_dict[col].append(np.mean(patient_col_data).tolist())

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

    for i in range(6, 20, 2):
        print('Generating window size: ', i)
        a = do_sliding_window(data, i)

        a.to_pickle(f'./dataset_OSAS-window-{i}.pkl')
