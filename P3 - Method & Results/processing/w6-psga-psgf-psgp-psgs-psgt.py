import pandas as pd
import numpy as np
import pickle
from matplotlib import pyplot as plt
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE

f = open('./dataset_OSAS-window-6.pkl', 'rb')
data = pickle.load(f)                                
f.close()

patient_ids = np.unique(data['patient'])
patient_count = len(patient_ids)

training_count = int(patient_count * .8)
testing_count = int(patient_count * .2)

assert( (training_count + testing_count) == patient_count)

## Break out the patients
training_patients = patient_ids[:training_count]
testing_patients = patient_ids[training_count:]

training_data_set = np.array([])

X = []
y = []
cnt = 0
for pid in training_patients:
    patient = data[data['patient'] == pid]
    
    #rr = patient['RR(rpm)'].to_numpy()
    #hr = patient['HR(bpm)'].to_numpy()
    #spO2 = patient['SpO2(%)'].to_numpy()
    #pvcs = patient['PVCs(/min)'].to_numpy()
    #ecg = patient['signal_ecg_i'].to_numpy()
    psga = patient['PSG_Abdomen'].to_numpy()
    psgf = patient['PSG_Flow'].to_numpy()
    psgp = patient['PSG_Position'].to_numpy()
    psgs = patient['PSG_Snore'].to_numpy()
    psgt = patient['PSG_Thorax'].to_numpy()

    labels = np.array( list( map( lambda x: 1 if x.__contains__('APNEA') else 0, patient['event'].to_list())))
    for i in range(len(psga)):
        psgastd = np.std(psga[i])
        psgfstd = np.std(psgf[i])
        psgpstd = np.std(psgp[i])
        psgsstd = np.std(psgs[i])
        psgtstd = np.std(psgt[i])

        if (np.isnan(psgastd) or np.isnan(psgfstd) or np.isnan(psgpstd) or np.isnan(psgsstd) or np.isnan(psgtstd)):
            continue
        X.append([psgastd, psgfstd, psgpstd, psgsstd, psgtstd])
        y.append(labels[i])

X = StandardScaler().fit_transform(X)
oversample = SMOTE()
X, y = oversample.fit_resample(X, y)

X_t = []
y_t = []
for pid in testing_patients:
    patient = data[data['patient'] == pid]
    
    #rr = patient['RR(rpm)'].to_numpy()
    #hr = patient['HR(bpm)'].to_numpy()
    #spO2 = patient['SpO2(%)'].to_numpy()
    #pvcs = patient['PVCs(/min)'].to_numpy()
    #ecg = patient['signal_ecg_i'].to_numpy()
    #psgflow = patient['PSG_Flow'].to_numpy()
    
    psga = patient['PSG_Abdomen'].to_numpy()
    psgf = patient['PSG_Flow'].to_numpy()
    psgp = patient['PSG_Position'].to_numpy()
    psgs = patient['PSG_Snore'].to_numpy()
    psgt = patient['PSG_Thorax'].to_numpy()
    labels = np.array( list( map( lambda x: 1 if x.__contains__('APNEA') else 0, patient['event'].to_list())))
    
    for i in range(len(psga)):
        psgastd = np.std(psga[i])
        psgfstd = np.std(psgf[i])
        psgpstd = np.std(psgp[i])
        psgsstd = np.std(psgs[i])
        psgtstd = np.std(psgt[i])

        if (np.isnan(psgastd) or np.isnan(psgfstd) or np.isnan(psgpstd) or np.isnan(psgsstd) or np.isnan(psgtstd)):
            continue
        X_t.append([psgastd, psgfstd, psgpstd, psgsstd, psgtstd])
        y_t.append(labels[i])

X_t = StandardScaler().fit_transform(X_t)

print("allpsg std")

import time
a=time.time()
clf = svm.SVC(C=10, kernel='linear')
clf.fit(X, y)
b = time.time()
print(( b - a ) / 60)

tp = 0
fp = 0
tf = 0
ff = 0

Y_pred = clf.predict(X_t)
report = classification_report(y_t, Y_pred)
print(report)

for i in range(len(X_t)):
    yh = clf.predict([X_t[i]])
    if yh:
        if y_t[i]:
            tp += 1
        else:
            fp += 1
    else:
        if y_t[i]:
            ff += 1
        else:
            tf += 1

print(tp, fp, tf, ff)
