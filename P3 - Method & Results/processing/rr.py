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
    
    rr = patient['RR(rpm)'].to_numpy()
    #hr = patient['HR(bpm)'].to_numpy()
    #spO2 = patient['SpO2(%)'].to_numpy()
    #pvcs = patient['PVCs(/min)'].to_numpy()
    #ecg = patient['signal_ecg_i'].to_numpy()
    
    labels = np.array( list( map( lambda x: 1 if x.__contains__('APNEA') else 0, patient['event'].to_list())))
    for i in range(len(rr)):
    	if (np.isnan(rr[i])):
    		continue
    	X.append([rr[i]])#, hr[i], spO2[i], pvcs[i], np.std(ecg[i])])
    	y.append(labels[i])

X = StandardScaler().fit_transform(X)
oversample = SMOTE()
X, y = oversample.fit_resample(X, y)

X_t = []
y_t = []
for pid in testing_patients:
    patient = data[data['patient'] == pid]
    
    rr = patient['RR(rpm)'].to_numpy()
    #hr = patient['HR(bpm)'].to_numpy()
    #spO2 = patient['SpO2(%)'].to_numpy()
    #pvcs = patient['PVCs(/min)'].to_numpy()
    #ecg = patient['signal_ecg_i'].to_numpy()
    
    labels = np.array( list( map( lambda x: 1 if x.__contains__('APNEA') else 0, patient['event'].to_list())))
    
    for i in range(len(rr)):
    	if (np.isnan(rr[i])):
    		continue
    	X_t.append([rr[i]])#, hr[i], spO2[i], pvcs[i], np.std(ecg[i])])
    	y_t.append(labels[i])

X_t = StandardScaler().fit_transform(X_t)

print("w6 rr")

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
