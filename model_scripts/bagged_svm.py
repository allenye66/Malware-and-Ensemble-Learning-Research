import pandas as pd
import numpy as np
import sklearn.metrics as metrics
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels
from sklearn.svm import LinearSVC


df = pd.read_csv('/Users/allen/Desktop/Malware-Research/csv/all_data.csv')

df.Family = df.Family.replace({"ADLOAD": 0})
df.Family = df.Family.replace({"AGENT": 1})
df.Family = df.Family.replace({"ALLAPLE_A": 2})
df.Family = df.Family.replace({"BHO": 3})
df.Family = df.Family.replace({"BIFROSE": 4})
df.Family = df.Family.replace({"CEEINJECT": 5})
df.Family = df.Family.replace({"CYCBOT_G": 6})
df.Family = df.Family.replace({"FAKEREAN": 7})
df.Family = df.Family.replace({"HOTBAR": 8})
df.Family = df.Family.replace({"INJECTOR": 9})

df.Family = df.Family.replace({"LOLYDA_BF": 10})
df.Family = df.Family.replace({"ONLINEGAMES": 11})
df.Family = df.Family.replace({"RENOS": 12})
df.Family = df.Family.replace({"RIMECUD_A": 13})
df.Family = df.Family.replace({"SMALL": 14})
df.Family = df.Family.replace({"STARTPAGE": 15})
df.Family = df.Family.replace({"TOGA_RFN": 16})
df.Family = df.Family.replace({"VB": 17})
df.Family = df.Family.replace({"VBINJECT": 18})
df.Family = df.Family.replace({"VOBFUS": 19})

df.Family = df.Family.replace({"VUNDO": 20})
df.Family = df.Family.replace({"WINTRIM_BX": 21})
df.Family = df.Family.replace({"WINWEBSEC": 22})
df.Family = df.Family.replace({"ZBOT": 23})

df = df.loc[:, df.columns != 'Total Opcodes']
df = df.loc[:, df.columns != 'File Name']

for i in range(31):
    df = df.drop(df.columns[1], axis=1)



opcode_sequence = (df.drop(df.columns[0], axis=1))
opcode_sequence = np.asarray(opcode_sequence)


labels = np.asarray(df[['Family']].copy())



X_train, X_test, y_train, y_test = train_test_split(opcode_sequence, labels, test_size=0.20, random_state=42)




from sklearn import svm
from sklearn.svm import LinearSVC

from sklearn.ensemble import BaggingClassifier

svm = LinearSVC(dual=False, random_state = 42)
clf = BaggingClassifier(base_estimator=svm, n_estimators=31, random_state=314).fit(X_train,y_train)


preds = clf.predict(X_test)
accuracy = metrics.accuracy_score(y_test, preds)
print("bagged svm:", accuracy)



