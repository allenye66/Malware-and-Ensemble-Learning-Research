from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from itertools import product
from sklearn.ensemble import VotingClassifier
import pandas as pd
import numpy as np
import sklearn.metrics as metrics
from sklearn.metrics import precision_score, recall_score, accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels
from sklearn.neural_network import MLPClassifier
import xgboost as xgb
import time
from xgboost import XGBClassifier

start_time = time.time()

df = pd.read_csv('/Users/allen/Desktop/Malware-Research/csv/all_data.csv')


df = df.loc[:, df.columns != 'Total Opcodes']
df = df.loc[:, df.columns != 'File Name']

labels = np.asarray(df[['Family']].copy())

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
labels = le.fit_transform(labels)

for i in range(31):
    df = df.drop(df.columns[1], axis=1)



opcode_sequence = (df.drop(df.columns[0], axis=1))
X_train, X_test, y_train, y_test = train_test_split(opcode_sequence, labels, test_size=0.1, random_state=42)




D_train = xgb.DMatrix(X_train, label= y_train)
D_test = xgb.DMatrix(X_test, label=y_test)
param = { 'eta': 0.3, 'max_depth': 3,  'objective': 'multi:softprob',  'num_class': 21} 
steps = 20 

clf1 = XGBClassifier()
clf2 = DecisionTreeClassifier(max_depth=60)
clf3 = RandomForestClassifier(n_estimators = 200, min_samples_split = 2, min_samples_leaf = 1, max_features = 'auto', max_depth= 40)

#clf3 = MLPClassifier(hidden_layer_sizes=(80, 80), max_iter= 100000) 

eclf = VotingClassifier(estimators=[('xgboost', clf1), ('dt', clf2), ('rf', clf3)], voting='soft', weights=[3, 2, 3])

clf1 = xgb.train(param, D_train, steps)
clf2 = clf2.fit(X_train, y_train)
clf3 = clf3.fit(X_train, y_train)
eclf = eclf.fit(X_train, y_train)

preds1 = clf1.predict(D_test)
preds2 = clf2.predict(X_test)
preds3 = clf3.predict(X_test)
preds_eclf = eclf.predict(X_test)

print("--- %s seconds ---" % (time.time() - start_time))


print("first classifier:", metrics.accuracy_score(y_test, preds1))
print("second classifier:", metrics.accuracy_score(y_test, preds2))
print("third classifier:", metrics.accuracy_score(y_test, preds3))

print("Precision = {}".format(precision_score(y_test, preds_eclf, average='macro')))
print("Recall = {}".format(recall_score(y_test, preds_eclf, average='macro')))
print("Accuracy = {}".format(accuracy_score(y_test, preds_eclf)))
print("Balanced Accuracy = {}".format(balanced_accuracy_score(y_test, preds_eclf)))
print("F1 = {}".format(f1_score(y_test, preds_eclf, average='weighted')))

