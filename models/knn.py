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
from sklearn.svm import LinearSVC
import time

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





from sklearn.neighbors import KNeighborsClassifier


acc_scores = []

for i in range(70, 90):
	if i % 2 == 1:
		knn = KNeighborsClassifier(n_neighbors=i)
		knn.fit(X_train,y_train)
		best_preds = knn.predict(X_test)
		acc = accuracy_score(y_test, best_preds)
		acc_scores.append(acc)
		print(i, acc)
print(acc_scores)


'''
knn = KNeighborsClassifier(n_neighbors=93)
knn.fit(X_train,y_train)
best_preds = knn.predict(X_test)
print("Accuracy = {}".format(accuracy_score(y_test, best_preds)))
print("Balanced Accuracy = {}".format(balanced_accuracy_score(y_test, best_preds)))
print("Precision = {}".format(precision_score(y_test, best_preds, average='macro')))
print("Recall = {}".format(recall_score(y_test, best_preds, average='macro')))
print("F1 = {}".format(f1_score(y_test, best_preds, average='weighted')))

print("--- %s seconds ---" % (time.time() - start_time))
'''