
import pandas as pd
import numpy as np
import keras
import tensorflow as tf
import sklearn.metrics as metrics
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels
import seaborn as sns
import pickle
from sklearn.metrics import precision_score, recall_score, accuracy_score, balanced_accuracy_score, f1_score
import os
from collections import Counter
from keras.models import model_from_json



json_file = open('/Users/allen/Desktop/Malware-Research/bagged_cnn/bagged_cnn_9.json', 'r')
loaded_model_json = json_file.read()
json_file.close()

loaded_model = tf.keras.models.model_from_json(loaded_model_json)
loaded_model.load_weights("/Users/allen/Desktop/Malware-Research/bagged_cnn/bagged_cnn_9.h5")






df = pd.read_csv('/Users/allen/Desktop/Malware-Research/csv/all_data.csv')
family_count = [162, 184, 986, 332, 156, 873, 597, 553, 129, 158, 210, 532, 153, 180, 406, 346, 937, 929, 762, 837, 303]


df = df.loc[:, df.columns != 'Total Opcodes']
df = df.loc[:, df.columns != 'File Name']

labels = np.asarray(df[['Family']].copy())

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
labels = le.fit_transform(df['Family'])

for i in range(31):
    df = df.drop(df.columns[1], axis=1)

test_size = 0.2


opcode_sequence = (df.drop(df.columns[0], axis=1))
X_train, X_test, y_train, y_test = train_test_split(opcode_sequence, labels, test_size=test_size, random_state=42)


X_test = tf.reshape(X_test, (1945, 1000, 1))

p = loaded_model.predict_classes(X_test)
print("Accuracy = {}".format(accuracy_score(y_test, p)))


