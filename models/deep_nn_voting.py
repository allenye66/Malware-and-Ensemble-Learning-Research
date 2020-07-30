# %% [code]
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

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
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        
        print(os.path.join(dirname, filename))

# %% [code]
'''
models = []
for i in range(10):
    json_filename = '/kaggle/input/bagged-malware-cnn/bagged_cnn_' + str(i) + ".json"
    model_name = "bagged_cnn" + str(i)
    h5_filename = "/kaggle/input/bagged-malware-cnn/bagged_cnn_" + str(i) + ".h5"
    
    json_file = open(json_filename)
    loaded_json = json_file.read()
    json_file.close()
    model_name = tf.keras.models.model_from_json(loaded_json)
    model_name.load_weights(h5_filename)
    print(model_name.predict(X_test))
    #models.append(model)

print(models)

'''

# %% [code]
#baggged cnn
json_file = open('/kaggle/input/bagged-malware-cnn/bagged_cnn_0.json', 'r')
mj0 = json_file.read()
json_file.close()

bagged_cnn_0 = tf.keras.models.model_from_json(mj0)
bagged_cnn_0.load_weights("/kaggle/input/bagged-malware-cnn/bagged_cnn_0.h5")
json_file = open('/kaggle/input/bagged-malware-cnn/bagged_cnn_1.json', 'r')
mj1 = json_file.read()
json_file.close()

bagged_cnn_1 = tf.keras.models.model_from_json(mj1)
bagged_cnn_1.load_weights("/kaggle/input/bagged-malware-cnn/bagged_cnn_1.h5")
json_file = open('/kaggle/input/bagged-malware-cnn/bagged_cnn_2.json', 'r')
mj2 = json_file.read()
json_file.close()

bagged_cnn_2 = tf.keras.models.model_from_json(mj2)
bagged_cnn_2.load_weights("/kaggle/input/bagged-malware-cnn/bagged_cnn_2.h5")
json_file = open('/kaggle/input/bagged-malware-cnn/bagged_cnn_3.json', 'r')
mj3 = json_file.read()
json_file.close()

bagged_cnn_3 = tf.keras.models.model_from_json(mj3)
bagged_cnn_3.load_weights("/kaggle/input/bagged-malware-cnn/bagged_cnn_3.h5")
json_file = open('/kaggle/input/bagged-malware-cnn/bagged_cnn_4.json', 'r')
mj4 = json_file.read()
json_file.close()

bagged_cnn_4 = tf.keras.models.model_from_json(mj4)
bagged_cnn_4.load_weights("/kaggle/input/bagged-malware-cnn/bagged_cnn_4.h5")
json_file = open('/kaggle/input/bagged-malware-cnn/bagged_cnn_5.json', 'r')
mj5 = json_file.read()
json_file.close()

bagged_cnn_5 = tf.keras.models.model_from_json(mj5)
bagged_cnn_5.load_weights("/kaggle/input/bagged-malware-cnn/bagged_cnn_5.h5")
json_file = open('/kaggle/input/bagged-malware-cnn/bagged_cnn_6.json', 'r')
mj6 = json_file.read()
json_file.close()

bagged_cnn_6 = tf.keras.models.model_from_json(mj6)
bagged_cnn_6.load_weights("/kaggle/input/bagged-malware-cnn/bagged_cnn_6.h5")
json_file = open('/kaggle/input/bagged-malware-cnn/bagged_cnn_7.json', 'r')
mj7 = json_file.read()
json_file.close()

bagged_cnn_7 = tf.keras.models.model_from_json(mj7)
bagged_cnn_7.load_weights("/kaggle/input/bagged-malware-cnn/bagged_cnn_7.h5")
json_file = open('/kaggle/input/bagged-malware-cnn/bagged_cnn_8.json', 'r')
mj8 = json_file.read()
json_file.close()

bagged_cnn_8 = tf.keras.models.model_from_json(mj8)
bagged_cnn_8.load_weights("/kaggle/input/bagged-malware-cnn/bagged_cnn_8.h5")
json_file = open('/kaggle/input/bagged-malware-cnn/bagged_cnn_9.json', 'r')
mj9 = json_file.read()
json_file.close()

bagged_cnn_9 = tf.keras.models.model_from_json(mj9)
bagged_cnn_9.load_weights("/kaggle/input/bagged-malware-cnn/bagged_cnn_9.h5")

# %% [code]
#bagged lstm
json_file = open('/kaggle/input/bagged-lstm/bagged_lstm_5.json', 'r')
loaded_model_json = json_file.read()
json_file.close()

bagged_lstm_4 = tf.keras.models.model_from_json(loaded_model_json)
bagged_lstm_4.load_weights("/kaggle/input/bagged-lstm/bagged_lstm_5.h5")


json_file = open('/kaggle/input/bagged-lstm/bagged_lstm_4.json', 'r')
loaded_model_json = json_file.read()
json_file.close()

bagged_lstm_3 = tf.keras.models.model_from_json(loaded_model_json)
bagged_lstm_3.load_weights("/kaggle/input/bagged-lstm/bagged_lstm_4.h5")


json_file = open('/kaggle/input/bagged-lstm/bagged_lstm_3.json', 'r')
loaded_model_json = json_file.read()
json_file.close()

bagged_lstm_2 = tf.keras.models.model_from_json(loaded_model_json)
bagged_lstm_2.load_weights("/kaggle/input/bagged-lstm/bagged_lstm_3.h5")


json_file = open('/kaggle/input/bagged-lstm/bagged_lstm_2.json', 'r')
loaded_model_json = json_file.read()
json_file.close()

bagged_lstm_1 = tf.keras.models.model_from_json(loaded_model_json)
bagged_lstm_1.load_weights("/kaggle/input/bagged-lstm/bagged_lstm_2.h5")

json_file = open('/kaggle/input/bagged-lstm/bagged_lstm_1.json', 'r')
loaded_model_json = json_file.read()
json_file.close()

bagged_lstm_0 = tf.keras.models.model_from_json(loaded_model_json)
bagged_lstm_0.load_weights("/kaggle/input/bagged-lstm/bagged_lstm_1.h5")

# %% [code]
#boosted lstm


json_file = open('/kaggle/input/boosted-all/boosted_lstm_4.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
boosted_lstm_4 = tf.keras.models.model_from_json(loaded_model_json)
boosted_lstm_4.load_weights("/kaggle/input/boosted-all/boosted_lstm_4.h5")



json_file = open('/kaggle/input/boosted-all/boosted_lstm_3.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
boosted_lstm_3 = tf.keras.models.model_from_json(loaded_model_json)
boosted_lstm_3.load_weights("/kaggle/input/boosted-all/boosted_lstm_3.h5")




json_file = open('/kaggle/input/boosted-all/boosted_lstm_2.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
boosted_lstm_2 = tf.keras.models.model_from_json(loaded_model_json)
boosted_lstm_2.load_weights("/kaggle/input/boosted-all/boosted_lstm_2.h5")



json_file = open('/kaggle/input/boosted-all/boosted_lstm_1.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
boosted_lstm_1 = tf.keras.models.model_from_json(loaded_model_json)
boosted_lstm_1.load_weights("/kaggle/input/boosted-all/boosted_lstm_1.h5")



json_file = open('/kaggle/input/boosted-all/boosted_lstm_0.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
boosted_lstm_0 = tf.keras.models.model_from_json(loaded_model_json)
boosted_lstm_0.load_weights("/kaggle/input/boosted-all/boosted_lstm_0.h5")


# %% [code]
#boosted cnn
json_file = open('/kaggle/input/boostcnn/boosted_cnn_0.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
boosted_cnn_0 = tf.keras.models.model_from_json(loaded_model_json)
boosted_cnn_0.load_weights("/kaggle/input/boostcnn/boosted_cnn_0.h5")


json_file = open('/kaggle/input/boostcnn/boosted_cnn_1.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
boosted_cnn_1 = tf.keras.models.model_from_json(loaded_model_json)
boosted_cnn_1.load_weights("/kaggle/input/boostcnn/boosted_cnn_1.h5")


json_file = open('/kaggle/input/boostcnn/boosted_cnn_2.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
boosted_cnn_2 = tf.keras.models.model_from_json(loaded_model_json)
boosted_cnn_2.load_weights("/kaggle/input/boostcnn/boosted_cnn_2.h5")

json_file = open('/kaggle/input/boostcnn/boosted_cnn_3.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
boosted_cnn_3 = tf.keras.models.model_from_json(loaded_model_json)
boosted_cnn_3.load_weights("/kaggle/input/boostcnn/boosted_cnn_3.h5")



json_file = open('/kaggle/input/boostcnn/boosted_cnn_4.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
boosted_cnn_4 = tf.keras.models.model_from_json(loaded_model_json)
boosted_cnn_4.load_weights("/kaggle/input/boostcnn/boosted_cnn_4.h5")

json_file = open('/kaggle/input/boostcnn/boosted_cnn_5.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
boosted_cnn_5 = tf.keras.models.model_from_json(loaded_model_json)
boosted_cnn_5.load_weights("/kaggle/input/boostcnn/boosted_cnn_5.h5")



json_file = open('/kaggle/input/boostcnn/boosted_cnn_6.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
boosted_cnn_6 = tf.keras.models.model_from_json(loaded_model_json)
boosted_cnn_6.load_weights("/kaggle/input/boostcnn/boosted_cnn_6.h5")



json_file = open('/kaggle/input/boostcnn/boosted_cnn_7.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
boosted_cnn_7= tf.keras.models.model_from_json(loaded_model_json)
boosted_cnn_7.load_weights("/kaggle/input/boostcnn/boosted_cnn_7.h5")


json_file = open('/kaggle/input/boostcnn/boosted_cnn_8.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
boosted_cnn_8 = tf.keras.models.model_from_json(loaded_model_json)
boosted_cnn_8.load_weights("/kaggle/input/boostcnn/boosted_cnn_8.h5")

json_file = open('/kaggle/input/boostcnn/boosted_cnn_9.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
boosted_cnn_9 = tf.keras.models.model_from_json(loaded_model_json)
boosted_cnn_9.load_weights("/kaggle/input/boostcnn/boosted_cnn_9.h5")

# %% [code]
df = pd.read_csv('/kaggle/input/final-opcodes/all_data.csv')
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
X_train, X_test, y_train, y_test = train_test_split(opcode_sequence, labels, test_size=test_size, random_state = 42)


X_test = tf.reshape(X_test, (1945, 1000, 1))

# %% [code]
predictions = []
#all bagged cnn
#models = [bagged_cnn_0, bagged_cnn_1,bagged_cnn_2, bagged_cnn_3, bagged_cnn_4, bagged_cnn_5, bagged_cnn_6, bagged_cnn_7, bagged_cnn_8, bagged_cnn_9]

#all bagged lstm
#models = [bagged_lstm_0, bagged_lstm_1, bagged_lstm_2, bagged_lstm_3, bagged_lstm_4]

#bagged cnn and lstm
#models = [bagged_lstm_0, bagged_lstm_1, bagged_lstm_2, bagged_lstm_3, bagged_lstm_4,bagged_cnn_0, bagged_cnn_1,bagged_cnn_2, bagged_cnn_3, bagged_cnn_4, bagged_cnn_5, bagged_cnn_6, bagged_cnn_7, bagged_cnn_8, bagged_cnn_9 ]

#all boosted lstm
#models = [boosted_lstm_0, boosted_lstm_1, boosted_lstm_2, boosted_lstm_3, boosted_lstm_4]

#all boosted cnn
#models = [boosted_cnn_0, boosted_cnn_1, boosted_cnn_2, boosted_cnn_3, boosted_cnn_4, boosted_cnn_5, boosted_cnn_6, boosted_cnn_7, boosted_cnn_8, boosted_cnn_9]

#boosted cnn and lstm
#models = [boosted_cnn_0, boosted_cnn_1, boosted_cnn_2, boosted_cnn_3, boosted_cnn_4, boosted_cnn_5, boosted_cnn_6, boosted_cnn_7, boosted_cnn_8, boosted_cnn_9, boosted_lstm_0, boosted_lstm_1, boosted_lstm_2, boosted_lstm_3, boosted_lstm_4]

#only cnn
#models = [boosted_cnn_0, boosted_cnn_1, boosted_cnn_2, boosted_cnn_3, boosted_cnn_4, boosted_cnn_5, boosted_cnn_6, boosted_cnn_7, boosted_cnn_8, boosted_cnn_9, bagged_cnn_0, bagged_cnn_1,bagged_cnn_2, bagged_cnn_3, bagged_cnn_4, bagged_cnn_5, bagged_cnn_6, bagged_cnn_7, bagged_cnn_8, bagged_cnn_9]

#only lstm
#models = [bagged_lstm_0, bagged_lstm_1, bagged_lstm_2, bagged_lstm_3, bagged_lstm_4, boosted_lstm_0, boosted_lstm_1, boosted_lstm_2, boosted_lstm_3, boosted_lstm_4]


#all boosted and bagged
models = [bagged_lstm_0, bagged_lstm_1, bagged_lstm_2, bagged_lstm_3, bagged_lstm_4,bagged_cnn_0, bagged_cnn_1,bagged_cnn_2, bagged_cnn_3, bagged_cnn_4, bagged_cnn_5, bagged_cnn_6, bagged_cnn_7, bagged_cnn_8, bagged_cnn_9, boosted_cnn_0, boosted_cnn_1, boosted_cnn_2, boosted_cnn_3, boosted_cnn_4, boosted_cnn_5, boosted_cnn_6, boosted_cnn_7, boosted_cnn_8, boosted_cnn_9, boosted_lstm_0, boosted_lstm_1, boosted_lstm_2, boosted_lstm_3, boosted_lstm_4 ]


for model in models:
    
    preds = model.predict(X_test)
    preds = list(preds)
    
    one_model = []
    for i in preds:
        i = list(i)
        p = (i.index(max(i)))
        one_model.append(p)
    predictions.append(one_model)
    #print(one_model)
    #print("______________________________________next model__________")
            
print(predictions)
        
    

# %% [code]
  
def most_frequent(List): 
    counter = 0
    num = List[0]
    if len(List) == 1:
        return num
    for i in List: 
        curr_frequency = List.count(i) 
        if(curr_frequency> counter): 
            counter = curr_frequency 
            num = i 
  
    return num 

# %% [code]
voting = []
for i in range(1945):
    arr = []
    for j in range(len(models)):
        arr.append(predictions[j][i])
        if j == len(models)-1:
            print(arr)
            print(most_frequent(arr))
            voting.append(most_frequent(arr))
print((voting))

# %% [code]
print("Accuracy = {}".format(accuracy_score(y_test, voting)))
print("Balanced Accuracy = {}".format(balanced_accuracy_score(y_test, voting)))
print("Precision = {}".format(precision_score(y_test, voting, average='weighted')))
print("Recall = {}".format(recall_score(y_test, voting, average='weighted')))
print("F1 = {}".format(f1_score(y_test, voting, average='weighted')))

# %% [code]


# %% [code]
