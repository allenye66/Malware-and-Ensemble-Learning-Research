# %% [code]
import numpy as np 
import pandas as pd 
from sklearn import model_selection
from sklearn.metrics import accuracy_score
import keras
from keras.models import Sequential 
from keras.layers import Activation, MaxPooling1D, Dropout, Flatten, Reshape, Dense, Conv1D, LSTM,SpatialDropout1D
from sklearn.metrics import precision_score, recall_score, accuracy_score, balanced_accuracy_score, f1_score

import numpy as np
import pandas as pd
import seaborn as sns
import os
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split


# %% [code]
#reading in the dataset as a Pandas dataframe variable
df = pd.read_csv('/Users/allen/Desktop/Malware-Research/csv/all_data.csv')

#this dataset has a lot of extra columns we do not need(File Name, Total Opcodes, mov, push, call...)

# %% [code]
#the families we are classifying
print(df.Family.unique())

# %% [code]
#here we are deleting the extra unecessary columns
df = df.loc[:, df.columns != 'Total Opcodes']
df = df.loc[:, df.columns != 'File Name']

#this is our labels for training
labels = np.asarray(df[['Family']].copy())

#encoding the labels to numbers
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
labels = le.fit_transform(labels)

#removing more unecessary columns
for i in range(31):
    df = df.drop(df.columns[1], axis=1)
    
    
#this is our training data
opcode_sequence = (df.drop(df.columns[0], axis=1))



# %% [code]
#the shape of the data right now
print(opcode_sequence.shape)


# %% [code]
#the 1D CNN CONV1D input layer needs to take in a shape of a 3 by 1 array so we reshape it
X_train, X_test, y_train, y_test = train_test_split(opcode_sequence, labels, random_state = 0, test_size = 0.2, stratify = labels)
print(X_train.shape)
print(X_test.shape)
#opcode_sequence = tf.reshape(opcode_sequence, (9725, 1000, 1))
X_train = tf.reshape(X_train, (7780, 1000, 1))
X_test = tf.reshape(X_test, (1945, 1000, 1))


# %% [code]
#plot the accuracy and the validation accuracy
def plot_acc(h):
    plt.plot(h.history['accuracy'])
    plt.plot(h.history['val_accuracy'])

    plt.title('model accuracy')
    plt.ylabel('accuracy and loss')
    plt.xlabel('epoch')

    plt.legend(['acc', 'val acc' ], loc='upper left')
    plt.show()

# %% [code]
#plot the loss and validation loss
def plot_loss(h):
    plt.plot(h.history['loss'])
    plt.plot(h.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('accuracy and loss')
    plt.xlabel('epoch')

    plt.legend(['loss', 'val loss' ], loc='upper left')
    plt.show()

# We are now done preprocessing our dataset and ready to start training the models
model = Sequential()
#the shape of the input is (9725, 1000, 1), where there are 9725 training samples and each training sample has 1k features
model.add(Conv1D(filters= 64, kernel_size=3, activation ='relu',strides = 2, padding = 'valid', input_shape= (1000, 1))) #not sure about a good filter/kernel size
model.add(MaxPooling1D(pool_size=2))
model.add(Conv1D(filters= 128, kernel_size=3, activation ='relu',strides = 2, padding = 'valid'))
model.add(MaxPooling1D(pool_size=2))
model.add(Dropout(0.9))
model.add(MaxPooling1D(pool_size=2))

model.add(Flatten())
model.add(Dense(21)) 
model.add(Activation('softmax'))
opt = keras.optimizers.Adam(learning_rate=0.001)
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


preds = model.predict_classes(X_test)
print("Accuracy = {}".format(accuracy_score(y_test, preds)))
print("Balanced Accuracy = {}".format(balanced_accuracy_score(y_test, preds)))
print("Precision = {}".format(precision_score(y_test, preds, average='macro')))
print("Recall = {}".format(recall_score(y_test, preds, average='macro')))
print("F1 = {}".format(f1_score(y_test, preds, average='weighted')))
#plot the acc and loss graphs
plot_acc(history)
plot_loss(history)
