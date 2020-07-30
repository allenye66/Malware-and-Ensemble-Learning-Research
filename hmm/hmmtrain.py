# -*- coding: utf-8 -*-
"""
Created on Sat Jul 11 11:32:33 2020

@author: Gavin
"""
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
from hmmlearn import hmm
import tensorflow as tf
#import matplotlib.pyplot as plt
import pickle

##Set-up and initializiatio
files = dictionary = {'ADLOAD' : 162, 'AGENT' : 184, 'ALLAPLE_A' : 986, 'BHO' : 332, 'BIFROSE' : 156, 'CEEINJECT' : 873, 'CYCBOT_G' : 597, 'FAKEREAN' : 553,
                  'HOTBAR' : 129, 'INJECTOR' : 158, 'ONLINEGAMES' : 210, 'RENOS' : 532, 'RIMECUD_A' : 153, 'SMALL' : 180, 
                  'TOGA_RFN' : 406, 'VB' : 346, 'VBINJECT' : 937, 'VOBFUS' : 929 , 'VUNDO' : 762, 'WINWEBSEC' : 837, 'ZBOT' : 303}

##ADLOAD - 80%, AGENT - 80%, AllAPLE_A : 70%, BHO: 75%, BIFROSE: 80%, CEEINJECT: 70%, CYCBOT_G: 75%, FAKEREAN: 75%, HOTBAR: 80%, INJECTOR: 80%, ONLINEGAMES: 80%, RENOS: 75%, RIMCUD_A: 80%
##SMALL: 80%, TOGA_RFN: 75%, VB: 75%, VBINJECT: 70%, VOBFUS: 70%, VUNDO: 70%, WINWEBSEC: 70%, ZBOT: 75%
dataSelect = {'ADLOAD' : (0,130), 'AGENT' : (161, 310), 'ALLAPLE_A' : (345, 1036), 'BHO' : (1331, 1582), 'BIFROSE' : (1663, 1790), 'CEEINJECT' : (1819, 2430), 'CYCBOT_G' : (2692, 3140), 'FAKEREAN' : (3290, 3705),
              'HOTBAR' : (3842, 3945), 'INJECTOR' : (3972, 4098), 'ONLINEGAMES' : (4129, 4297), 'RENOS' : (4339, 4738), 'RIMECUD_A' : (4871, 4993), 'SMALL' : (5024, 5168), 
              'TOGA_RFN' : (5204, 5508), 'VB' : (5610, 5869), 'VBINJECT' : (5956, 6612), 'VOBFUS' : (6893, 7543) , 'VUNDO' : (7822, 8355), 'WINWEBSEC' : (8584, 9170), 'ZBOT' : (9421, 9648)}

dataSelect2 = {'ADLOAD' : (0,161), 'AGENT' : (161, 345), 'ALLAPLE_A' : (345, 1331), 'BHO' : (1331, 1663), 'BIFROSE' : (1663, 1819), 'CEEINJECT' : (1819, 2692), 'CYCBOT_G' : (2692, 3290), 'FAKEREAN' : (3290, 3842),
              'HOTBAR' : (3842, 3972), 'INJECTOR' : (3972, 4129), 'ONLINEGAMES' : (4129, 4339), 'RENOS' : (4339, 4871), 'RIMECUD_A' : (4871, 5024), 'SMALL' : (5024, 5204), 
              'TOGA_RFN' : (5204, 5610), 'VB' : (5610, 5956), 'VBINJECT' : (5956, 6893), 'VOBFUS' : (6893, 7822) , 'VUNDO' : (7822, 8584), 'WINWEBSEC' : (8584, 9421), 'ZBOT' : (9421, 9724)}

families = [ 'ADLOAD', 'AGENT' , 'ALLAPLE_A', 'BHO', 'BIFROSE', 'CEEINJECT', 'CYCBOT_G','FAKEREAN', 'HOTBAR', 'INJECTOR',
            'ONLINEGAMES', 'RENOS', 'RIMECUD_A', 'SMALL', 'TOGA_RFN', 'VB', 'VBINJECT',
            'VOBFUS', 'VUNDO', 'WINWEBSEC', 'ZBOT']

dataset = pd.read_csv('all_data2_new.csv')

##Training model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

testX = np.empty((0,0), dtype = np.int8)
testY = np.empty(0)
trainX = np.empty((0,0), dtype = np.int8)
trainY = np.empty(0)
count = 0
all_models = []
for i in families:
    X = dataset.iloc[dataSelect2[i][0]:dataSelect2[i][1], 34:]
    Y = dataset.iloc[dataSelect2[i][0]:dataSelect2[i][1], 1]
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.25, random_state = 23)
    count += X_test.shape[0]
    
    testX = np.append(testX, X_test).reshape(count, 1000)
    testY = np.append(testY, Y_test)

    trainX = np.append(trainX, X_train).reshape(-1, 1000)
    trainY = np.append(trainY, Y_train)

    
    
    model = hmm.MultinomialHMM(n_components=10, n_iter=200, tol=0.5)
    model.fit(X_train)
    all_models.append(model)
    print("done")
    
le = LabelEncoder()
testY = le.fit_transform(testY)

filename = 'finalized_model.sav'
pickle.dump(all_models, open(filename, 'wb'))

filename = 'X_test.sav'
pickle.dump(testX, open(filename, 'wb'))

filename = 'Y_test.sav'
pickle.dump(testY, open(filename, 'wb'))

filename = 'X_train.sav'
pickle.dump(X_train, open(filename, 'wb'))

filename = 'Y_train.sav'
pickle.dump(Y_train, open(filename, 'wb'))

