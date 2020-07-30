import os
from hmmlearn import hmm
import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt
import re

'''
This function takes the observation sequence as input and trains the HMM model with 2 stats and random restarts.
What it returns?: The best model in all restarts.
You can change the number of states by changing value for n_components
'obs_seq'---> This is observation sequence consisting of digits in the range 0 to M-1 where M is no of distinct observation symbol.
This is bit tricky if you plan to train on opcode sequence.
You will have to convert observation seq into sequence of numbers. 
For example "mov, add, sub, mov, mov" should be converted into 0,1,2,0,0. 
Here mapping is mov-->0,add-->1 and so on
Aftr training HMM, 0th column in B matrix represents mov. same for A matrix.
One thing to note here that there should not be any number missing in the observation sequence. i.e. 0,1,3,0,0 is invalid sequence as 2 is not present anywhere.
Be careful when doing mapping.

Also, I have already added code for random restarts here. It is not available out of box in hmmlearn. I will recommend debugging once to understand working.

'''

dataset = pd.read_csv('all_data2_new.csv')
X = dataset.iloc[:, 34:]
Y = dataset.iloc[:, 1]

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
Y = le.fit_transform(Y)

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.25, random_state = 42)



def train_hmm_random_restarts(obs_seq):
    len_o = len(obs_seq[0])
    if len_o>30000:
        random_restarts = 10
    elif len_o>10000:
        random_restarts = 30
    elif len_o>=5000:
        random_restarts = 50
    else:
        random_restarts = 100
    model= hmm.MultinomialHMM(n_components=2, n_iter=500, tol=0.5)
       # model.verbose=True
    model.fit(X=obs_seq)
    prev_model = model
    prev_log_prob = model.monitor_.history.pop()
    #random_restarts = 0
    while(random_restarts!=0):
        model= hmm.MultinomialHMM(n_components=2, n_iter=500, tol=0.5)
        #    model.verbose=True
            model.fit(X=obs_seq)
        log_prob = model.monitor_.history.pop()
        if (prev_log_prob < log_prob):
            prev_model = model
            prev_log_prob = log_prob
        random_restarts -= 1
    return prev_model

mymodel = train_hmm_random_restarts(np.array([[0,1,3,0,0,0,1,2,2,2,2], [1,2,1,1,2,3,1,2]]))
print(mymodel.emissionprob_)
print(mymodel.transmat_)
