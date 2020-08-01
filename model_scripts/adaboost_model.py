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
from sklearn.metrics import balanced_accuracy_score
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
import seaborn as sns
import pickle
from sklearn.metrics import balanced_accuracy_score
import os
import time
from collections import Counter
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

#print(mapping)

opcode_sequence = (df.drop(df.columns[0], axis=1))
X_train, X_test, y_train, y_test = train_test_split(opcode_sequence, labels, test_size=0.1, random_state=42)

mapping = Counter(y_test)
#print(Counter(y_test))
mapping = dict(sorted(mapping.items()))
'''
from sklearn.ensemble import AdaBoostClassifier
from sklearn.datasets import make_classification

clf = AdaBoostClassifier(n_estimators= 1000, learning_rate= 0.5, algorithm='SAMME')
clf.fit(X_train, y_train)

preds = clf.predict(X_test)
accuracy = metrics.accuracy_score(y_test, preds)
print(accuracy)

print("balanced_accuracy_score:", metrics.balanced_accuracy_score(y_test, preds))
print("--- %s seconds ---" % (time.time() - start_time))

pickle.dump(clf, open('adaboost_model.sav', 'wb'))

'''
#--- 259.12324500083923 seconds ---


#0.5308430431802604
#balanced_accuracy_score: 0.40621031502551685




with open('/Users/allen/Desktop/Malware-Research/saved_models/adaboost_model.sav', 'rb') as file:  
   model = pickle.load(file)

best_preds = model.predict(X_test)


label_map = {"0":"ADLOAD","1":"AGENT","2":"ALLAPLE_A","3":"BHO","4":"BIFROSE","5":"CEEINJECT","6":"CYCBOT_G","7":"FAKEREAN","8":"HOTBAR","9":"INJECTOR","10":"ONLINEGAMES","11":"RENOS","12":"RIMECUD_A","13":"SMALL","14":"TOGA_RFN","15":"VB","16":"VBINJECT","17":"VOBFUS", "18":"VUNDO","19":"WINWEBSEC","20":"ZBOT"  }


#print(y_test)


def write_cm(cm):
	file = open("/Users/allen/Desktop/Malware-Research/confusion_matrix_files/cm_ada.txt","w")
	for y in range(0, 21):
		for x in range(0, 21):
			string = (str(x) + " " + str(y) + " "+ str(round(cm[y][x],4)))
			file.write(string + "\n")


	file.close()


def plot_confusion_matrix(y_true,y_predicted):
	cm = metrics.confusion_matrix(y_true, y_predicted)
	l = list(cm)
	#print(l)

	s = 0

	for array in l:
		for value in array:
			s += value

	ooga = []
	counter = 0
	for array in l:
		array = list(array)
		array = [round(x /mapping[counter],3)  for x in array]
		ooga.append(array)
		counter += 1

	print(ooga)


	#cm = list((cm.T / cm.astype(np.float).sum(axis=1)).T)


	write_cm(ooga)
	#print ("Plotting the Confusion Matrix")


	labels = list(label_map.values())


	df_cm = pd.DataFrame(ooga,index = labels,columns = labels)
	fig = plt.figure()
	ax = sns.heatmap(df_cm, annot=True,cmap='Blues', fmt='g')

	plt.yticks([0.5,1.5,2.5,3.5,4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5, 11.5, 12.5, 13.5, 14.5, 15.5, 16.5, 17.5, 18.5, 19.5, 20.5], labels,va='center')

	plt.ylabel('True label')
	plt.xlabel('Predicted label')
 
	plt.show()
	plt.close()

plot_confusion_matrix(y_test, best_preds)

