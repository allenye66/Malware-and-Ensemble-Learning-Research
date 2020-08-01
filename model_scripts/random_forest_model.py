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
from sklearn.metrics import precision_score, recall_score, accuracy_score, balanced_accuracy_score, f1_score
import os
from collections import Counter

df = pd.read_csv('/Users/allen/Desktop/Malware-Research/csv/all_data.csv')
#print(df.shape)
#print(df.groupby('Family').agg(['count', 'sum']))

family_count = [162, 184, 986, 332, 156, 873, 597, 553, 129, 158, 210, 532, 153, 180, 406, 346, 937, 929, 762, 837, 303]
#print(sum(family_count))

test_size = 0.2

test_size_count = [i * test_size for i in family_count]
#print(test_size_count)




df = df.loc[:, df.columns != 'Total Opcodes']
df = df.loc[:, df.columns != 'File Name']

labels = np.asarray(df[['Family']].copy())

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
labels = le.fit_transform(df['Family'])

for i in range(31):
    df = df.drop(df.columns[1], axis=1)



opcode_sequence = (df.drop(df.columns[0], axis=1))
X_train, X_test, y_train, y_test = train_test_split(opcode_sequence, labels, test_size=test_size)



'''

#{'n_estimators': 100, 'min_samples_split': 2, 'min_samples_leaf': 1, 'max_features': 'auto', 'max_depth': 50}
#{'n_estimators': 300, 'min_samples_split': 2, 'min_samples_leaf': 2, 'max_features': 'sqrt', 'max_depth': 30}
#{'n_estimators': 500, 'min_samples_split': 15, 'min_samples_leaf': 1, 'max_features': 'auto', 'max_depth': 30}

#{'n_estimators': 200, 'min_samples_split': 2, 'min_samples_leaf': 1, 'max_features': 'auto', 'max_depth': 40}        
#rf = RandomForestClassifier(n_estimators = 300, min_samples_split = 2, min_samples_leaf =2, max_features = 'sqrt', max_depth= 30)

#rf = RandomForestClassifier(n_estimators = 500, min_samples_split = 15, min_samples_leaf = 1, max_features = 'auto', max_depth= 30)


rf = RandomForestClassifier(n_estimators = 200, min_samples_split = 2, min_samples_leaf = 1, max_features = 'auto', max_depth= 40)

#rf = RandomForestClassifier(max_depth=50, random_state=0)
rf.fit(X_train, y_train)

best_preds = rf.predict(X_test)
print("random_state=1,learning_rate_init = 0.0001, batch_size = 256, hidden_layer_sizes=(500, 500), max_iter= 1000000")
print("Accuracy = {}".format(accuracy_score(y_test, best_preds)))
print("Balanced Accuracy = {}".format(balanced_accuracy_score(y_test, best_preds)))
print("Precision = {}".format(precision_score(y_test, best_preds, average='macro')))
print("Recall = {}".format(recall_score(y_test, best_preds, average='macro')))
print("F1 = {}".format(f1_score(y_test, best_preds, average='weighted')))


pickle.dump(rf, open('random_forest_model.sav', 'wb'))
'''

#print(y_test)
#print(list(y_test))

mapping = Counter(y_test)
#print(Counter(y_test))
mapping = dict(sorted(mapping.items()))
#print(mapping)
#print(mapping)


total_families = []
for key, value in mapping.items():
	total_families.append(value)

print(total_families)


with open('/Users/allen/Desktop/Malware-Research/saved_models/random_forest_model.sav', 'rb') as file:  
   rf_pikl = pickle.load(file)

best_preds = rf_pikl.predict(X_test)
print("random_state=1,learning_rate_init = 0.0001, batch_size = 256, hidden_layer_sizes=(500, 500), max_iter= 1000000")
print("Accuracy = {}".format(accuracy_score(y_test, best_preds)))
print("Balanced Accuracy = {}".format(balanced_accuracy_score(y_test, best_preds)))
print("Precision = {}".format(precision_score(y_test, best_preds, average='weighted')))
print("Recall = {}".format(recall_score(y_test, best_preds, average='weighted')))
print("F1 = {}".format(f1_score(y_test, best_preds, average='weighted')))


'''
#n_estimators = [500, 800, 1500, 2500, 5000]
n_estimators = [100]
min_samples_split = [2, 5, 10, 15, 20]
min_samples_leaf = [1, 2, 5, 10, 15]
max_features = ['auto', 'sqrt','log2']
#max_depth = [10, 20, 30, 40, 50, 60, 70, 80]
max_depth = [50, 60, 70]
max_depth.append(None)





##RESUTS
#n_estimators = 500, min_samples_split = 15, min_samples_leaf = 1, max_features = 'auto', max_depth= 30 
	#70%
#n_estimators = 150, min_samples_split = 2, min_samples_leaf = 1, max_features = 'auto', max_depth= 50
	#~71




grid_param = {'n_estimators': n_estimators, 'max_features': max_features, 'max_depth':max_depth, 'min_samples_split':min_samples_split, 'min_samples_leaf': min_samples_leaf}

from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import RandomizedSearchCV
RFR = RandomForestRegressor(random_state = 1)
RFR_random = RandomizedSearchCV(estimator = RFR, param_distributions = grid_param, n_iter = 500, cv =5, verbose = 2, random_state= 42, n_jobs = -1)
RFR_random.fit(X_train, y_train)
print(RFR_random.best_params_)



'''


label_map = {"0":"ADLOAD","1":"AGENT","2":"ALLAPLE_A","3":"BHO","4":"BIFROSE","5":"CEEINJECT","6":"CYCBOT_G","7":"FAKEREAN","8":"HOTBAR","9":"INJECTOR","10":"ONLINEGAMES","11":"RENOS","12":"RIMECUD_A","13":"SMALL","14":"TOGA_RFN","15":"VB","16":"VBINJECT","17":"VOBFUS", "18":"VUNDO","19":"WINWEBSEC","20":"ZBOT"  }



#print(y_test)


def write_cm(cm):
	file = open("/Users/allen/Desktop/Malware-Research/confusion_matrix_files/cm_rf.txt","w")
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






'''

array = (confusion_matrix(y_test, rf_preds))

df_cm = pd.DataFrame(array, index = [i for i in "ABCDEFGHIJK"], columns = [i for i in "ABCDEFGHIJK"])
plt.figure(figsize = (10,7))
sns.heatmap(df_cm, annot=True)
'''

#plot_confusion_matrix( y_test2, rf_preds, classes=labels, title= 'Confusion matrix, without normalization')

#plot_confusion_matrix(labels_test, rf_preds,classes=class_names, normalize=True, title='Normalized confusion matrix')




#print(y_test2)
#print(rf_preds)
