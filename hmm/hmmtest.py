import pickle
import numpy as np




families = [ 'ADLOAD', 'AGENT' , 'ALLAPLE_A', 'BHO', 'BIFROSE', 'CEEINJECT', 'CYCBOT_G','FAKEREAN', 'HOTBAR', 'INJECTOR',
        'ONLINEGAMES', 'RENOS', 'RIMECUD_A', 'SMALL', 'TOGA_RFN', 'VB', 'VBINJECT',
        'VOBFUS', 'VUNDO', 'WINWEBSEC', 'ZBOT']

file = '/Users/gavinwong/Desktop/Repos/SJUMalwareEnsembleResearch/Models/hmm/Bagging/finalized_model'
fileX = '/Users/gavinwong/Desktop/Repos/SJUMalwareEnsembleResearch/Models/hmm/Bagging/X_test.sav'
fileY = '/Users/gavinwong/Desktop/Repos/SJUMalwareEnsembleResearch/Models/hmm/Bagging/Y_test.sav'

X_test = pickle.load(open(fileX, 'rb'))
Y_test = pickle.load(open(fileY, 'rb'))
Y_pred = np.empty(0, dtype=np.int8)
scores = [0] * 21

def predict(row, modelFile): #data is 2d np array
    all_models = pickle.load(open(modelFile, 'rb'))
    bestScore = -9999999999
    count = -1
    for model in all_models:
        count += 1
        try:
            score = model.score(np.reshape(row, (-1, 1)))
            global scores
            scores[count] += score
        except:
            continue

def checkPred(array):
    bestScore = -9999999999
    best_model = -1
    for i in range(21):
        if array[i] > bestScore:
            bestScore = array[i]
            best_model = i
    return best_model



for row in X_test: #change later
    for i in range(1,3):
        modelFile = file + str(i) + '.sav'
        predict(row, modelFile)
    final_Pred = checkPred(scores)
    Y_pred = np.append(Y_pred, final_Pred)
    print(families[final_Pred])
    print(scores)
    scores = [0] * 21

        



        
    
    
filename4 = '/Users/gavinwong/Desktop/Repos/SJUMalwareEnsembleResearch/Models/hmm/Bagging/Y_pred.sav'
# print(Y_pred)
pickle.dump(Y_pred, open(filename4, 'wb'))

from sklearn.metrics import accuracy_score
print("-------------------------")
print(accuracy_score(Y_test, Y_pred))