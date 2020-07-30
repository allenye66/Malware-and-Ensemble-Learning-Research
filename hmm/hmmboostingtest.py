import pickle
import numpy as np

families = [ 'ADLOAD', 'AGENT' , 'ALLAPLE_A', 'BHO', 'BIFROSE', 'CEEINJECT', 'CYCBOT_G','FAKEREAN', 'HOTBAR', 'INJECTOR',
            'ONLINEGAMES', 'RENOS', 'RIMECUD_A', 'SMALL', 'TOGA_RFN', 'VB', 'VBINJECT',
            'VOBFUS', 'VUNDO', 'WINWEBSEC', 'ZBOT']

file = '/Users/gavinwong/Desktop/Repos/SJUMalwareEnsembleResearch/Models/hmm/Boosting/finalized_model'
filename2 = '/Users/gavinwong/Desktop/Repos/SJUMalwareEnsembleResearch/Models/hmm/X_train.sav'
filename3 = '/Users/gavinwong/Desktop/Repos/SJUMalwareEnsembleResearch/Models/hmm/Y_train.sav'
filename4 = '/Users/gavinwong/Desktop/Repos/SJUMalwareEnsembleResearch/Models/hmm/Boosting/errors5.sav'
filename5 = '/Users/gavinwong/Desktop/Repos/SJUMalwareEnsembleResearch/Models/hmm/Boosting/Y_pred5.sav'
X = pickle.load(open(filename2, 'rb'))
Y = pickle.load(open(filename3, 'rb'))
print(X.shape)
asd
error = [np.empty((0,0), dtype = np.int8)] * 21
scores = [0] * 21
array = [0] * 21


def predict(row, modelFile): #data is 2d np array
    all_models = pickle.load(open(modelFile, 'rb'))
    bestScore = -9999999999
    best_model = 0
    count = -1
    for model in all_models:
        count += 1
        try:
            score = model.score(np.reshape(row, (-1, 1)))
            if score > bestScore:
                bestScore = score
                best_model = count
        except:
            continue
    global scores
    scores[best_model] += bestScore
    return best_model

def checkPred(array):
    bestScore = -1
    count = -1
    best_model = -1
    for i in range(21):
        if array[i] > count:
            count = array[i]
            best_model = i
            bestScore = scores[i]
        elif array[i] == count and scores[i] > bestScore:
            best_model = i
            bestScore = scores[i]
    return best_model

#Displaying results
# print(Y_pred)

for row in X: #change later
    for i in range(1,6):
        modelFile = file + str(i) + '.sav'
        pred = predict(row, modelFile)
        array[pred] += 1
    final_Pred = checkPred(array)
    Y_pred = np.append(Y_pred, final_Pred)
    print(families[final_Pred])
    array = [0] * 21


from sklearn.metrics import accuracy_score
print("-------------------------")
print(accuracy_score(Y, Y_pred))
pickle.dump(Y_pred, open(filename5, 'wb'))

for i in range(7284):
    if(Y[i] != Y_pred[i]):
        error[Y[i]] = np.append(error[Y[i]], X[i])

pickle.dump(error, open(filename4, 'wb'))





