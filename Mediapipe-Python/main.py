# pip install virtualenv
# virtualenv mypython

# attivare tesivenv
# tesivenv\Scripts\activate

# pip install mediapipe opencv-python


import cv2
import mediapipe as mp
import numpy as np
import math
import os
from scipy.sparse.sputils import matrix

from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import TensorBoard

#nostri moduli
import Tpose_module
import annotateData_module
import train_module
import math_module
import decisionTree_module

#Tpose_module.Tpose()

#crea i video e i file pkl annotati dai file mp4 presenti nelle sottocartelle di alzateLaterali

#annotateData_module.createAnnotation("alzateLaterali")
sequences,labels,actions=annotateData_module.readAnnotation("alzateLaterali")

print(len(sequences))
#print(sequences)
print(np.array(sequences[2]).shape) #(numero di frame(=numero di rilevazioni di mediapipe) , 33*4=132    33(numero di marker per mediapipe pose) * 4(x,y,z,visibility))
labels=np.array(labels)
print(labels)
print(type(labels))
print("numero labels: "+str(len(labels)))



X = np.array(sequences, dtype="object")
print(X.shape)



y = to_categorical(labels).astype(int)

'''
#model,X_train, X_test, y_train, y_test=train_module.train(X,y,actions)

###############################sotto commentato

model = Sequential()
model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(None,132)))
model.add(LSTM(128, return_sequences=True, activation='relu'))
model.add(LSTM(64, return_sequences=False, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(actions.shape[0], activation='softmax'))

model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

model.load_weights('action.h5')   #carica modello

import pickle

with open("x_test_out", 'rb') as infile:
	X_test = pickle.load(infile)

with open("y_test_out", 'rb') as infile:
	y_test = pickle.load(infile)
#####################fine commento

#prediction
train_module.test(X_test,y_test,model,actions)

'''
####################################                   DECISION TREE

feature_X=math_module.calculate_feature_alzateLaterali(X)

X=decisionTree_module.conversione_dataset_al(feature_X)

# Split dataset into training set and test set
'''
X_train, X_test, y_train, y_test = decisionTree_module.split(X,y)
model=decisionTree_module.train(X_train,y_train)
'''

#oppure carica dati già splittati e modello trainato
X_train, X_test, y_train, y_test,model=decisionTree_module.load_split_model()



# decisionTree_module.findBestHyperparameters(X_train, y_train, X_test, y_test)


import optuna
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
# from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation
from sklearn.metrics import accuracy_score

def objective(trial):
	dtc_params = dict(
		max_depth = trial.suggest_int("max_depth", 2, 10),
		min_samples_split = trial.suggest_int("min_samples_split", 2, 10),
		max_leaf_nodes = int(trial.suggest_int("max_leaf_nodes", 2, 10)),
		criterion = trial.suggest_categorical("criterion", ["gini", "entropy"]),
	)
	DTC = DecisionTreeClassifier(**dtc_params) # DTC con i range di parametri dati
	DTC.fit(X_train, y_train) # Training del modello con i dati 

	error = 1.0 - accuracy_score(y_test, DTC.predict(X_test))
	return error


# 3. Create a study object and optimize the objective function.
study = optuna.create_study() # di default è minimize, quindi di minimizzare l'errore
study.optimize(objective, n_trials=500)

print(study.best_params)
print(1.0 - study.best_value)






#Predict the response for test dataset
# y_pred = model.predict(X_test)

# decisionTree_module.accuracy_score(y_test, y_pred,actions)