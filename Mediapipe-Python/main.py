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
import pandas
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

#Predict the response for test dataset
y_pred = model.predict(X_test)



print("\nPREDICTIONS\n")
decisionTree_module.accuracy_score(y_test, y_pred)

matrix = [[0 for x in range(4)] for y in range(4)]
errori=0
for i in range(0,len(y_test)):
	if(np.argmax(y_pred[i]) != np.argmax(y_test[i])):
		matrix[np.argmax(y_test[i])][np.argmax(y_pred[i])]=matrix[np.argmax(y_test[i])][np.argmax(y_pred[i])] + 1
		errori=errori+1
	#print("valore predetto per campione "+ str(i)+ ": "+str(actions[np.argmax(y_pred[i])])) #prediction
	#print("valore effettivo per campione "+ str(i)+ ": "+str(actions[np.argmax(y_test[i])])+"\n") #valore effettivo


for i in range(4):
	matrix[i][i]="X"

import pandas as pd
mat=pd.DataFrame(np.row_stack(matrix))
mat.columns=["predict as "+str(actions[0]), "predict as "+str(actions[1]), "predict as "+str(actions[2]), "predict as "+str(actions[3])]
mat.index=[actions[0], actions[1], actions[2], actions[3]]


print("error matrix")
print("numero campioni di test: "+str(len(y_pred))+"   campioni erroneamente classificati: "+str(errori)+"\n")
print(mat)


