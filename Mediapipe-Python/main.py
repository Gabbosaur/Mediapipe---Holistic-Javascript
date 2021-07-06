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


from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import TensorBoard

#nostri moduli
import Tpose_module
import annotateData_module
import train_module

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
print(len(labels))



X = np.array(sequences, dtype="object")
print(X.shape)



y = to_categorical(labels).astype(int)

model,X_train, X_test, y_train, y_test=train_module.train(X,y,actions)

###############################sotto commentato
'''
	model = Sequential()
	model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(None,132)))
	model.add(LSTM(128, return_sequences=True, activation='relu'))
	model.add(LSTM(64, return_sequences=False, activation='relu'))
	model.add(Dense(64, activation='relu'))
	model.add(Dense(32, activation='relu'))
	model.add(Dense(actions.shape[0], activation='softmax'))

	model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

	#model.load_weights('action.h5')   #carica modello
'''
#####################fine commento

#prediction
res = model.predict_generator(train_module.test_generator(X_test))

print(actions[np.argmax(res[4])]) #prediction

print(actions[np.argmax(y_test[4])]) #valore effettivo


