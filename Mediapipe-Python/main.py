# pip install virtualenv
# virtualenv mypython

# attivare tesivenv
# tesivenv\Scripts\activate

# pip install mediapipe opencv-python


import pathlib
import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
import math
import os
from pygame import mixer
import pygame
from scipy.sparse.sputils import matrix

from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import TensorBoard
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score as AS
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score

from sklearn import tree

from gtts import gTTS
from playsound import playsound



#nostri moduli
import Tpose_module					# Heuristics live webcam
import annotateData_module			# Annotate video and getting mediapipe structure data
import math_module					# Calcolo features (angoli)
import train_module					# LSTM
import decisionTree_module
import randomForest_module
import alzateLaterali_live_module 	# Alzate Laterali - Live webcam with ML algos
import svm_module
import gradientBoosting_module
import xgboost_module



# dimensioni finestra terminale
#cmd = 'mode 500,500'
#os.system(cmd)

#Tpose_module.Tpose()

#crea i video e i file pkl annotati dai file mp4 presenti nelle sottocartelle di alzateLaterali

#annotateData_module.createAnnotation("alzateLaterali")
sequences, labels, actions = annotateData_module.readAnnotation("alzateLaterali")

#print(len(sequences)) #187
##print(sequences)
#print(np.array(sequences[2]).shape) #(numero di frame(=numero di rilevazioni di mediapipe) , 33*4=132    33(numero di marker per mediapipe pose) * 4(x,y,z,visibility))
labels=np.array(labels)
#print(labels) #array di 0000011111112222223333
#print(type(labels)) #ndarray
#print("numero labels: "+str(len(labels))) #187


X = np.array(sequences, dtype="object")
# print(X.shape)

y = to_categorical(labels).astype(int)

'''
# # # LSTM # # #

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

feature_X = math_module.calculate_feature_alzateLaterali(X)

X = math_module.conversione_dataset_al(feature_X)

print("-------- DECISION TREE --------")

# Split dataset into training set and test set
'''
X_train, X_test, y_train, y_test = math_module.split(X,y)


study=decisionTree_module.findBestHyperparameters(X_train, y_train, X_test, y_test)
#train
model=decisionTree_module.train(X_train,y_train,study.best_params)
'''

#oppure carica dati già splittati e modello trainato
#X_train, X_test, y_train, y_test,model  = math_module.load_split_model('decision_tree.sav')
X_train, X_test, y_train, y_test  = math_module.load_split_model()
model=math_module.load_model('decision_tree.sav')

print("Normal training: ")
y_pred_DT = decisionTree_module.train_and_score(X_train, X_test, y_train, y_test)
math_module.confusionMatrix(y_test, y_pred_DT, actions)


# print("Best HP training: ")				# ----------------------------------------- scommentare per optuna -------
# # Final cross val score: 0.9443
# study=decisionTree_module.findBestHyperparameters(X_train, y_train)
# model=decisionTree_module.train(X_train, y_train,study.best_params)
# # Predict the response for test dataset
# y_pred = model.predict(X_test)
# math_module.confusionMatrix(y_test, y_pred, actions)


## Plot tree structure
# plt.figure(figsize=(13,9))
# tree.plot_tree(model, fontsize=9)
# plt.show()


# score = cross_val_score(model, X_train, y_train, cv=5)
# print("cross val score DT:\t\t", score)
# print("cross val score DT mean:\t", score.mean())

# ---------- carico miglior modello
print("\n\nBest HP training: \n")
print("-------- DECISION TREE --------")
best_model_decisionTree=math_module.load_model('decision_tree.sav')
cross_score = cross_val_score(best_model_decisionTree, X_train, y_train, cv=5)
print("Cross val score: %f accuracy with a standard deviation of %f" % (cross_score.mean(), cross_score.std()))

print("\n-------------------------------------------\n")

# Predict the response for test dataset
y_pred = best_model_decisionTree.predict(X_test)

recallW = recall_score(y_test, y_pred, average='weighted')
precisionW = precision_score(y_test, y_pred, average='weighted')
f_scoreW = f1_score(y_true=y_test, y_pred=y_pred, average='weighted')

recall = recall_score(y_test, y_pred, average=None)
precision = precision_score(y_test, y_pred, average=None)
f_score = f1_score(y_true=y_test, y_pred=y_pred, average=None)

print("Recall score:\t\t "+ str(recall) + "\tweighted average:\t" + str(recallW))
print("Precision score:\t "+ str(precision) + "\tweighted average:\t" + str(precisionW))
print("F1 score:\t\t "+ str(f_score) + "\tweighted average:\t" + str(f_scoreW))

math_module.confusionMatrix(y_test, y_pred, actions)


##########################################				RANDOM FOREST
print("\n-------- RANDOM FOREST ---------")
print("\nNormal training: ")
y_pred_RF = randomForest_module.train_and_score(X_train, X_test, y_train, y_test)

# aggiornamento iniziale: lo score cambia se non gli passo alcun random_state, se metto random_state=1 migliora a 0.91, se metto None o 0 è uguale al Decision Tree (?)
# aggiornamento 9/09/2021:	random_state = 3,42			--> score: 0.92982		<- accuracy score, non crossvalscore
# 							random_state = 0			--> score: 0.91228
#							random_state = 1,2			--> score: 0.89473
#							random_state = None			--> score: random

math_module.confusionMatrix(y_test, y_pred_RF, actions)

# Finding best HYPERPARAMETERS + Training
# print("\nBest HP training: ")		# ----------------------------------------- scommentare per optuna -------
# # Final cross val score: 0.9692307692307693, sd: 0.028782

# study=randomForest_module.findBestHyperparameters(X_train, y_train)
# model=randomForest_module.train(X_train,y_train,study.best_params)
# y_pred = model.predict(X_test)

# math_module.confusionMatrix(y_test, y_pred, actions)


# ---------- carico miglior modello
print("\n-------- RANDOM FOREST ---------")

# print("\n\nBest HP training: \n")
best_model_random_forest=math_module.load_model('random_forest.sav')
cross_score = cross_val_score(best_model_random_forest, X_train, y_train, cv=5)
print("Cross val score: %f accuracy with a standard deviation of %f" % (cross_score.mean(), cross_score.std()))

print("\n-------------------------------------------\n")

# Predict the response for test dataset
y_pred = best_model_random_forest.predict(X_test)

recallW = recall_score(y_test, y_pred, average='weighted')
precisionW = precision_score(y_test, y_pred, average='weighted')
f_scoreW = f1_score(y_true=y_test, y_pred=y_pred, average='weighted')

recall = recall_score(y_test, y_pred, average=None)
precision = precision_score(y_test, y_pred, average=None)
f_score = f1_score(y_true=y_test, y_pred=y_pred, average=None)

print("Recall score:\t\t "+ str(recall) + "\tweighted average:\t" + str(recallW))
print("Precision score:\t "+ str(precision) + "\tweighted average:\t" + str(precisionW))
print("F1 score:\t\t "+ str(f_score) + "\tweighted average:\t" + str(f_scoreW))


math_module.confusionMatrix(y_test, y_pred, actions)



# ##########################################				SUPPORT VECTOR MACHINE
print("\n-------- SVM ---------")
y_pred_SVM = svm_module.train_and_score(X_train, X_test, y_train, y_test)

# aggiornamento iniziale: 	random state = 0,1,2,42		--> score: 0.43859

math_module.confusionMatrix(y_test, y_pred_SVM, actions)

print("Best HP training: ")
# Final cross val score: 0.9923076923076923, sd: 0.015385
# study=svm_module.findBestHyperparameters(X_train, y_train)
# model=svm_module.train(X_train, y_train,study.best_params)
# y_pred = model.predict(X_test)

# math_module.confusionMatrix(y_test, math_module.oneD_to_oneHot(y_pred), actions)


# ----------  carico miglior modello
print("\n-------- SVM ---------")
# print("\n\nBest HP training: \n")
best_model_SVM=math_module.load_model('svm.sav')
cross_score = cross_val_score(best_model_SVM, X_train, math_module.oneHot_to_1D(y_train), cv=5)
print("Cross val score: %f accuracy with a standard deviation of %f" % (cross_score.mean(), cross_score.std()))

print("\n-------------------------------------------\n")

# Predict the response for test dataset
y_pred = best_model_SVM.predict(X_test)

recallW = recall_score(math_module.oneHot_to_1D(y_test), y_pred, average='weighted')
precisionW = precision_score(math_module.oneHot_to_1D(y_test), y_pred, average='weighted')
f_scoreW = f1_score(y_true=math_module.oneHot_to_1D(y_test), y_pred=y_pred, average='weighted')

recall = recall_score(math_module.oneHot_to_1D(y_test), y_pred, average=None)
precision = precision_score(math_module.oneHot_to_1D(y_test), y_pred, average=None)
f_score = f1_score(y_true=math_module.oneHot_to_1D(y_test), y_pred=y_pred, average=None)

print("Recall score:\t\t "+ str(recall) + "\tweighted average:\t" + str(recallW))
print("Precision score:\t "+ str(precision) + "\tweighted average:\t" + str(precisionW))
print("F1 score:\t\t "+ str(f_score) + "\tweighted average:\t" + str(f_scoreW))

math_module.confusionMatrix(y_test, math_module.oneD_to_oneHot(y_pred), actions)




svm_module.nestedCV(X_train, y_train)






# ##########################################				GRADIENT BOOSTING
print("\n-------- GRADIENT BOOSTING --------")
y_pred_GB = gradientBoosting_module.train_and_score(X_train, X_test, y_train, y_test)

math_module.confusionMatrix(y_test, y_pred_GB, actions)


# print("Best HP training: ")
# study=gradientBoosting_module.findBestHyperparameters(X_train, y_train)
# model=gradientBoosting_module.train(X_train, y_train,study.best_params)
# y_pred = model.predict(X_test)

# math_module.confusionMatrix(y_test, math_module.oneD_to_oneHot(y_pred), actions)

# ---------- carico miglior modello
print("\n-------- GRADIENT BOOSTING --------")

# print("\n\nBest HP training: \n")

best_model_gradient_boosting=math_module.load_model('gradient_boosting.sav')
cross_score = cross_val_score(best_model_gradient_boosting, X_train, math_module.oneHot_to_1D(y_train), cv=5)
print("Cross val score: %f accuracy with a standard deviation of %f" % (cross_score.mean(), cross_score.std()))
print("\n-------------------------------------------\n")

# Predict the response for test dataset
y_pred = best_model_gradient_boosting.predict(X_test)

recallW = recall_score(math_module.oneHot_to_1D(y_test), y_pred, average='weighted')
precisionW = precision_score(math_module.oneHot_to_1D(y_test), y_pred, average='weighted')
f_scoreW = f1_score(y_true=math_module.oneHot_to_1D(y_test), y_pred=y_pred, average='weighted')

recall = recall_score(math_module.oneHot_to_1D(y_test), y_pred, average=None)
precision = precision_score(math_module.oneHot_to_1D(y_test), y_pred, average=None)
f_score = f1_score(y_true=math_module.oneHot_to_1D(y_test), y_pred=y_pred, average=None)

print("Recall score:\t\t "+ str(recall) + "\tweighted average:\t" + str(recallW))
print("Precision score:\t "+ str(precision) + "\tweighted average:\t" + str(precisionW))
print("F1 score:\t\t "+ str(f_score) + "\tweighted average:\t" + str(f_scoreW))


math_module.confusionMatrix(y_test, math_module.oneD_to_oneHot(y_pred), actions)




##########################################				EXTREME GRADIENT BOOSTING
print("\n-------- EXTREME GRADIENT BOOSTING --------")
print("Normal training: ")
y_pred_XGB = xgboost_module.train_and_score(X_train, X_test, y_train, y_test)

math_module.confusionMatrix(y_test, y_pred_XGB, actions)

# # ----------------------------------------------------- scommentare per optuna
# print("Best HP training: ")
# study=xgboost_module.findBestHyperparameters(X_train, y_train)
# model=xgboost_module.train(X_train, y_train,study.best_params)
# y_pred = model.predict(X_test)
# print(y_pred)
# print("\n\n ^ y_pred")
# math_module.confusionMatrix(y_test, math_module.oneD_to_oneHot(y_pred), actions)


# ---------- carico miglior modello
print("\n\nBest HP training: \n")
print("\n-------- EXTREME GRADIENT BOOSTING --------")

best_model_XGB=math_module.load_model('xgboost.sav')
cross_score = cross_val_score(best_model_XGB, X_train, math_module.oneHot_to_1D(y_train), cv=5)
print("Cross val score: %f accuracy with a standard deviation of %f" % (cross_score.mean(), cross_score.std()))
# Predict the response for test dataset
y_pred = best_model_XGB.predict(X_test)


print("\n-------------------------------------------\n")
recallW = recall_score(math_module.oneHot_to_1D(y_test), y_pred, average='weighted')
precisionW = precision_score(math_module.oneHot_to_1D(y_test), y_pred, average='weighted')
f_scoreW = f1_score(y_true=math_module.oneHot_to_1D(y_test), y_pred=y_pred, average='weighted')

recall = recall_score(math_module.oneHot_to_1D(y_test), y_pred, average=None)
precision = precision_score(math_module.oneHot_to_1D(y_test), y_pred, average=None)
f_score = f1_score(y_true=math_module.oneHot_to_1D(y_test), y_pred=y_pred, average=None)

print("Recall score:\t\t "+ str(recall) + "\tweighted average:\t" + str(recallW))
print("Precision score:\t "+ str(precision) + "\tweighted average:\t" + str(precisionW))
print("F1 score:\t\t "+ str(f_score) + "\tweighted average:\t" + str(f_scoreW))


math_module.confusionMatrix(y_test, math_module.oneD_to_oneHot(y_pred), actions)





'''
#########################################				Live webcam testing
num_rep = 3

text = "Hai concluso le tue " + str(num_rep) + " ripetizioni"
tts = gTTS(text, lang='it')
tts.save("Mediapipe-Python/audio/fineAlzateLaterali.mp3")

tutte_le_rep = alzateLaterali_live_module.alzateLaterali_live(num_rep)

# init mixer and play sound
pygame.init()
mixer.init()
mixer.music.load("Mediapipe-Python/audio/fineAlzateLaterali.mp3")
mixer.music.play()

while pygame.mixer.music.get_busy():
	pygame.time.Clock().tick(10)



# calcolo features e conversione in dataframe
feature_rep = math_module.calculate_feature_alzateLaterali(tutte_le_rep)
X_rep = math_module.conversione_dataset_al(feature_rep)

esercizio = ["Esercizio con braccia piegate.", "Esercizio con braccia asimmetriche.", "Esercizio con braccia minore di 90 gradi.", "Esercizio OK."]



prediction = best_model_decisionTree.predict(X_rep) # returna array di array (one-hot encoding)
print("DT:")
print(prediction)
for i in range(0,len(prediction)):
	if 1 in prediction[i]:
		print("DT: valore predetto per campione "+ str(i)+ ": "+str(esercizio[np.argmax(prediction[i])])) #prediction
	else:
		print("DT: valore predetto per campione "+ str(i)+ ": campione nullo")

prediction = best_model_random_forest.predict(X_rep) # returna array di array (one-hot encoding)
print("RF:")
print(prediction)
for i in range(0,len(prediction)):
	if 1 in prediction[i]:
		print("RF: valore predetto per campione "+ str(i)+ ": "+str(esercizio[np.argmax(prediction[i])])) #prediction
	else:
		print("RF: valore predetto per campione "+ str(i)+ ": campione nullo")

prediction = best_model_SVM.predict(X_rep) # returna array monodimensionale
print("SVM:")
print(prediction)
text = "Risultato: "
for i in range(0,len(prediction)):
	print("SVM: valore predetto per campione "+ str(i)+ ": "+ str(esercizio[prediction[i]])) #prediction
	text = text + " ripetizione numero " + str(i+1) + ": " + str(esercizio[prediction[i]]+"\n")
tts = gTTS(text, lang='it')
tts.save("Mediapipe-Python/audio/outcomeAlzateLaterali.mp3")


prediction = best_model_gradient_boosting.predict(X_rep) # returna array monodimensionale
print("GB:")
print(prediction)
for i in range(0,len(prediction)):
	print("GB: valore predetto per campione "+ str(i)+ ": "+str(esercizio[prediction[i]])) #prediction

prediction = best_model_XGB.predict(X_rep) # returna array monodimensionale
print("XGB:")
print(prediction)
for i in range(0,len(prediction)):
	print("XGB: valore predetto per campione "+ str(i)+ ": "+str(esercizio[prediction[i]])) #prediction


# play sound
mixer.music.load("Mediapipe-Python/audio/outcomeAlzateLaterali.mp3")
mixer.music.play()

while pygame.mixer.music.get_busy():
	pygame.time.Clock().tick(10)

'''

'''
0-->0-52 braccia piegate
1-->53-95 braccia asimmetriche
2-->96-141 no 90 gradi
3-->142-187 ok
'''
'''
#calcolo feature
print("... sto calcolando le features ...")
feature_X=math_module.calculate_feature_alzateLaterali(record_movimento)
print("featureX: " + feature_X)

X=decisionTree_module.conversione_dataset_al(feature_X)

prediction = model.predict(X) # array di probabilità. for example: [0 0.3 0.4 0.3] e prendiamo con argmax l'indice con il valore più alto
print("prediction: " + prediction)
print("valore predetto: " + str(actions[np.argmax(prediction)])) #prediction
'''