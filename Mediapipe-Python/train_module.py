from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import TensorBoard
import tensorflow as tf
import numpy as np
import os

def train_generator(X_train, y_train):
	X_iterator = iter(X_train)
	y_iterator = iter(y_train)
	'''
	print("x")
	print(X_iterator)
	print("y")
	print(y_iterator)

	for i in y_iterator:
		print(i)
	k=0
	for h in X_iterator:
		print("array numero" +str(k))
		print(h)
		k=k+1
		if(k==10):
			break
	'''
	for i,j in X_iterator, y_iterator:
		yield (i,j)

def test_generator(X_test):
	X_iterator = iter(X_test)
	for i in X_iterator:
		yield i


'''
def train_generator(X_train, y_train):
	i=0
	while True:
		yield X_train[i],y_train[i]
		i=i+1


def test_generator(X_test):
	i=0
	while True:
		yield X_test[i]
		i=i+1
'''


def train(X,y,actions):

	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05)

	#x_train_tensor = tf.convert_to_tensor(X_train,dtype=float)
	print("primo video del train")
	print(X_train[0])
	print(type(X_train))

	log_dir = os.path.join('Logs')
	tb_callback = TensorBoard(log_dir=log_dir)

	#array di liste di array

	model = Sequential()
	model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(None,132)))
	model.add(LSTM(128, return_sequences=True, activation='relu'))
	model.add(LSTM(64, return_sequences=False, activation='relu'))
	model.add(Dense(64, activation='relu'))
	model.add(Dense(32, activation='relu'))
	model.add(Dense(actions.shape[0], activation='softmax'))

	model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])


	#model.fit(X_train, y_train, epochs=2000, callbacks=[tb_callback])
	model.fit(X_train, y_train, batch_size=1 , epochs=2000,callbacks=[tb_callback])

	print(model.summary())

	model.save('action.h5')
	return model,X_train, X_test, y_train, y_test



