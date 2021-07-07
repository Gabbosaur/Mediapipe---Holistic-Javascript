from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.utils import Sequence

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


def trainNostro(X,y,actions):

	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05)

	#x_train_tensor = tf.convert_to_tensor(X_train,dtype=float)
	#print("primo video del train")
	#print(X_train[0])
	print(type(X_train))
	#print(X_train)
	print(X_train.shape)
	X_train = np.asarray(X_train)
	print(type(X_train))
	print(type(X_train[0]))

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



class MyBatchGenerator(Sequence):
	'Generates data for Keras'
	def __init__(self, X, y, batch_size=1, shuffle=True):
		'Initialization'
		self.X = X
		self.y = y
		self.batch_size = batch_size
		self.shuffle = shuffle
		self.on_epoch_end()

	def __len__(self):
		'Denotes the number of batches per epoch'
		return int(np.floor(len(self.y)/self.batch_size))

	def __getitem__(self, index):
		return self.__data_generation(index)

	def on_epoch_end(self):
		'Shuffles indexes after each epoch'
		self.indexes = np.arange(len(self.y))
		if self.shuffle == True:
			np.random.shuffle(self.indexes)

	def __data_generation(self, index):
		Xb = np.empty((self.batch_size, *self.X[index].shape))
		yb = np.empty((self.batch_size, *self.y[index].shape))
		# naively use the same sample over and over again
		for s in range(0, self.batch_size):
			Xb[s] = self.X[index]
			yb[s] = self.y[index]
		return Xb, yb



def train(X,y,actions):
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05)

	#x_train_tensor = tf.convert_to_tensor(X_train,dtype=float)
	#print("primo video del train")
	#print(X_train[0])
	print(type(X_train))
	#print(X_train)
	print(X_train.shape)
	X_train = np.asarray(X_train)
	print(type(X_train))
	print(type(X_train[0]))

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
	print(model.summary())
	model.fit_generator(MyBatchGenerator(X_train, y_train, batch_size=1), epochs=1000, callbacks=[tb_callback])

	print(model.summary())

	model.save('action.h5')
	return model,X_train, X_test, y_train, y_test

