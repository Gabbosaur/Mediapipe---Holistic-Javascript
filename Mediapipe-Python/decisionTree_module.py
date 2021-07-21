import pandas as pd
import numpy as np
import pickle
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation


def conversione_dataset_al(x):
	X=pd.DataFrame(np.row_stack(x))
	X.columns=["coeff_schiena", "coeff_braccia", "coeff_angolo_medio_braccio_destro", "coeff_angolo_medio_braccio_sinistro", "coeff_simm_x", "coeff_simm_y", "angolo_spalla_destra", "angolo_spalla_sinistra"]
	
	return X

def split(X,y):
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1) # 70% training and 30% test

	pd.DataFrame(X_train).to_csv('X_train_dec_tree.csv',index=False)
	pd.DataFrame(X_test).to_csv('X_test_dec_tree.csv',index=False)
	pd.DataFrame(y_train).to_csv('y_train_dec_tree.csv',index=False)
	pd.DataFrame(y_test).to_csv('y_test_dec_tree.csv',index=False)

	return X_train, X_test, y_train, y_test

def load_split_model():
	X_train = pd.read_csv('X_train_dec_tree.csv')
	X_test=pd.read_csv('X_test_dec_tree.csv')
	y_train=pd.read_csv('y_train_dec_tree.csv')
	y_test=pd.read_csv('y_test_dec_tree.csv')


	# load the model from disk
	loaded_model = pickle.load(open('decision_tree.sav', 'rb'))

	return X_train.to_numpy(), X_test.to_numpy(), y_train.to_numpy(), y_test.to_numpy(), loaded_model



def accuracy_score(y_test, y_pred):
	# Model Accuracy, how often is the classifier correct?
	print("Accuracy:",metrics.accuracy_score(y_test, y_pred))


def train(X_train,y_train):
	# Create Decision Tree classifer object
	clf = DecisionTreeClassifier()

	# Train Decision Tree Classifer
	clf = clf.fit(X_train,y_train)

	# save the model to disk
	filename = 'decision_tree.sav'
	pickle.dump(clf, open(filename, 'wb'))

	return clf

