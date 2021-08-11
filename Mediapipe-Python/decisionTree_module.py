import pandas as pd
import numpy as np
import pickle
import optuna
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn.metrics import accuracy_score as AS
import sklearn.metrics as metrics

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



def accuracy_score(y_test, y_pred, actions):
	print("\nPREDICTIONS\n")
	# Model Accuracy, how often is the classifier correct?
	print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

	matrix = [[0 for x in range(len(actions))] for y in range(len(actions))]
	errori=0
	for i in range(0,len(y_test)):
		if(np.argmax(y_pred[i]) != np.argmax(y_test[i])):
			matrix[np.argmax(y_test[i])][np.argmax(y_pred[i])]=matrix[np.argmax(y_test[i])][np.argmax(y_pred[i])] + 1
			errori=errori+1
		#print("valore predetto per campione "+ str(i)+ ": "+str(actions[np.argmax(y_pred[i])])) #prediction
		#print("valore effettivo per campione "+ str(i)+ ": "+str(actions[np.argmax(y_test[i])])+"\n") #valore effettivo


	for i in range(len(actions)):
		matrix[i][i]="X"

	mat=pd.DataFrame(np.row_stack(matrix))
	col=[]
	for i in range(len(actions)):
		col.append(actions[i])

	mat.columns=col
	mat.index=actions

	print("numero campioni di test: "+str(len(y_pred))+"   campioni erroneamente classificati: "+str(errori)+"\n")
	print(mat)


def train(X_train,y_train,best_params):
	# Create Decision Tree classifer object
	model = DecisionTreeClassifier(**best_params)

	model.fit(X_train, y_train) # Training del modello con i dati


	# save the model to disk
	filename = 'decision_tree.sav'
	pickle.dump(model, open(filename, 'wb'))

	return model

def findBestHyperparameters(X_train, y_train, X_test, y_test):
	def objective(trial):
		dtc_params = dict(
			max_depth = trial.suggest_int("max_depth", 2, 10),
			min_samples_split = trial.suggest_int("min_samples_split", 2, 10),
			max_leaf_nodes = int(trial.suggest_int("max_leaf_nodes", 2, 10)),
			criterion = trial.suggest_categorical("criterion", ["gini", "entropy"]),
		)
		DTC = DecisionTreeClassifier(**dtc_params) # DTC con i range di parametri dati
		DTC.fit(X_train, y_train) # Training del modello con i dati

		error = 1.0 - AS(y_test, DTC.predict(X_test))
		return error


	# 3. Create a study object and optimize the objective function.
	study = optuna.create_study() # di default è minimize, quindi di minimizzare l'errore
	study.optimize(objective, n_trials=500)

	print(study.best_params) # Printa i migliori parametri
	print(1.0 - study.best_value) # Printa l'accuracy
	return study


