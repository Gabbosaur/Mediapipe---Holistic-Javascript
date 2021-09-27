import pandas as pd
import numpy as np
import pickle
import optuna
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.metrics import accuracy_score as AS
from sklearn.model_selection import cross_val_score
import sklearn.metrics as metrics


def train_and_score(X_train, X_test, y_train, y_test):

	clf = DecisionTreeClassifier(random_state=0) # random_state da fissare perché se no cambia valori ogni tot volte
	clf.fit(X=X_train, y=y_train)
	y_pred = clf.predict(X_test)
	score = clf.score(X_test,y_test) # testing accuracy

	ac_score = metrics.accuracy_score(y_test, y_pred)
	print("\nDecision Tree score:\t", score)
	print("metrics.accuracy score DT:\t"+ str(ac_score))
	cross_score = cross_val_score(clf, X_train, y_train, cv=5) # training accuracy
	print("cross val score DT:\t\t" + str(cross_score)) # array di 5 elementi
	print("cross val score DT mean:\t" + str(cross_score.mean()))
	print("%f accuracy with a standard deviation of %f" % (cross_score.mean(), cross_score.std()))

	return y_pred # testing accuracy


def train(X_train,y_train,best_params):
	# Create Decision Tree classifer object
	model = DecisionTreeClassifier(**best_params, random_state=0)

	model.fit(X_train, y_train) # Training del modello con i dati

	cross_score = cross_val_score(model, X_train, y_train, cv=5) # training accuracy
	print("best: %f accuracy with a standard deviation of %f" % (cross_score.mean(), cross_score.std()))

	# save the model to disk
	filename = 'decision_tree.sav'
	pickle.dump(model, open(filename, 'wb'))

	return model

def findBestHyperparameters(X_train, y_train, X_test, y_test):
	def objective(trial):
		dtc_params = dict(
			max_depth = trial.suggest_int("max_depth", 2, 100),
			min_samples_split = trial.suggest_int("min_samples_split", 2, 100),
			max_leaf_nodes = int(trial.suggest_int("max_leaf_nodes", 2, 100)),
			criterion = trial.suggest_categorical("criterion", ["gini", "entropy"]),
		)
		DTC = DecisionTreeClassifier(**dtc_params, random_state=0) # DTC con i range di parametri dati

		cross_score = cross_val_score(DTC, X_train, y_train, cv=5)
		error = 1.0 - cross_score.mean()
		return error


	# 3. Create a study object and optimize the objective function.
	study = optuna.create_study() # di default è minimize, quindi di minimizzare l'errore
	study.optimize(objective, n_trials=500)

	print(study.best_params) # Printa i migliori parametri
	print(1.0 - study.best_value) # Printa l'accuracy
	return study


