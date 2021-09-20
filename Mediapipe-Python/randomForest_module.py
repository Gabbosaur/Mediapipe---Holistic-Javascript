from sklearn.ensemble import RandomForestClassifier
import sklearn.metrics as metrics
import pickle
import optuna
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score as AS
import math_module

def train_and_score(X_train, X_test, y_train, y_test):

    clf = RandomForestClassifier(random_state=0) # random_state da fissare perché se no cambia valori ogni tot volte
    clf.fit(X=X_train, y=y_train)
    y_pred = clf.predict(X_test)
    score = clf.score(X_test,y_test) # testing accuracy

    ac_score = metrics.accuracy_score(y_test, y_pred)
    print("\nRandom Forest score:\t", score)
    print("metrics.accuracy score RF:\t"+ str(ac_score))

    cross_score = cross_val_score(clf, X_train, y_train, cv=5) # training accuracy
    print("cross val score RF:\t\t" + str(cross_score)) # array di 5 elementi
    print("cross val score RF mean:\t" + str(cross_score.mean()))

    return y_pred # testing accuracy


def train(X_train,y_train,best_params):

	model = RandomForestClassifier(**best_params, random_state=0)
	model.fit(X_train, y_train) # Training del modello con i dati

	# save the model to disk
	filename = 'random_forest.sav'
	pickle.dump(model, open(filename, 'wb'))

	return model


def findBestHyperparameters(X_train, y_train, X_test, y_test):
	def objective(trial):
		dtc_params = dict(
			max_depth = trial.suggest_int("max_depth", 2, 50),
			min_samples_split = trial.suggest_int("min_samples_split", 2, 150),
			min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 60),
			n_estimators = trial.suggest_int("n_estimators", 50, 1000),
		)
		DTC = RandomForestClassifier(**dtc_params) # DTC con i range di parametri dati
		DTC.fit(X_train, y_train) # Training del modello con i dati


		# error = 1.0 - AS(y_test, DTC.predict(X_test))
		error = 1.0 - metrics.accuracy_score(y_test, DTC.predict(X_test))
		return error


	# 3. Create a study object and optimize the objective function.
	study = optuna.create_study() # di default è minimize, quindi di minimizzare l'errore
	study.optimize(objective, n_trials=200)

	print(study.best_params) # Printa i migliori parametri
	print(1.0 - study.best_value) # Printa l'accuracy
	return study