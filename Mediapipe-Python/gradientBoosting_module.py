from sklearn.ensemble import GradientBoostingClassifier
import sklearn.metrics as metrics
from sklearn.model_selection import cross_val_score
import math_module as mm
import pickle
import optuna

def train_and_score(X_train, X_test, y_train, y_test):

	y_train_new = mm.oneHot_to_1D(y_train)
	y_test_new = mm.oneHot_to_1D(y_test)

	clf = GradientBoostingClassifier(random_state=0) # random_state da fissare perché se no cambia valori ogni tot volte
	clf.fit(X=X_train, y=y_train_new)
	y_pred = clf.predict(X_test)
	score = clf.score(X_test,y_test_new) # testing accuracy

	ac_score = metrics.accuracy_score(y_test_new, y_pred)
	print("\nGradient Boosting score:\t", score)
	print("metrics.accuracy score GB:\t"+ str(ac_score))

	cross_score = cross_val_score(clf, X_train, y_train_new, cv=5) # training accuracy
	print("cross val score GB:\t\t" + str(cross_score)) # array di 5 elementi
	print("cross val score GB mean:\t" + str(cross_score.mean()))
	print("%f accuracy with a standard deviation of %f" % (cross_score.mean(), cross_score.std()))
	y_pred = mm.oneD_to_oneHot(y_pred)

	return y_pred # testing accuracy


def train(X_train,y_train,best_params):

	y_train = mm.oneHot_to_1D(y_train)

	model = GradientBoostingClassifier(**best_params, random_state=0)
	model.fit(X_train, y_train) # Training del modello con i dati

	cross_score = cross_val_score(model, X_train, y_train, cv=5) # training accuracy
	print("best: %f accuracy with a standard deviation of %f" % (cross_score.mean(), cross_score.std()))
	# save the model to disk
	filename = 'gradient_boosting.sav'
	pickle.dump(model, open(filename, 'wb'))

	return model


def findBestHyperparameters(X_train, y_train):

	y_train = mm.oneHot_to_1D(y_train)

	def objective(trial):
		dtc_params = dict(
	        max_depth=trial.suggest_int("max_depth", 2, 10),
			learning_rate=trial.suggest_float("learning_rate", 1e-4, 1e-1, log=True),
			n_estimators=trial.suggest_int("n_estimators", 100, 2000),
			subsample=trial.suggest_float("subsample", 0.2, 1.0),
		)
		DTC = GradientBoostingClassifier(**dtc_params, random_state=0)
		cross_score = cross_val_score(DTC, X_train, y_train, cv=5)
		error = 1.0 - cross_score.mean()
		return error


	# 3. Create a study object and optimize the objective function.
	study = optuna.create_study() # di default è minimize, quindi di minimizzare l'errore
	study.optimize(objective, n_trials=150)

	print(study.best_params) # Printa i migliori parametri
	print(1.0 - study.best_value) # Printa l'accuracy
	return study