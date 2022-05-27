import numpy as np
from sklearn import svm
import sklearn.metrics as metrics
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.model_selection import cross_val_score
import math_module as mm
import pickle
import optuna


def train_and_score(X_train, X_test, y_train, y_test):
    
	y_train_new = mm.oneHot_to_1D(y_train)
	y_test_new = mm.oneHot_to_1D(y_test)

	clf = svm.SVC(random_state=0)
	clf.fit(X=X_train, y=y_train_new)
	y_pred = clf.predict(X_test)
	score = clf.score(X_test,y_test_new)

	ac_score = metrics.accuracy_score(y_test_new, y_pred)
	print("\nSVM score:\t", score)
	print("metrics.accuracy score SVM:\t"+ str(ac_score))

	cross_score = cross_val_score(clf, X_train, y_train_new, cv=5)
	print("cross val score SVM:\t\t" + str(cross_score))
	print("cross val score SVM mean:\t" + str(cross_score.mean()))
	print("%f accuracy with a standard deviation of %f" % (cross_score.mean(), cross_score.std()))

	y_pred = mm.oneD_to_oneHot(y_pred)
	return y_pred


def train(X_train,y_train,best_params):

	y_train = mm.oneHot_to_1D(y_train)
	
	model = svm.SVC(**best_params, random_state=0)
	model.fit(X_train, y_train) # Training del modello con i dati

	cross_score = cross_val_score(model, X_train, y_train, cv=5) # training accuracy
	print("best: %f accuracy with a standard deviation of %f" % (cross_score.mean(), cross_score.std()))
	# save the model to disk
	filename = 'svm.sav'
	pickle.dump(model, open(filename, 'wb'))

	return model


def findBestHyperparameters(X_train, y_train):

	y_train = mm.oneHot_to_1D(y_train)

	def objective(trial):
		dtc_params = dict(
			kernel=trial.suggest_categorical('kernel',['rbf','poly','linear','sigmoid']),
			C=trial.suggest_float("C",0.1,3.0,log=True),
			gamma=trial.suggest_categorical('gamma',['auto','scale']),
			degree=trial.suggest_int("degree",1,3,log=True),
		)
		DTC = svm.SVC(**dtc_params, random_state=0)
		cross_score = cross_val_score(DTC, X_train, y_train, cv=5)
		error = 1.0 - cross_score.mean()
		return error


	# 3. Create a study object and optimize the objective function.
	study = optuna.create_study() # di default è minimize, quindi di minimizzare l'errore
	study.optimize(objective, n_trials=150)

	print(study.best_params) # Printa i migliori parametri
	print(1.0 - study.best_value) # Printa l'accuracy
	return study


def nestedCV(X_train, y_train):

	NUM_TRIALS = 150
	p_grid = {"kernel" : ['rbf','poly','linear','sigmoid'],"C": [1, 10, 100], "gamma": [0.01, 0.1], 'degree':[1,3]}
	nested_scores = np.zeros(NUM_TRIALS)
	# Loop for each trial
	nested_clf = []
	for i in range(NUM_TRIALS):

		# Choose cross-validation techniques for the inner and outer loops,
		# independently of the dataset.
		# E.g "GroupKFold", "LeaveOneOut", "LeaveOneGroupOut", etc.
		inner_cv = KFold(n_splits=5, shuffle=True, random_state=i)
		outer_cv = KFold(n_splits=5, shuffle=True, random_state=i)

		# Nested CV with parameter optimization
		clf = GridSearchCV(estimator=svm, param_grid=p_grid, cv=inner_cv)
		y_1D = mm.oneHot_to_1D(y_train)
		nested_score = cross_val_score(clf, X=X_train, y=y_1D, cv=outer_cv)
		nested_scores[i] = nested_score.mean()

		nested_clf[i] = clf

	print("nested scores: ")
	print(nested_scores)
	print(nested_clf)