from cv2 import mean
import optuna
from sklearn import metrics
from sklearn.datasets import load_iris
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, make_scorer, precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score, KFold
import numpy as np
import math_module
import math_module as mm
#############################################à
import annotateData_module

sequences, labels, actions = annotateData_module.readAnnotation("alzateLaterali")

def findBestHyperForRMSE(X_train, y_train,cv_inner):

	#y_train = mm.oneHot_to_1D(y_train)

	def objective(trial):
		dtc_params = dict(
			max_depth = trial.suggest_int("max_depth", 2, 30),
			min_samples_split = trial.suggest_int("min_samples_split", 2, 15),
			min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 15),
			n_estimators = trial.suggest_int("n_estimators", 100, 1000),
		)
		DTC = RandomForestClassifier(**dtc_params, random_state=0)
		cross_score = cross_val_score(DTC, X_train, y_train, cv=cv_inner,scoring=make_scorer(f1_score,average='weighted'))
		print("f1 weighted per cross val "+str(cross_score.mean()))
		error = 1.0 - cross_score.mean()
		return error


	# 3. Create a study object and optimize the objective function.
	study = optuna.create_study() # di default è minimize, quindi di minimizzare l'errore
	study.optimize(objective, n_trials=150)

	print()
	print(study.best_params) # Printa i migliori parametri
	# print(1.0 - study.best_value) # Printa l'accuracy
	return study


# Load the dataset
X_ltrain, X_ltest, y_ltrain, y_ltest  = math_module.load_split_model()


################################################
X_ltrain = np.concatenate((X_ltrain, X_ltest))
y_ltrain = np.concatenate((y_ltrain, y_ltest))

cv_outer = KFold(n_splits=5, shuffle=True, random_state=1)
# enumerate splits
list_f1score=list()
list_study=list()
list_best_acc=list()
outer_results = list()
i=0
j=0

for train_ix, test_ix in cv_outer.split(X_ltrain):
	
	X_trainval, X_test = X_ltrain[train_ix, :], X_ltrain[test_ix, :]
	y_trainval, y_test = y_ltrain[train_ix], y_ltrain[test_ix]

	cv_inner = KFold(n_splits=5, shuffle=True, random_state=1)
	
	study=findBestHyperForRMSE(X_trainval, y_trainval,cv_inner)

	best_acc=1.0 - study.best_value
	best_hyper=study.best_params
	list_study.append(study)
	list_best_acc.append(best_acc)
	j=j+1

	model = RandomForestClassifier(**best_hyper, random_state=0)
	model.fit(X_trainval, y_trainval)

	yhat = model.predict(X_test)

	
	###############################################################
	recallW = recall_score(y_test, yhat, average='weighted')
	precisionW = precision_score(y_test, yhat, average='weighted')
	f_scoreW = f1_score(y_true=y_test, y_pred=yhat, average='weighted')

	recall = recall_score(y_test, yhat, average=None)
	precision = precision_score(y_test, yhat, average=None)
	
	f_score = f1_score(y_true=y_test, y_pred=yhat, average=None)
	list_f1score.append(f_score)
	print("\n-------- RANDOM FOREST --------\n")
	print("Recall score:\t\t "+ str(recall) + "\tweighted average:\t" + str(recallW))
	print("Precision score:\t "+ str(precision) + "\tweighted average:\t" + str(precisionW))
	print("F1 score:\t\t "+ str(f_score) + "\tweighted average:\t" + str(f_scoreW))

	math_module.confusionMatrix(y_test, math_module.oneD_to_oneHot(yhat), actions)


	i=i+1

print('\nRF nested cross validation f-1 score: %.3f with standard deviation: %.3f\n' % (np.mean(list_f1score), np.std(list_f1score)))
