import optuna
from matplotlib import pyplot as plt
from sklearn.metrics import f1_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score, KFold
import numpy as np
from xgboost import XGBClassifier

import math_module as mm

def findBestHyperForRMSE(X_train, y_train,cv_inner):

	y_train = mm.oneHot_to_1D(y_train)

	def objective(trial):
		dtc_params = dict(
		     max_depth=trial.suggest_int("max_depth", 2, 10),
			learning_rate=trial.suggest_float("learning_rate", 1e-4, 1e-1, log=True),
			n_estimators=trial.suggest_int("n_estimators", 100, 2000),
			min_child_weight=trial.suggest_int("min_child_weight", 1, 10),
			colsample_bytree=trial.suggest_float("colsample_bytree", 0.2, 1.0),
			subsample=trial.suggest_float("subsample", 0.2, 1.0),
			reg_alpha=trial.suggest_float("reg_alpha", 1e-4, 1e2, log=True),
			reg_lambda=trial.suggest_float("reg_lambda", 1e-4, 1e2, log=True),
			gamma=trial.suggest_float("gamma", 0, 50),
		)
		DTC = XGBClassifier(**dtc_params, use_label_encoder=False, eval_metric = 'mlogloss', random_state=0)
		cross_score = cross_val_score(DTC, X_train, y_train, cv=cv_inner)
		error = 1.0 - cross_score.mean()
		return error


	# 3. Create a study object and optimize the objective function.
	study = optuna.create_study() # di default è minimize, quindi di minimizzare l'errore
	study.optimize(objective, n_trials=150)

	print(study.best_params) # Printa i migliori parametri
	print(1.0 - study.best_value) # Printa l'accuracy
	return study


# Load the dataset
X_ltrain, X_ltest, y_ltrain, y_ltest  = mm.load_split_model()


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

	model = DecisionTreeClassifier(**best_hyper, random_state=0)
	model.fit(X_trainval, y_trainval)

	yhat = model.predict(X_test)

	f_score = f1_score(y_true=mm.oneHot_to_1D(y_test), y_pred=yhat, average=None)
	list_f1score.append(f_score)
	i=i+1
print('f-1 score: %.3f (%.3f)' % (np.mean(list_f1score), np.std(list_f1score)))

print(list_f1score)
print(len(list_best_acc))