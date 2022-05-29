from cv2 import mean
import optuna
from sklearn import svm
from sklearn import metrics
from sklearn.datasets import load_iris
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, make_scorer
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, cross_val_score, KFold
import numpy as np
import math_module
import math_module as mm

def findBestHyperForRMSE(X_train, y_train,cv_inner):

	y_train = mm.oneHot_to_1D(y_train)

	def objective(trial):
		dtc_params = dict(
			kernel=trial.suggest_categorical('kernel',['rbf','poly','linear','sigmoid']),
			C=trial.suggest_float("C",0.1,3.0,log=True),
			gamma=trial.suggest_categorical('gamma',['auto','scale']),
			degree=trial.suggest_int("degree",1,3,log=True),
		)
		DTC = svm.SVC(**dtc_params, random_state=0)
		cross_score = cross_val_score(DTC, X_train, y_train, cv=cv_inner,scoring=make_scorer(f1_score,average='weighted'))
		print("f1 weighted per cross val "+str(cross_score.mean()))
		error = 1.0 - cross_score.mean()
		# result = DTC.fit(X_train, y_train)
		# #best_model = result.best_estimator_
		# yhat = result.predict(X_test)
		# acc = metrics.accuracy_score(mm.oneHot_to_1D(y_test), yhat)
		# error = 1.0 - acc
		return error


	# 3. Create a study object and optimize the objective function.
	study = optuna.create_study() # di default è minimize, quindi di minimizzare l'errore
	study.optimize(objective, n_trials=150)

	print(study.best_params) # Printa i migliori parametri
	print(1.0 - study.best_value) # Printa l'accuracy
	return study


# Number of random trials
NUM_TRIALS = 30

# Load the dataset
X_ltrain, X_ltest, y_ltrain, y_ltest  = math_module.load_split_model()

# zero=0
# uno=0
# due=0
# tre=0
# for x in mm.oneHot_to_1D(y_ltrain):
# 	if x==0:
# 		zero=zero+1
# 	elif x==1:
# 		uno=uno+1
# 	elif x==2:
# 		due=due+1
# 	elif x==3:
# 		tre=tre+1

# print("classi in ltrain: zero="+str(zero)+" uno="+str(uno)+" due="+str(due)+" tre="+str(tre))
# zero=0
# uno=0
# due=0
# tre=0
# for x in mm.oneHot_to_1D(y_ltest):
# 	if x==0:
# 		zero=zero+1
# 	elif x==1:
# 		uno=uno+1
# 	elif x==2:
# 		due=due+1
# 	elif x==3:
# 		tre=tre+1
# print("classi in ltest: zero="+str(zero)+" uno="+str(uno)+" due="+str(due)+" tre="+str(tre))


# exit()
# Set up possible values of parameters to optimize over
p_grid = {"kernel" : ['rbf','poly','linear','sigmoid'],"C": [1, 10, 100], "gamma": [0.01, 0.1], 'degree':[1,3]}

# We will use a Support Vector Classifier with "rbf" kernel
#svm = SVC(kernel="rbf")

# Arrays to store scores
non_nested_scores = np.zeros(NUM_TRIALS)
nested_scores = np.zeros(NUM_TRIALS)
list_clf = []
# # Loop for each trial
# for i in range(NUM_TRIALS):

# 	# Choose cross-validation techniques for the inner and outer loops,
# 	# independently of the dataset.
# 	# E.g "GroupKFold", "LeaveOneOut", "LeaveOneGroupOut", etc.
# 	inner_cv = KFold(n_splits=5, shuffle=True, random_state=i)
# 	outer_cv = KFold(n_splits=5, shuffle=True, random_state=i)

# 	# Nested CV with parameter optimization
# 	clf = GridSearchCV(estimator=svm, param_grid=p_grid, cv=inner_cv)
# 	nested_score = cross_val_score(clf, X=X_train, y=math_module.oneHot_to_1D(y_train), cv=outer_cv)
# 	nested_scores[i] = nested_score.mean()
# 	list_clf.append(clf)

# print(nested_scores)
# print(np.max(nested_scores))
# max_nested_score = np.max(nested_scores)
# max_index = nested_scores.index(max_nested_score)

# print(list_clf)

###################################################################################################################
# cv_outer = KFold(n_splits=5, shuffle=True, random_state=1)
# # enumerate splits
# list_accuracy=list()
# list_study=list()
# list_best_acc=list()
# outer_results = list()
# i=0
# j=0
# svm_model = SVC(kernel="rbf")

# for train_ix, test_ix in cv_outer.split(X_ltrain):
	
# 	X_trainval, X_test = X_ltrain[train_ix, :], X_ltrain[test_ix, :]
# 	y_trainval, y_test = y_ltrain[train_ix], y_ltrain[test_ix]

# 	cv_inner = KFold(n_splits=5, shuffle=True, random_state=1)
	
# 	for train_ix2, val_ix2 in cv_inner.split(X_trainval):
# 		X_val, X_train = X_ltrain[train_ix2, :], X_ltrain[val_ix2, :]
# 		y_val, y_train = y_ltrain[train_ix2], y_ltrain[val_ix2]

# 		#y_train = mm.oneHot_to_1D(y_train)

# 		# print(X_train)
# 		# print("---------")
# 		# print(y_train)
# 		study=findBestHyperForRMSE(X_train, y_train,X_val,y_val)

# 		best_acc=1.0 - study.best_value
# 		best_hyper=study.best_params
# 		list_study.append(study)
# 		list_best_acc.append(best_acc)
# 		j=j+1

# 	max_mean=max(list_best_acc[j-5:j])
# 	index=list_best_acc.index(max_mean)
# 	bestHyperParam=list_study[index].best_params

# 	model = svm.SVC(**bestHyperParam, random_state=0)
# 	model.fit(X_trainval, mm.oneHot_to_1D(y_trainval))

# 	yhat = model.predict(X_test)

# 	f_score = f1_score(y_true=math_module.oneHot_to_1D(y_test), y_pred=yhat, average=None)
# 	list_accuracy.append(f_score)
# 	# list_accuracy.append(metrics.accuracy_score(yhat, mm.oneHot_to_1D(y_test)))
# 	i=i+1
# print('f-1 score: %.3f (%.3f)' % (np.mean(list_accuracy), np.std(list_accuracy)))

# print(list_accuracy)
# print(len(list_best_acc))



cv_outer = KFold(n_splits=5, shuffle=True, random_state=1)
# enumerate splits
list_f1score=list()
list_study=list()
list_best_acc=list()
outer_results = list()
i=0
j=0
svm_model = SVC(kernel="rbf")

for train_ix, test_ix in cv_outer.split(X_ltrain):
	
	X_trainval, X_test = X_ltrain[train_ix, :], X_ltrain[test_ix, :]
	y_trainval, y_test = y_ltrain[train_ix], y_ltrain[test_ix]

	cv_inner = KFold(n_splits=5, shuffle=True, random_state=1)
	
	# for train_ix2, val_ix2 in cv_inner.split(X_trainval):
	# 	X_val, X_train = X_ltrain[train_ix2, :], X_ltrain[val_ix2, :]
	# 	y_val, y_train = y_ltrain[train_ix2], y_ltrain[val_ix2]

		#y_train = mm.oneHot_to_1D(y_train)

		# print(X_train)
		# print("---------")
		# print(y_train)
	#trova i migliori hyperparameter 
	study=findBestHyperForRMSE(X_trainval, y_trainval,cv_inner)

	best_acc=1.0 - study.best_value
	best_hyper=study.best_params
	list_study.append(study)
	list_best_acc.append(best_acc)
	j=j+1

	# max_mean=max(list_best_acc[j-5:j])
	# index=list_best_acc.index(max_mean)
	#bestHyperParam=list_study[index].best_params

	model = svm.SVC(**best_hyper, random_state=0)
	model.fit(X_trainval, mm.oneHot_to_1D(y_trainval))

	yhat = model.predict(X_test)

	f_score = f1_score(y_true=math_module.oneHot_to_1D(y_test), y_pred=yhat, average=None)
	list_f1score.append(f_score)
	# list_accuracy.append(metrics.accuracy_score(yhat, mm.oneHot_to_1D(y_test)))
	i=i+1
print('SVM nested cross validation f-1 score: %.3f with standard deviation: %.3f' % (np.mean(list_f1score), np.std(list_f1score)))
