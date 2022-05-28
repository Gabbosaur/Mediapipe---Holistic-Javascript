from sklearn.datasets import load_iris
from matplotlib import pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, cross_val_score, KFold
import numpy as np
import math_module



# Number of random trials
NUM_TRIALS = 30

# Load the dataset
X_train, X_test, y_train, y_test  = math_module.load_split_model()

# Set up possible values of parameters to optimize over
p_grid = {"kernel" : ['rbf','poly','linear','sigmoid'],"C": [1, 10, 100], "gamma": [0.01, 0.1], 'degree':[1,3]}

# We will use a Support Vector Classifier with "rbf" kernel
svm = SVC(kernel="rbf")

# Arrays to store scores
non_nested_scores = np.zeros(NUM_TRIALS)
nested_scores = np.zeros(NUM_TRIALS)
list_clf = []
# Loop for each trial
for i in range(NUM_TRIALS):

	# Choose cross-validation techniques for the inner and outer loops,
	# independently of the dataset.
	# E.g "GroupKFold", "LeaveOneOut", "LeaveOneGroupOut", etc.
	inner_cv = KFold(n_splits=5, shuffle=True, random_state=i)
	outer_cv = KFold(n_splits=5, shuffle=True, random_state=i)

	# Nested CV with parameter optimization
	clf = GridSearchCV(estimator=svm, param_grid=p_grid, cv=inner_cv)
	nested_score = cross_val_score(clf, X=X_train, y=math_module.oneHot_to_1D(y_train), cv=outer_cv)
	nested_scores[i] = nested_score.mean()
	list_clf.append(clf)

print(nested_scores)
print(np.max(nested_scores))
max_nested_score = np.max(nested_scores)
max_index = nested_scores.index(max_nested_score)

print(list_clf)