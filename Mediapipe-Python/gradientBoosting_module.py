from sklearn.ensemble import GradientBoostingClassifier
import sklearn.metrics as metrics
from sklearn.model_selection import cross_val_score
import math_module as mm

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

    y_pred = mm.oneD_to_oneHot(y_pred)

    return y_pred # testing accuracy

