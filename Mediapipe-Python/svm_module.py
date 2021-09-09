from sklearn import svm
import sklearn.metrics as metrics
from sklearn.model_selection import cross_val_score

def _oneHot_to_1D(y):

    y_final = []
    for i in range(0,len(y)):
        for j in range(0, 4):
            if y[i,j] == 1:
                y_final.append(j)
                break
    
    return y_final

def train_and_score(X_train, X_test, y_train, y_test):
    
    y_train_new = _oneHot_to_1D(y_train)
    y_test_new = _oneHot_to_1D(y_test)

    clf = svm.SVC()
    clf.fit(X=X_train, y=y_train_new)
    y_pred = clf.predict(X_test)
    score = clf.score(X_test,y_test_new)

    ac_score = metrics.accuracy_score(y_test_new, y_pred)
    print("metrics.accuracy score SVM:\t"+ str(ac_score))

    cross_score = cross_val_score(clf, X_train, y_train_new, cv=5)
    print("cross val score SVM:\t\t" + str(cross_score))
    print("cross val score SVM mean:\t" + str(cross_score.mean()))

    return score
