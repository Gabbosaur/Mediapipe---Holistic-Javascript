from sklearn.ensemble import RandomForestClassifier
import sklearn.metrics as metrics
from sklearn.model_selection import cross_val_score

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

