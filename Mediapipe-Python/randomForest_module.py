from sklearn.ensemble import RandomForestClassifier


def train_and_score(X_train, X_test, y_train, y_test):

    clf = RandomForestClassifier(random_state=0)
    clf.fit(X=X_train, y=y_train)
    y_pred = clf.predict(X_test)
    score = clf.score(X_test,y_test)

    return score
