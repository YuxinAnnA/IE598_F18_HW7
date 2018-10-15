import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score


# import dataset and split it into training and testing
df_wine = pd.read_csv('/Users/yuxin/Desktop/course/2018Fall/IE598/HW/HW5/wine.csv')
X, y = df_wine.iloc[:, 0:12].values, df_wine['Class']
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.1,
                                                    random_state=42,
                                                    stratify=y)

# part 1
n_estimators = np.arange(1,1500,500)
score = []
for k in n_estimators:
    forest = RandomForestClassifier(n_estimators=k, random_state=42)
    forest.fit(X_train, y_train)
    train_pred = forest.predict(X_train)
    score = cross_val_score(estimator=forest, X=X_train, y=y_train, cv=10, n_jobs=1)
    print sum(score)/10

# part 2

feat_labels = df_wine.columns[1:]
forest = RandomForestClassifier(n_estimators=500,random_state=42)
forest.fit(X_train, y_train)
importances = forest.feature_importances_
indices = np.argsort(importances)[::-1]
for f in range(X_train.shape[1]):
    print("%2d) %-*s %f" % (f + 1, 30,feat_labels[indices[f]],importances[indices[f]]))


print("My name is Yuxin Sun")
print("My NetID is: yuxins5")
print("I hereby certify that I have read the University policy on Academic Integrity and that I am not in violation.")