
import numpy as np
from sklearn import cross_validation
from sklearn import datasets
from sklearn import svm
from sklearn import metrics
import matplotlib.pyplot as plt

import pandas as pd

pd.read_table('ecrimes.csv')
ecrimes = pd.read_table('ecrimes.csv', sep=',')


ecrimes.drop(['datex', 'Datxe'], axis=1, inplace=True)
ecrimes.drop(['Coffense'], axis=1, inplace=True)
ecrimes['Cshift'] = ecrimes.SHIFT
ecrimes.Cshift.replace('DAY', '1', inplace=True)
ecrimes.Cshift.replace('EVENING', '2', inplace=True)
ecrimes.Cshift.replace('MIDNIGHT', '3', inplace=True)

ecrimes['Coffense'] = ecrimes.OFFENSE
ecrimes.Coffense.replace('THEFT/OTHER', '1', inplace=True)
ecrimes.Coffense.replace('THEFT F/AUTO', '2', inplace=True)
ecrimes.Coffense.replace('ROBBERY', '3', inplace=True)
ecrimes.Coffense.replace('BURGLARY', '4', inplace=True)
ecrimes.Coffense.replace('MOTOR VEHICLE THEFT', '5', inplace=True)
ecrimes.Coffense.replace('ASSAULT W/DANGEROUS WEAPON ', '6', inplace=True)
ecrimes.Coffense.replace('SEX ABUSE', '7', inplace=True)
ecrimes.Coffense.replace('HOMICIDE', '8', inplace=True)
ecrimes.Coffense.replace('ARSON' , '9', inplace=True)


ecrimes.dtypes
feature_cols = ['Fahrenheit', 'WARD', 'cshift','BLOCKXCOORD','BLOCKYCOORD','year']
X = ecrimes[feature_cols]
y = ecrimes.Coffense

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=5)

from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression(C=1e9)
logreg.fit(X_train, y_train)

logreg.coef_

y_pred_class = logreg.predict(X_test)
y_pred_prob = logreg.predict_proba(X_test)[:, 1]

from sklearn import metrics
metrics.accuracy_score(y_test, y_pred_class)    
metrics.confusion_matrix(y_test, y_pred_class)  
           

from sklearn.cross_validation import cross_val_score
tat = cross_val_score(logreg, X, y, scoring='log_loss', cv=10)
tat.mean()    
tat.std()

feature_cols = ['Fahrenheit', 'WARD', 'cshift','BLOCKXCOORD','BLOCKYCOORD','year']
X = ecrimes[feature_cols]
cross_val_score(logreg, X, y, scoring='log_loss', cv=10).mean() 

from sklearn.linear_model import LinearRegression
lm = LinearRegression()
scores = cross_val_score(lm, X, y, cv=10, scoring='mean_squared_error')
print scores


ecrimes.dtypes
feature_cols = ['Fahrenheit', 'WARD', 'Coffense','BLOCKXCOORD','BLOCKYCOORD','year']
X = ecrimes[feature_cols]
y = ecrimes.cshift

from sklearn.cross_validation import train_test_split
from sklearn.neighbors import KNeighborsClassifier
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=4)


knn = KNeighborsClassifier(n_neighbors=20)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
print metrics.accuracy_score(y_test, y_pred)

from sklearn.cross_validation import KFold
kf = KFold(25, n_folds=5, shuffle=False)

print '{} {:^61} {}'.format('Iteration', 'Training set observations', 'Testing set observations')
for iteration, data in enumerate(kf, start=1):
    print '{:^9} {} {:^25}'.format(iteration, data[0], data[1])
    
k_range = range(1, 80)
k_scores = []
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X, y, cv=10, scoring='accuracy')
    k_scores.append(scores.mean())
print k_scores


plt.plot(k_range, k_scores)
plt.xlabel('Value of K for KNN')
plt.ylabel('Cross-Validated Accuracy')

knn = KNeighborsClassifier(n_neighbors=60)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
print metrics.accuracy_score(y_test, y_pred) 

ecrimes.boxplot(column='Fahrenheit', by='Coffense')

pd.scatter_matrix(ecrimes[['OFFENSE', 'cshift']])
plt.xlabel('SHIFT')
plt.ylabel('OFFENSES')



y_pred_prob1 = logreg.predict_proba(X_test)[:, 1]
y_pred_prob1[:10] 

%matplotlib inline
import matplotlib.pyplot as plt
plt.hist(y_pred_prob1)

import numpy as np
y_pred_prob2 = np.sqrt(y_pred_prob1)
y_pred_prob2[:10]