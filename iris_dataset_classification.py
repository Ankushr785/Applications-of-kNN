# -*- coding: utf-8 -*-
"""
Created on Wed May 17 16:47:45 2017

@author: Ankush Raut
"""
import numpy as np
from sklearn import datasets
iris = datasets.load_iris()

x = iris.data
y= iris.target

x_train = x[:130]
y_train = y[:130]



x_validate = x[130:]
y_validate = y[130:]

from sklearn.neighbors import KNeighborsClassifier

neighbor = KNeighborsClassifier(n_neighbors = 7, weights = 'distance')
model = neighbor.fit(x_train, y_train)

from sklearn.cross_validation import cross_val_score
score = cross_val_score(model, x_train, y_train, cv = 10)
from scipy.stats import sem
def mean_scores(scores):
    return("Mean score: {0:.3f}(+/-{1:.3f})").format(np.mean(scores), sem(scores))
    
print(mean_scores(score))


predicted = neighbor.predict(x_validate)
print(predicted)

diff = 0
for i in range(len(predicted)):
    if predicted[i]!=y_validate[i]:
        diff+=1
    else:
        diff+=0
        
accuracy = 1-(diff/len(predicted))
print(accuracy)