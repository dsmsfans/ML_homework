import numpy as np
import pandas as pd
import seaborn as sns
import random as rd
sns.set_palette('husl')
import matplotlib.pyplot as plt
#matplotlib inline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
data = pd.read_csv('Iris.csv', names=names)

kaverage = []
kaverage2 = []
naive = []
seed = []
for i in range(10):
    seed.append(rd.randint(0,100))
for k in range(1,16,2):   #k from 1 to 15 
    total1 = 0
    total2 = 0
    #seed = rd.randint(0,100)
    for i in range(10):  
        array = data.values
        X = array[:,0:4]
        Y = array[:,4]
        scoring = 'accuracy'
        #main
        X_train, X_validation, Y_train, Y_validation = train_test_split(X, Y, test_size=0.3,random_state=seed[i])
        knn = KNeighborsClassifier(n_neighbors = k)
        knn.fit(X_train, Y_train)
        predictions = knn.predict(X_validation)
        acc1 = accuracy_score(Y_validation, predictions)
        total1 = total1 + acc1
        #sub_problem
        X2_train, X2_validation, Y2_train, Y2_validation = train_test_split(X_train, Y_train, test_size=0.3,random_state=seed[i])
        knn = KNeighborsClassifier(n_neighbors = k)
        knn.fit(X2_train, Y2_train)
        predictions2 = knn.predict(X2_validation)
        acc2 = accuracy_score(Y2_validation, predictions2)
        total2 = total2 + acc2
        
        
    kaverage.append(total1/10)
    kaverage2.append(total2/10)

total3 = 0
for i in range(10):
    array = data.values
    X = array[:,0:4]
    Y = array[:,4]
    
    #naive
    X3_train, X3_validation, Y3_train, Y3_validation = train_test_split(X, Y, test_size=0.3,random_state = seed[i])
    clf = GaussianNB()
    clf.fit(X3_train, Y3_train)
    predictions3 = clf.predict(X3_validation)
    acc3 = accuracy_score(Y3_validation, predictions3)
    total3 = total3 + acc3
    
    
for i in range(8):
    naive.append(total3 / 10)

x = [1,3,5,7,9,11,13,15]
y1 = kaverage  
y2 = kaverage2
y3 = naive
plt.plot(x,y1,'oy-',label="first knn")
plt.plot(x,y2,'og-',label="second knn")
plt.plot(x,y3,'or--',label="naive")
plt.legend()
plt.xlabel("k's value")
plt.ylabel("accuracy")   
plt.show()   


#print(acc)
#print(confusion_matrix(Y_validation, predictions))
#print(classification_report(Y_validation, predictions))