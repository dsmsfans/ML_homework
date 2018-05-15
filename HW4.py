import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
import random as rd
import scipy as sc
import math
sns.set_palette('husl')
import matplotlib.pyplot as plt
import heapq as he
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import FastICA
from sklearn import cluster
from sklearn.mixture import GaussianMixture

#---------------------------------------------SOFM------------------------------------------
t = 0
learning_rate = 1
while learning_rate > 0.001:
    learning_rate = 0.1*math.exp(t/10)*math.exp(-1/(2*(0.1*math.exp(-t/10))**2))
    t = t + 1
print("total: ",t)
print("final learning_rate: ",learning_rate)
print("-----------------------------------------------")
#-----------------------------------------gradient decent-----------------------------------

g_learning = 0.1

x = rd.random()
y = rd.random()
landa = rd.random()

print("initial value:")
print("x:",x)
print("y ",y)
print("lambda ",landa)
print("-----------------------------------------------")
def lam(x,y,landa):
    result = x + y - landa * (x ** 2 + y ** 2 - 1)
    return result

for i in range(10):
    x = x + g_learning * (1 - 2 * landa * x)
    y = y + g_learning * (1 - 2 * landa * y)
    landa = landa + g_learning * (-x ** 2 - y ** 2 + 1)
    k = lam(x,y,landa)
    print(k,x,y,landa)
print("-----------------------------------------------") 
#--------------------------------------back propagation-------------------------------------
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
data = pd.read_csv('Iris.csv', names=names)
data = data[50:150]
