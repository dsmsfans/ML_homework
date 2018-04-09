import numpy as np
import pandas as pd
import seaborn as sns
import random as rd
import scipy as sc
sns.set_palette('husl')
import matplotlib.pyplot as plt
#matplotlib inline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA,FactorAnalysis
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
data = pd.read_csv('Iris.csv', names=names)

class1 = data.values[0:50,0:4]
class2 = data.values[50:100,0:4]
class3 = data.values[100:150,0:4]

class1 = np.double(class1)
class2 = np.double(class2)
class3 = np.double(class3)

total1 = np.zeros(4)
total2 = np.zeros(4)
total3 = np.zeros(4)

mean1 = np.mean(class1, axis = 0)
mean2 = np.mean(class2, axis = 0)
mean3 = np.mean(class3, axis = 0)

cov1 = np.zeros((4,4))
cov2 = np.zeros((4,4))
cov3 = np.zeros((4,4))

a1 = np.zeros((4,4))
a2 = np.zeros((4,4))
a3 = np.zeros((4,4))

for i in range(50):
    a1 = class1[i].reshape(4,1) - mean1.reshape(4,1)
    cov1 = a1 * a1.T + cov1
    a2 = class2[i].reshape(4,1) - mean2.reshape(4,1)
    cov2 = a2 * a2.T + cov2
    a3 = class3[i].reshape(4,1) - mean3.reshape(4,1)
    cov3 = a3 * a3.T + cov3
    
cov1 = cov1 / 50
cov2 = cov2 / 50
cov3 = cov3 / 50

eig1 = np.linalg.eigvals(cov1)
eig2 = np.linalg.eigvals(cov2)
eig3 = np.linalg.eigvals(cov3)

k1 = abs(max(eig1) / min(eig1))
k2 = abs(max(eig2) / min(eig2))
k3 = abs(max(eig3) / min(eig3))

#print(k1,k2,k3)



#----------------------------------PCA-----------------------------------------

total = []
seed = []
for i in range(10):
    seed.append(rd.randint(0,100))
for i in range(10):  
        array = data.values
        X = array[:,0:4]
        Y = array[:,4]
        X_train, X_validation, Y_train, Y_validation = train_test_split(X, Y, test_size=0.3,random_state=seed[i])

        pca = PCA(n_components = 2)
        X_train_pca = pca.fit_transform(X_train)
        X_validation_pca = pca.transform(X_validation)

        knn = KNeighborsClassifier(n_neighbors = 3)
        knn.fit(X_train, Y_train)
        predictions = knn.predict(X_validation)
        acc1 = accuracy_score(Y_validation, predictions)
        total.append(acc1)
        
#print("varinace: ",np.var(total))
#print("accuracy: ",np.mean(total))


#----------------------------------FA-----------------------------------------

fa = FactorAnalysis(n_components = 2)

data_fa = fa.fit_transform(data.values[:,0:4])

class1 = data_fa[0:50,0:4]
class2 = data_fa[50:100,0:4]
class3 = data_fa[100:150,0:4]

class1 = np.double(class1)
class2 = np.double(class2)
class3 = np.double(class3)

total1 = np.zeros(4)
total2 = np.zeros(4)
total3 = np.zeros(4)

mean1 = np.mean(class1, axis = 0)
mean2 = np.mean(class2, axis = 0)
mean3 = np.mean(class3, axis = 0)

cov1 = np.zeros((2,2))
cov2 = np.zeros((2,2))
cov3 = np.zeros((2,2))

a1 = np.zeros((4,4))
a2 = np.zeros((4,4))
a3 = np.zeros((4,4))

for i in range(50):
    a1 = class1[i].reshape(2,1) - mean1.reshape(2,1)
    cov1 = a1 * a1.T + cov1
    a2 = class2[i].reshape(2,1) - mean2.reshape(2,1)
    cov2 = a2 * a2.T + cov2
    a3 = class3[i].reshape(2,1) - mean3.reshape(2,1)
    cov3 = a3 * a3.T + cov3
    
cov1 = cov1 / 50
cov2 = cov2 / 50
cov3 = cov3 / 50

eig1 = np.linalg.eigvals(cov1)
eig2 = np.linalg.eigvals(cov2)
eig3 = np.linalg.eigvals(cov3)

k1 = abs(max(eig1) / min(eig1))
k2 = abs(max(eig2) / min(eig2))
k3 = abs(max(eig3) / min(eig3))

#print(k1,k2,k3)

#----------------------------------LDA----------------------------------------

lda = LinearDiscriminantAnalysis(n_components = 2)

data_lda = lda.fit_transform(data.values[:,0:4],data.values[:,4])

class1 = data_lda[0:50,0:4]
class2 = data_lda[50:100,0:4]
class3 = data_lda[100:150,0:4]

class1 = np.double(class1)
class2 = np.double(class2)
class3 = np.double(class3)

total1 = np.zeros(4)
total2 = np.zeros(4)
total3 = np.zeros(4)

mean1 = np.mean(class1, axis = 0)
mean2 = np.mean(class2, axis = 0)
mean3 = np.mean(class3, axis = 0)

cov1 = np.zeros((2,2))
cov2 = np.zeros((2,2))
cov3 = np.zeros((2,2))

a1 = np.zeros((4,4))
a2 = np.zeros((4,4))
a3 = np.zeros((4,4))

for i in range(50):
    a1 = class1[i].reshape(2,1) - mean1.reshape(2,1)
    cov1 = a1 * a1.T + cov1
    a2 = class2[i].reshape(2,1) - mean2.reshape(2,1)
    cov2 = a2 * a2.T + cov2
    a3 = class3[i].reshape(2,1) - mean3.reshape(2,1)
    cov3 = a3 * a3.T + cov3
    
cov1 = cov1 / 50
cov2 = cov2 / 50
cov3 = cov3 / 50

eig1 = np.linalg.eigvals(cov1)
eig2 = np.linalg.eigvals(cov2)
eig3 = np.linalg.eigvals(cov3)

k1 = abs(max(eig1) / min(eig1))
k2 = abs(max(eig2) / min(eig2))
k3 = abs(max(eig3) / min(eig3))

print(k1,k2,k3)
