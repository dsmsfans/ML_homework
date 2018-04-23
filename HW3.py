import numpy as np
import pandas as pd
import seaborn as sns
import random as rd
sns.set_palette('husl')
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.decomposition import FastICA
from sklearn import cluster
from sklearn.mixture import GaussianMixture
from sklearn import preprocessing
from sklearn import tree
import graphviz

#--------------------------------------------------ICA-------------------------------------------------

names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
data = pd.read_csv('Iris.csv', names=names)

total = []
seed = []
top2 = list()

for i in range(10):
    seed.append(rd.randint(0,100))

array = data.values
ica = FastICA(n_components=4)
X_train_ica = ica.fit_transform(array[:,0:4])

ica_sum = [np.sum(X_train_ica[:,i]**2)for i in range(4)]
            
index = np.argsort(ica_sum)[::-1]

for i in X_train_ica:
    top2.append([i[index[0]],i[index[1]]]) 
    
X = np.array(top2)

for i in range(10):  
    array = data.values
    Y = array[:,4]
    X_train, X_validation, Y_train, Y_validation = train_test_split(X, Y, test_size=0.3,random_state=seed[i])
    
    knn = KNeighborsClassifier(n_neighbors = 3)
    knn.fit(X_train, Y_train)
    predictions = knn.predict(X_validation)
    acc1 = accuracy_score(Y_validation, predictions)
    total.append(acc1)

print("varinace: ",np.var(total))
print("accuracy: ",np.mean(total))

#-------------------------------------------------K-means-----------------------------------------------

dataset = data.iloc[:,0:4].values
kmeans = cluster.KMeans(n_clusters = 3).fit_predict(dataset)

#plt.scatter(dataset[kmeans == 0, 0], dataset[kmeans == 0, 1], s = 100, c = 'red', label = 'Iris-setosa')
#plt.scatter(dataset[kmeans == 1, 0], dataset[kmeans == 1, 1], s = 100, c = 'blue', label = 'Iris-versicolour')
#plt.scatter(dataset[kmeans == 2, 0], dataset[kmeans == 2, 1], s = 100, c = 'green', label = 'Iris-virginica')

#for i in range(150):
#    if i < 50:
#        if kmeans[i] != 1:
#            print(i)
#    if i >= 50 and i <100:
#        if kmeans[i] != 0:
#            print(i)       
#    if i >= 100:
#        if kmeans[i] != 2:
#            print(i)

#-----------------------------------------------GMM------------------------------------------------------
gmm = GaussianMixture(n_components = 3,max_iter = 3000)

X_gmm = list()
for i in array[:,2]:
    X_gmm.append([i])

gmm.means_init = np.array([[1],[4],[6]])
gmm.covariances_init = np.array([[1],[1],[1]])
gmm.weights_init = np.array([0.5,0.25,0.25])

gmm.fit(X_gmm)
gmm_result = gmm.predict(X_gmm)

print("mean: ",gmm.means_)
print("covarinace: ",gmm.covariances_)
print("weight: ",gmm.weights_)

class0 = 0
class1 = 0
class2 = 0

for i in gmm_result:
    if i == 1:
        class1 += 1
    if i == 0:
        class0 += 1
    if i == 2:
        class2 += 1

print("class0: ",class0)
print("class1: ",class1)
print("class2: ",class2)

#----------------------------------------------C4.5 tree

treedata = pd.read_csv('4.csv')
X = treedata.loc[:,treedata.columns != 'play']
Y = treedata['play']
X = pd.get_dummies(X)

LE = preprocessing.LabelEncoder()
Y = LE.fit_transform(Y.values)

Dtree = tree.DecisionTreeClassifier(criterion = 'entropy')
Dtree = Dtree.fit(X,Y)
out = tree.export_graphviz(Dtree, out_file = None)
graph = graphviz.Source(out)
graph.render("4_tree")
#-----------------------------------------------------5

treedata2 = pd.read_csv('5.csv')
X2 = treedata2.loc[:,treedata2.columns != 'play']
Y2 = treedata2['play']

X2['outlook'] = LE.fit_transform(X2['outlook'])
X2['windy'] = LE.fit_transform(X2['windy'])
Y2 = LE.fit_transform(Y2.values)

Dtree2 = tree.DecisionTreeClassifier(criterion = 'entropy')
Dtree2 = Dtree2.fit(X2,Y2)
out2 = tree.export_graphviz(Dtree2, out_file = None)
graph2 = graphviz.Source(out2)
graph2.render("5_tree")