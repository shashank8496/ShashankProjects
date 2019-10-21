import pandas as pd
import matplotlib.pyplot as plt

data=[]
#datas=[]
with open('/Users/Vamshi/Desktop/GMU/AIT-590/Python Project/iris.txt',encoding='utf-16') as f:
       data = [line.strip().split("\t") for line in f]
df = pd.DataFrame(data, columns = ['id','sepal length','sepal width','petal length','petal width','target'])
x = df.drop(['id','target'], axis=1)

def plotgraph(modelname, X ):

	Df = pd.DataFrame(data = X, columns = ['component 1', 'component 2'])
	finalDf = pd.concat([Df, df['target']], axis = 1)	
	fig = plt.figure(figsize = (8,8))
	ax = fig.add_subplot(1,1,1) 
	ax.set_xlabel(modelname + ' 1', fontsize = 10)
	ax.set_ylabel(modelname +' 2', fontsize = 10)
	ax.set_title(modelname, fontsize = 20)
	targets = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
	colors = ['r', 'g', 'b']
	for target, color in zip(targets,colors):
		indicesToKeep = finalDf['target'] == target
		ax.scatter(finalDf.loc[indicesToKeep,'component 1'], finalDf.loc[indicesToKeep, 'component 2'] , c = color , s = 20)
	ax.legend(targets)
	ax.grid()


##PCA
from sklearn.decomposition import PCA
pca = PCA(n_components=2 )
pc = pca.fit_transform(x)
plotgraph('PCA',pc)

#Nonlinear kernelPCA
from sklearn.decomposition import KernelPCA
kpca = KernelPCA(n_components = 2, kernel = 'rbf')
xkpca = kpca.fit_transform(x)
plotgraph('kernel pca',xkpca)

##Isomap
from sklearn.manifold import Isomap
isomap = Isomap(n_components=2)
isomap.fit(x)
X_isomap = isomap.transform(x)
plotgraph('Isomap',X_isomap)

##SVD
from sklearn.decomposition import TruncatedSVD
SVD_ = TruncatedSVD(n_components=2)
SVD_.fit(x)
X_svd = SVD_.transform(x)
#X_svd_reconst = SVD_.inverse_transform(X_svd)
plotgraph('SVD',X_svd)