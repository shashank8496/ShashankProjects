import pandas as pd
import matplotlib.pyplot as plt

with open('/Users/Vamshi/Desktop/GMU/AIT-590/Python Project/iris.txt',encoding='utf-16') as f:
       data = [line.strip().split("\t") for line in f]
df = pd.DataFrame(data, columns = ['id','sepal length','sepal width','petal length','petal width','target'])

df.isnull().sum() ##checking for missing values

df.head() #print first 5 lines of dataset

df.dtypes
df["sepal length"]=pd.to_numeric(df["sepal length"])
df["sepal width"]=pd.to_numeric(df["sepal width"])
df["petal length"]=pd.to_numeric(df["petal length"])
df["petal width"]=pd.to_numeric(df["petal width"])

from sklearn.preprocessing import StandardScaler
features = ['sepal length', 'sepal width', 'petal length', 'petal width']
# Separating out the features
x = df.loc[:, features].values
# Standardizing the features
x = StandardScaler().fit_transform(x)

def plotgraph(modelname, X ):
	Df = pd.DataFrame(data = X, columns = ['component 1', 'component 2']) 
    #storing new features generated in new Dataframe df
	finalDf = pd.concat([Df, df['target']], axis = 1)	
	fig = plt.figure(figsize = (8,8))  #size of 8*8
	ax = fig.add_subplot(1,1,1) 
	ax.set_xlabel(modelname + ' 1', fontsize = 10) #xlabel
	ax.set_ylabel(modelname +' 2', fontsize = 10)  #ylabel
	ax.set_title(modelname, fontsize = 20)         #title
	targets = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
	colors = ['r', 'g', 'b']
	for target, color in zip(targets,colors): #new iterator with both variables combined (random)
		indicesToKeep = finalDf['target'] == target
		ax.scatter(finalDf.loc[indicesToKeep,'component 1'], finalDf.loc[indicesToKeep, 'component 2'] , c = color , s = 20)
	ax.legend(targets)
	ax.grid()

df.corr()#correlation

##PCA (linear)
#Transform higher-dimensional set of features that could be possibly correlated 
#into a lower-dimensional set of linearly uncorrelated features.
from sklearn.decomposition import PCA
pca = PCA(n_components=2)#,whiten=True,random_state=20
#n_components - Number of components to keep
#When True (False by default) the components_ vectors are multiplied by the square root of n_samples 
#and then divided by the singular values to ensure uncorrelated outputs with unit component-wise 
#variances.
pc = pca.fit_transform(x)
plotgraph('PCA',pc)


#Nonlinear kernelPCA
from sklearn.decomposition import KernelPCA
kpca = KernelPCA(n_components = 2, kernel = 'rbf')
#kernel=linear,rbf(radial basis function),sigmoid,cosine
xkpca = kpca.fit_transform(x)
plotgraph('kernel pca',xkpca)

##Isomap 
#It is a nonlinear dimensionality reduction method based on spectral theory that 
#attempts to preserve geodetic distances in the lower dimension.
from sklearn.manifold import Isomap
isomap = Isomap(n_components=2)#n_jobs = 4
isomap.fit(x)
X_isomap = isomap.transform(x)
plotgraph('Isomap',X_isomap)


##Gaussian Random Projection
#data with a very large dimension (d) are projected in a two-dimensional space (kd)
# with a random matrix.
from sklearn.random_projection import GaussianRandomProjection
GRP = GaussianRandomProjection(n_components=2, random_state=20)
#random state is like seed to the function
GRP.fit(x)
X_grd = GRP.transform(x)
plotgraph('GRP',X_grd)