# -*- coding: utf-8 -*-
import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("/Users/Vamshi/Desktop/GMU/AIT-590/Experimental project/class project/training_data.csv")
df = pd.DataFrame(df, columns = ['ID (this is not a feature)','age','workclass','fnlwgt','education','education-num','marital-status','occupation','relationship','race','sex','capital-gain','capital-loss','hours-per-week','native-country','salary'])
df=df.replace(" ?",None)
onehotdf=df.copy()

#######below code is for label encoder
from sklearn import preprocessing
collection = ['workclass','education','marital-status','occupation','relationship','race','sex','native-country','salary']
for x in collection:
    le = preprocessing.LabelEncoder()
    le.fit(df[x])
    list(le.classes_)
    df[x]=le.transform(df[x]) 

x= df.drop(['salary','ID (this is not a feature)'], axis = 1)#16-2=14 features
y=df["salary"]#target

from sklearn import model_selection
x_train, x_test , y_train, y_test = model_selection.train_test_split(x, y, test_size=0.2, random_state=1)
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor()
rf.fit(x_train, y_train)
##Features sorted by their score
important= (sorted(zip(map(lambda x: round(x, 4), rf.feature_importances_), x), reverse=True))
count=0
finalfeature=[]
for x in range(1, 15):
    if(important[x][0]>0.01):
        count=count+1
        finalfeature.append(important[x][1])
    else:
        break    
features=finalfeature
features.append('salary')
#######below code is for onehot encoder
df=onehotdf[features]
cat_cols = df.select_dtypes(exclude = 'number')
num_cols = df.select_dtypes(include = 'number')
onehot_cat_cols = pd.get_dummies(cat_cols)
onehot_cat_cols.head()
df_final = pd.concat([num_cols,onehot_cat_cols],sort=True,axis=1)
y = df_final['salary_ <=50K']
x = df_final.drop(['salary_ <=50K','salary_ >50K'], axis = 1)

# Standardizing the features
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
x = StandardScaler().fit_transform(x)
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.2,random_state=0)
accuracies = {}

##naive baysian classifier
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import BernoulliNB
model = BernoulliNB()
fModel = model.fit(x_train, y_train)
predictions = fModel.predict(x_test)
accuracies['naive baysian classifier'] = accuracy_score(y_test, predictions)
print(accuracy_score(y_test, predictions))

##linear Regression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn import metrics
lm = LinearRegression()
model = lm.fit(x_train,y_train)
predicted=model.predict(x_test)
predicted[predicted> 0.5]=1
predicted[predicted<= 0.5]=0
accuracies['linear Regression']=metrics.accuracy_score(y_test,predicted)
print('accuracy:',metrics.accuracy_score(y_test,predicted))
print('mse:',mean_squared_error(y_test,predicted))
print(pd.DataFrame({'Predicted':predicted,'Actual': y_test}))

##Ridge 
from sklearn.linear_model import RidgeCV
ridgeReg = RidgeCV(alphas=(0.01,0.1,0.3,0.5,1,5), cv=10).fit(x_train,y_train)
pred_test = ridgeReg.predict(x_test)
pred_test[pred_test> 0.5]=1
pred_test[pred_test<= 0.5]=0
mse = mean_squared_error(y_test,pred_test)
accuracies['Ridge']=metrics.accuracy_score(y_test,pred_test)
print('accuracy:',metrics.accuracy_score(y_test,pred_test))
print('mean squared error:',mse)
print(pd.DataFrame({'Predicted':pred_test,'Actual': y_test}))

##lasso 
from sklearn.linear_model import Lasso, LassoCV
Lass = Lasso(max_iter = 10000, normalize = True)
Lassoc = LassoCV(alphas = None, cv = 10, max_iter = 10000, normalize = True)
Lassoc.fit(x_train, y_train)
Lass.set_params(alpha=Lassoc.alpha_)
Lass.fit(x_train, y_train)
mean_squared_error(y_test, Lass.predict(x_test))
predicted=Lass.predict(x_test)
predicted[predicted> 0.5]=1
predicted[predicted<= 0.5]=0
accuracies['Lasso']=metrics.accuracy_score(y_test,predicted)
print('accuracy:',metrics.accuracy_score(y_test,predicted))
print('mean squared error:',mean_squared_error(y_test,predicted))
print(pd.DataFrame({'Predicted':predicted,'Actual': y_test}))

##logistic regression 
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(x_train,y_train)
acc = lr.score(x_test,y_test)
accuracies['logistic regression ']=acc
print("Test Accuracy {:.2f}%".format(acc))

#Classification report
from sklearn.metrics import classification_report
print(classification_report(y_test,predictions))
from sklearn.metrics import accuracy_score
accuracies['Classification report ']=accuracy_score(y_test, predictions)
print("Accuracy:", accuracy_score(y_test, predictions))

##perceptron 
from sklearn.linear_model import Perceptron
ppn = Perceptron(eta0=0.1, random_state=0)
ppn.fit(x_train, y_train) 
y_pred = ppn.predict(x_test) 
accuracies['Perceptron']=accuracy_score(y_test, y_pred)
print('Accuracy: %.2f' % accuracy_score(y_test, y_pred))

##descision tree 
from sklearn.tree import DecisionTreeClassifier 
classifier = DecisionTreeClassifier(max_leaf_nodes=128)
classifier.fit(x_train, y_train)
y_predict_test=classifier.predict(x_test)
y_predict_train=classifier.predict(x_train)
accuracies['descision tree']=classifier.score(x_test, y_test)
print("Classifier Accuracy:", '%f'%classifier.score(x_test, y_test))

##KNN
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
scoreList = []
for i in range(1,5):
    knn2 = KNeighborsClassifier(n_neighbors = i) 
    knn2.fit(x_train, y_train)
    scoreList.append(knn2.score(x_test, y_test))
    
plt.plot(range(1,5), scoreList)
plt.xticks(np.arange(1,5,1))
plt.xlabel("K value")
plt.ylabel("Score")
plt.show()
knn = KNeighborsClassifier(n_neighbors = 3)  
knn.fit(x_train, y_train)

prediction = knn.predict(x_test)
print("{} NN Score: {:.2f}%".format(3, knn.score(x_test, y_test)*100))
acc = max(scoreList)
accuracies['KNN'] = acc
print("Maximum KNN Score is {:.2f}%".format(acc))

##Ridge classifier
from sklearn.linear_model import RidgeClassifier
from sklearn import preprocessing
from sklearn.metrics import classification_report
clf = RidgeClassifier().fit(x_train, y_train)
clf.score(x_train, y_train) 
accuracies['Ridge classifier']=clf.score(x_train, y_train)
print(classification_report(y_test,clf.predict(x_test)))

##non linear classifier/SVM
from sklearn.svm import SVC
svc = SVC(gamma=0.1)
svc.fit(x_train, y_train)
y_pred = svc.predict(x_test)

#Classification Report
from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))
score_svc = svc.score(x_test,y_test)
accuracies['SVC']=score_svc
print('Accuracy of SVC: ', score_svc)

##Exploratory data analysis
import seaborn as sns
numerical = ['age','fnlwgt','education-num','capital-gain','capital-loss','hours-per-week','salary']
newdata=df[numerical]
corr = newdata.corr()
plt.subplots(figsize=(20, 9))
sns.heatmap(corr, annot=True)
plt.show()

df['native-country'].value_counts().plot(kind='bar', color="SteelBlue")
plt.title("Bar chart showing Native country count",color="Blue")
plt.xlabel('native-country')
plt.ylabel('count')
plt.show()

##feature generation 
def plotgraph(modelname, X ):
	Df = pd.DataFrame(data = X, columns = ['component 1', 'component 2']) 
    #storing new features generated in new Dataframe df
	finalDf = pd.concat([Df, df_final['salary_ <=50K']], axis = 1)	
	fig = plt.figure(figsize = (8,8))  #size of 8*8
	ax = fig.add_subplot(1,1,1) 
	ax.set_xlabel(modelname + ' 1', fontsize = 10) #xlabel
	ax.set_ylabel(modelname +' 2', fontsize = 10)  #ylabel
	ax.set_title(modelname, fontsize = 20)         #title
	targets = [0, 1]
	colors = ['r', 'g']
	for target, color in zip(targets,colors): #new iterator with both variables combined (random)
		indicesToKeep = finalDf['salary_ <=50K'] == target
		ax.scatter(finalDf.loc[indicesToKeep,'component 1'], finalDf.loc[indicesToKeep, 'component 2'] , c = color , s = 20)
	ax.legend(targets)
	ax.grid()

df.corr()#correlation

##PCA (linear)

from sklearn.decomposition import PCA
pca = PCA(n_components=2)#,whiten=True,random_state=20
pc = pca.fit_transform(x)
plotgraph('PCA',pc)

##Gaussian Random Projection
from sklearn.random_projection import GaussianRandomProjection
GRP = GaussianRandomProjection(n_components=2, random_state=20)
GRP.fit(x)
X_grd = GRP.transform(x)
plotgraph('GRP',X_grd)

######Evaluating classifiers
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
kfold = model_selection.KFold(n_splits=10, random_state=7)

# Classification Accuracy
model = LogisticRegression(solver='liblinear')
scoring = 'accuracy'
results = model_selection.cross_val_score(model, x, y, cv=kfold, scoring=scoring)
print("Accuracy: %0.3f (%0.3f)" % (results.mean(), results.std()))

# Logarithmic Loss
scoring = 'neg_log_loss'
results = model_selection.cross_val_score(model, x, y, cv=kfold, scoring=scoring)
print("LogLoss: %0.3f (%0.3f)" % (results.mean(), results.std()))

# Area Under ROC Curve
scoring = 'roc_auc'
results = model_selection.cross_val_score(model, x, y, cv=kfold, scoring=scoring)
print("AUC: %0.3f (%0.3f)" % (results.mean(), results.std()))

# Cross Validation Classification Confusion Matrix
model = LogisticRegression(solver='liblinear')
model.fit(x_train, y_train)
predicted = model.predict(x_test)

# Cross Validation Classification Report
report = classification_report(y_test, predicted)
print("Classification Report:")
print(report)

##Featurte selection
import seaborn as sns
corrmat = df_final.drop(['salary_ >50K'],axis=1).corr()
print(corrmat)
plt.figure(figsize=(50,50))
g=sns.heatmap(corrmat,annot=True,cmap="RdYlGn", fmt='.2f')
plt.show()

X = df_final.iloc[:,0:90]  
y = df_final.iloc[:,90:91]   
from sklearn.ensemble import ExtraTreesClassifier
import matplotlib.pyplot as plt
model = ExtraTreesClassifier()
model.fit(X,y)
print(model.feature_importances_) 
feat_importances = pd.Series(model.feature_importances_, index=X.columns)
feat_importances.nlargest(20).plot(kind='barh')
plt.show()

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

bestfeatures = SelectKBest(score_func=chi2, k=10)
fit = bestfeatures.fit(X,y)
dfscores = pd.DataFrame(fit.scores_)
dfcolumns = pd.DataFrame(X.columns)
featureScores = pd.concat([dfcolumns,dfscores],axis=1)
featureScores.columns = ['Specs','Score']  #naming the dataframe columns
print(featureScores.nlargest(10,'Score'))