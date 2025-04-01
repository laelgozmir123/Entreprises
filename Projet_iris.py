import numpy as np # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # data visualization
import seaborn as sns # data visualization
import sklearn # machine learning
# import iris dataset with sklearn
from sklearn.datasets import load_iris
iris = load_iris()
print(iris.DESCR)
df_iris = pd.DataFrame(iris.data, columns=iris.feature_names)
df_iris['target'] = iris.target
# Description of Iris dataset (mean, std, min, max, quartiles)
print(df_iris.describe())
# Names of the target values
print(iris.target_names)

# Count of each target value
print(df_iris['target'].value_counts())

# Same as above but in percentage
print(df_iris['target'].value_counts(normalize=True))
# Correlation matrix between all the features
df_iris.corr()
# Heatmap of the correlation matrix using seaborn https://seaborn.pydata.org/generated/seaborn.heatmap.html 
plt.title("Heatmap of the correlation matrix between all the features")
sns.heatmap(df_iris.drop('target',axis=1).corr(), annot=True, cmap='coolwarm') # annot = True to print the values inside the square
plt.show()
def rapport_corr(cible,data): #fonction qui calcule le rapport de corrélation
    #cible : 1 variable qualitative
    #data : un DataFrame qui contient des variables quantitatives
    #moyenne par variable
    m=data.mean()
    #SCT : variabilité totale = nbre d'ind*variance
    SCT=data.shape[0]*data.var(ddof=0)
    #DataFrame conditionnellement aux groupes
    Xb=data.groupby(cible)
    #effectifs conditionnels
    nk=Xb.size()
    #moyennes conditionnelles dans chaque groupe
    mk=Xb.mean()
    #pour chaque groupe écart à la moyenne par variable
    EMk=(mk-m)**2
    #pondéré par les effectifs du groupe
    EM=EMk.multiply(nk,axis=0)
    #somme des valeurs
    SCE=np.sum(EM,axis=0)
    #carré du rapport de corrélation
    R2=SCE/SCT
    R2trie=R2.sort_values(ascending=False)
    print(R2trie)
    #print(R2trie.index)
    plt.bar(range(1,R2trie.shape[0]+1),height=R2trie)
    return np.sqrt(R2)
rapport_corr(df_iris['target'],df_iris.drop('target',axis=1))
# Plot the pairwise relations in the iris dataset
sns.pairplot(df_iris, hue='target', height=2.5)
plt.legend(iris.target_names)
plt.show()
# X is the feature set and y is the target variable
X = df_iris.drop(['target'], axis=1)

y = df_iris['target']
# Use sklearn.model_selection 
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 42)
# Check the shapes of each subset
print(X_train.shape, X_test.shape)
print(y_train.shape, y_test.shape)
from sklearn.tree import DecisionTreeClassifier

# instantiate the DecisionTreeClassifier model with criterion gini index
clf_gini = DecisionTreeClassifier(criterion='gini', max_depth=5, random_state=0)

# fit the model aka train the model on X_train
clf_gini.fit(X_train, y_train)
y_pred_gini = clf_gini.predict(X_test)
from sklearn.metrics import accuracy_score

print('Model accuracy score with criterion gini index: {0:0.4f}'. format(accuracy_score(y_test, y_pred_gini)))
#compare training and test accuracies 
print('Training set score: {:.4f}'.format(clf_gini.score(X_train, y_train)))
print('Training set score: {:.4f}'.format(accuracy_score(clf_gini.predict(X_train),y_train)))


print('Test set score: {:.4f}'.format(clf_gini.score(X_test, y_test)))
acc_train, acc_test = [], []
for i in range(1,20):
  clf_gini_i = DecisionTreeClassifier(criterion='gini', max_depth=i, random_state=0)
  clf_gini_i.fit(X_train, y_train)
  acc_train.append(clf_gini_i.score(X_train, y_train))
  
  acc_test.append(clf_gini_i.score(X_test, y_test))
pd.DataFrame(data=[acc_train,acc_test], index=['train acc','test acc'])
plt.figure(figsize=(12,8))

from sklearn import tree

tree.plot_tree(clf_gini.fit(X_train, y_train)) 
import graphviz 
print(X_train.columns)
dot_data = tree.export_graphviz(clf_gini, out_file=None,
                                feature_names=iris.feature_names,  
                                class_names=iris.target_names,  
                                filled=True, rounded=True,
                                special_characters=True)
graph = graphviz.Source(dot_data)
graph
# instantiate the DecisionTreeClassifier model with criterion entropy
clf_en = DecisionTreeClassifier(criterion='entropy', max_depth=5, random_state=0)

# fit the model
clf_en.fit(X_train, y_train)
y_pred_en = clf_en.predict(X_test)
from sklearn.metrics import accuracy_score

print('Model accuracy score with criterion entropy: {0:0.4f}'. format(accuracy_score(y_test, y_pred_en)))
#compare training and test accuracies 
print('Training set score: {:.4f}'.format(clf_en.score(X_train, y_train)))

print('Test set score: {:.4f}'.format(clf_en.score(X_test, y_test)))
acc_train, acc_test = [], []
for i in range(1,20):
  clf_en_i = DecisionTreeClassifier(criterion='entropy', max_depth=i, random_state=0)
  clf_en_i.fit(X_train, y_train)
  acc_train.append(clf_en_i.score(X_train, y_train))
  
  acc_test.append(clf_en_i.score(X_test, y_test))
pd.DataFrame(data=[acc_train,acc_test], index=['train acc','test acc'])
plt.figure(figsize=(12,8))

from sklearn import tree

tree.plot_tree(clf_en.fit(X_train, y_train)) 
import graphviz 
dot_data = tree.export_graphviz(clf_en, out_file=None, 
                              feature_names=iris.feature_names,  
                              class_names=iris.target_names,
                              filled=True, rounded=True,  
                              special_characters=True)

graph = graphviz.Source(dot_data) 

graph
# Print the Confusion Matrix for GINI and identify the four pieces
# Use sklearn.metrics

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Confusion Matrix for GINI
print("Confusion Matrice for GINI:")
cm = confusion_matrix(y_test, y_pred_gini)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=iris.target_names)
disp.plot()
plt.show()
# Print the Confusion Matrix for Entropy and identify the four pieces

print("Confusion Matrix for Entropy:")
cm = confusion_matrix(y_test, y_pred_en)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=iris.target_names)
disp.plot()
plt.show()
from sklearn.model_selection import cross_val_score

# Cross validation score for GINI with 10 different splits
scores_gini = cross_val_score(clf_gini, X, y, cv=10)
print("Cross-validation scores for GINI: {}".format(scores_gini))
print("Mean cross-validation score: {:.2f}".format(scores_gini.mean()))
print("Standard deviation of cross-validation score: {:.2f}".format(scores_gini.std()))

# Cross validation score for Entropy with 10 different splits
scores_en = cross_val_score(clf_en, X, y, cv=10)
print("\n\nCross-validation scores for Entropy: {}".format(scores_en))
print("Mean cross-validation score: {:.2f}".format(scores_en.mean()))
print("Standard deviation of cross-validation score: {:.2f}".format(scores_en.std()))

from sklearn.metrics import classification_report

# Classification report for GINI
print(classification_report(y_test, y_pred_gini, target_names=iris.target_names))
# Classification report for Entropy
print(classification_report(y_test, y_pred_en, target_names=iris.target_names))
# Print decision boundary for decision tree with gini index
from sklearn.inspection import DecisionBoundaryDisplay
#import sklearn.inspection

# Decision boundary for sepal length and sepal width only
X = iris.data[:, :2]
sepal_classifier = DecisionTreeClassifier(criterion='gini', max_depth=3, random_state=0)
sepal_classifier.fit(X, iris.target)

disp = sklearn.inspection.DecisionBoundaryDisplay.from_estimator(sepal_classifier, X, xlabel=iris.feature_names[0],
                                              ylabel=iris.feature_names[1], alpha=0.5)
disp.ax_.scatter(X[:, 0], X[:, 1], c=iris.target, edgecolor="k")
plt.title("Sepal decision boundary for Decision Tree with Gini")
plt.show()
# Decision boundary for petal length and petal width only
X = iris.data[:, 2:]
petal_classifier = DecisionTreeClassifier(
    criterion='gini', max_depth=3, random_state=0)
petal_classifier.fit(X, iris.target)

disp = DecisionBoundaryDisplay.from_estimator(petal_classifier, X, xlabel=iris.feature_names[2],
                                              ylabel=iris.feature_names[3], alpha=0.5)
disp.ax_.scatter(X[:, 0], X[:, 1], c=iris.target, edgecolor="k")
plt.title("Petal decision boundary for Decision Tree with Gini")
plt.show()
from sklearn.decomposition import PCA
pca = PCA(n_components=2) # 2 principal components for 2D plot
pca.fit(X_train)
X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)

pca_classifier = DecisionTreeClassifier(criterion='gini', max_depth=3, random_state=0)
pca_classifier.fit(X_train_pca, y_train)

disp = DecisionBoundaryDisplay.from_estimator(pca_classifier, X_train_pca, xlabel='PC1',
                                              ylabel='PC2', alpha=0.5)
disp.ax_.scatter(X_train_pca[:, 0], X_train_pca[:,
                 1], c=y_train, edgecolor="k")
plt.title("Decision boundary for Decision Tree with Gini (PCA)")
plt.show()
