
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
df = pd.read_csv("50_Startups.csv")# Vu que le code et le fichier csv sont au même endroit
print(df.head())
print(df.shape)
print(df.describe())
print(df.isnull().sum())#faire la somme des elements manquants
#df.isna().sum() renvoit la même chose 
print(df.duplicated().sum())# determine le nombre de colonnnes dupliquées
df.info()
sns.pairplot(df)#graphique de dispersion pour chaque paire de variables numériques
plt.show()
df_numeric = df.select_dtypes(include=[np.number])#On exclut les valeurs non numeriques 
print(df_numeric.corr())#Relation de correlation
#O correspond à l'absence de corrélation lineaire 
#Vaut mieux utiliser la heatmap
sns.heatmap(df_numeric.corr(),annot=True,cmap='plasma')
plt.show()
plt.rcParams['figure.figsize']=[4,4]
sns.boxplot(y=df[["Profit"]].values.flatten(), palette="husl", width=0.7)
plt.title("Outliers Variable Distribution")
plt.ylabel("Profit Range")
plt.show()
#La présence de valeurs extrêmes peut indiquer des erreurs de saisie 
sns.boxplot(x='State',y='Profit',data=df)
plt.show()
#Visualisation de la data
sns.histplot(df ['Profit'],bins=5,kde=True)
plt.show()
#Machine learning
#Préparation des données en inputs et outputs
x=df[["R&D Spend","Administration","Marketing Spend"]]
y=df['Profit']
x=x.to_numpy()
y=y.to_numpy()
y=y.reshape(-1,1)
from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.3,random_state=42)
# 30% des tests pour le test
from sklearn.linear_model import LinearRegression
model=LinearRegression()
model.fit(xtrain,ytrain)
ypred=model.predict(xtest)
data=pd.DataFrame(data={"Predicted Profit":ypred.flatten()})
print(data)
#Evaluation du modèle
from sklearn.metrics import r2_score
r2Score=r2_score(ypred,ytest)
print("R2 score du modele est",r2Score*100)
from sklearn.metrics import mean_squared_error
mse=mean_squared_error(ypred,ytest)
print("Mean Squarred Error is",mse*100)
rmse=np.sqrt(mean_squared_error(ypred,ytest))
print("Root Mean Squarred Eroor is :",rmse*100)
from sklearn.metrics import mean_absolute_error
mae=mean_absolute_error(ypred,ytest)
print("Mean Absolute Error is",mae)

