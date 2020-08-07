import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('glass.csv')
print(df)


#Top 5 our data
df.head()


#Top 5 data from bottom
df.tail()


#Viewing the number of rows (214) and number of columns / features (10)
df.shape



#General information of data
df.info()



#Look at Descriptive Statistic of Data.
df.describe()



#Data is clean and can continue to the Explorary Data Analysis stage
df.isnull().sum()


#Univariate analysis Type (Target features).
sns.countplot(df['Type'], color='red')
plt.show()




#Univariate analysis of RI (Refractive Index).
f = plt.figure(figsize=(20,4))
f.add_subplot(1,2,1)
sns.distplot(df['RI'])
f.add_subplot(1,2,2)
sns.boxplot(df['RI']);
plt.show()

#Univariate analysis Na (Sodium).
f = plt.figure(figsize=(20,4))
f.add_subplot(1,2,1)
sns.distplot(df['Na'], color='green')
f.add_subplot(1,2,2)
sns.boxplot(df['Na'], color='green')
plt.show()


#Correlation between features
#df.corr().style.background_gradient().set_precision(2)

# Create a Naive Bayes object
nb = GaussianNB()
#Create variable x and y.
x = df.drop(columns=['Type'])
y = df['Type']
#Split data into training and testing data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=4)
#Training the model
nb.fit(x_train, y_train)
#Predict testing set
y_pred = nb.predict(x_test)
#Check performance of model
print(accuracy_score(y_test, y_pred))




