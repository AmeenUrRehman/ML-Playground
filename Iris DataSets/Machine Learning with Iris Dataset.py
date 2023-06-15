#!/usr/bin/env python
# coding: utf-8

# 

# # Iris Dataset Using Supervised Learning

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
sns.set_palette('husl')
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


# In[2]:


#Storing Iris Dataset into a variable called data
data = pd.read_csv('C:/Users/Ameen/ML PROJECTS/Iris DataSets/datasets/Iris.csv')


# In[3]:


#Print the first five rows of the dataset
print(data.head())


# In[4]:


data.shape


# In[5]:


#Check is there any null values present in dataset
print(data.isnull().sum())


# In[6]:


data['Species'].value_counts()


# #By running the above cells we conclude that:
# 1) There are 150 Observations with 4 features each (SepalLength , SepalWidth , PetalLength  , PetalWidth).
# 2) There are no null values , so we don't have to worry about that.
# 3) There are 50 Observations of each species (setosa , versicolor , virginica).

# In[7]:


data.info()


# In[8]:


data.describe()


# # Data Visualization

# In[9]:


plm = data.drop('Id' , axis= 1)
g = sns.pairplot(plm , hue = 'Species' , markers= 'o')
plt.show()


# In[10]:


plt.style.use('ggplot')
a = sns.violinplot(x = 'SepalLengthCm' , y = 'Species' , data = data , inner = 'quartile')
plt.show()
b = sns.violinplot(x = 'SepalWidthCm' , y = 'Species' , data = data , inner = 'quartile')
plt.show()
c = sns.violinplot(x = 'PetalLengthCm' , y = 'Species' , data = data , inner = 'quartile')
plt.show()
d = sns.violinplot(x = 'PetalWidthCm' , y = 'Species' , data = data , inner = 'quartile')
plt.show()


# # Modeling With Scikit - Learn

# In[11]:


X = data.drop(['Id' , 'Species'] , axis = 1)
y = data['Species']
print(X.head())
print(X.shape)
print(y.head())
print(y.shape)


# # Split the dataset into a training set and a testing set

# In[14]:


X_train , X_test , y_train , y_test = train_test_split(X,y,test_size = 0.4 , random_state = 5)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)


# In[18]:


k_range = list(range(1,26))
scores = []
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors = k)
    knn.fit(X_train , y_train)
    y_pred = knn.predict(X_test)
    scores.append(metrics.accuracy_score(y_test , y_pred))
plt.plot(k_range, scores)
plt.xlabel("Value of K for KNN")
plt.ylabel("Accuracy score")
plt.title("Finding best value of K from Plot for K-Neighbors-Classifier")
plt.show()


# In[31]:


logreg = LogisticRegression()
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)
print(metrics.accuracy_score(y_test , y_pred))


# # Choosing KNN to model Iris Species prediction with K=12

# After seeing that a value of k = 12 is a pretty good number of neighbors for this model, I used it to fit the model for the entire dataset instead of just the training set

# In[33]:


knn = KNeighborsClassifier(n_neighbors = 12)
knn.fit(X_train, y_train)

knn.predict([[6,3,4,2]])


# In[34]:


knn.predict([[3,2,1,3]])

Notes:
1) The accuracy score of the models depends on the observations in the testing set, which is determined by the seed of the pseudo-random number generator (random_state parameter).
2) As a model's complexity increases, the training accuracy (accuracy you get when you train and test the model on the same data) increases.
3) If a model is too complex or not complex enough, the testing accuracy is lower.
4) For KNN models, the value of k determines the level of complexity. A lower value of k means that the model is more complex
# In[ ]:




