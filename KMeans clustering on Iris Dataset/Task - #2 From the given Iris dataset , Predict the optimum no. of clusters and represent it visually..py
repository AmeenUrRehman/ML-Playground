#!/usr/bin/env python
# coding: utf-8

# AMEEN UR REHMAN TASK # 2 From the given Iris dataset , Predict the optimum no. of clusters and represent it visually.

# #  Predict the optimum no. of clusters and represent it visually. Using K-Means Clustering (Unsupervised Learning technique)

# In[42]:


#Import Basic libraries
import pandas as pd
import numpy as np
import seaborn as sns
sns.set_palette('husl')
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

from sklearn import metrics
from sklearn.cluster import KMeans


# In[43]:


#Storing Iris Dataset into a variable called data
data = pd.read_csv('C:/Users/Ameen/ML PROJECTS/KMeans clustering on Iris Dataset/Dataset/Iris.csv')


# In[44]:


#Print the first five rows of the dataset
print(data.head())


# In[45]:


data.shape


# In[46]:


#Check is there any null values present in dataset
print(data.isnull().sum())


# In[47]:


data['Species'].value_counts()


# #By running the above cells we conclude that:
# 1) There are 150 Observations with 4 features each (SepalLength , SepalWidth , PetalLength  , PetalWidth).
# 2) There are no null values , so we don't have to worry about that.
# 3) There are 50 Observations of each species (setosa , versicolor , virginica).

# In[48]:


data.info()


# In[49]:


data.describe()


# # Data Visualization

# In[50]:


plm = data.drop('Id' , axis= 1)
g = sns.pairplot(plm , hue = 'Species' , markers= 'o')
plt.show()


# In[51]:


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

# In[52]:


X = data.drop(['Id' , 'Species'] , axis = 1)
y = data['Species']
print(X.head())
print(X.shape)
print(y.head())
print(y.shape)


# In[53]:


x = data.iloc[:, [1, 2, 3, 4]].values


# In[54]:


import warnings
warnings.filterwarnings("ignore")


# In[55]:


k_range = list(range(1,9))
scores = []
for i in k_range:
    kmeans = KMeans(n_clusters = i, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 42)
    kmeans.fit(x)
    scores.append(kmeans.inertia_)
plt.plot(k_range, scores)
plt.xlabel("Value of n_clusters for KMeans")
plt.ylabel("within cluster sum of squares")
plt.title('The Elbow Method')
plt.show()


# In[56]:


kmeans = KMeans(n_clusters = 3, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 42)
y_pred = kmeans.fit_predict(x)


# In[57]:


#Visualising the clusters
plt.scatter(x[y_pred == 0, 0], x[y_pred == 0, 1], s = 100, c = 'red', label = 'Iris-setosa')
plt.scatter(x[y_pred == 1, 0], x[y_pred == 1, 1], s = 100, c = 'blue', label = 'Iris-versicolour')
plt.scatter(x[y_pred == 2, 0], x[y_pred == 2, 1], s = 100, c = 'green', label = 'Iris-virginica')

#Plotting the centroids of the clusters
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:,1], s = 100, c = 'yellow', label = 'Centroids')
plt.title("Clusters of Iris dataset")
plt.legend()


# # Notes:
# 1) A good clustering has tight clusters
# 2) Measures how spread out the clusters are (lower is better)
# 3) Less cluster means more inertia which is good for our model.

# In[ ]:




