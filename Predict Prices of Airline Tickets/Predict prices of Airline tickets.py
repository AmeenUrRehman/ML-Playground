#!/usr/bin/env python
# coding: utf-8

# In[132]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
train_data = pd.read_excel(r"C:\Users\Ameen\ML PROJECTS\Predict Prices of Airline Tickets\Data_train.xlsx")
train_data.head()


# In[133]:


train_data.info()


# In[134]:


train_data.isnull().sum()


# In[135]:


train_data.shape


# In[136]:


train_data[train_data['Total_Stops'].isnull()]


# In[137]:


train_data.dropna(inplace= True)
train_data.isnull().sum()


# In[138]:


data = train_data.copy()
data.head(2)


# In[139]:


data.dtypes


# In[140]:


def change_into_datetime(col):
    data[col] = pd.to_datetime(data[col])


# In[141]:


for feature in ['Date_of_Journey' , 'Dep_Time' , 'Arrival_Time']:
    change_into_datetime(feature)


# In[142]:


data.dtypes


# In[143]:


data['Journey_day'] = data['Date_of_Journey'].dt.day
data['Journey_day']


# In[144]:


data['Journey_Month'] = data['Date_of_Journey'].dt.month
data['Journey_Month']


# In[145]:


data['Journey_Year'] = data['Date_of_Journey'].dt.year
data['Journey_Year']


# In[146]:


data.head(2)


# In[147]:


data.drop("Date_of_Journey", axis = 1 , inplace = True)
data.head(2)


# In[148]:


def extract_hour_min(df,col):
    df[col+'_hour'] = df[col].dt.hour
    df[col+'_minute'] = df[col].dt.minute
    df.drop(col,axis=1,inplace = True)
    return df.head(2)


# In[149]:


extract_hour_min(data,'Dep_Time')


# In[150]:


extract_hour_min(data,'Arrival_Time')


# In[151]:


def flight_dep_time(x):
    if (x > 4) and (x <= 8):
        return 'Early Morning'
    elif (x > 8) and (x <= 12):
        return 'Morning'
    elif(x > 12) and (x <= 16):
        return 'Noon'
    elif(x > 16) and (x <= 20):
        return 'Evening'
    elif(x > 20) and (x <= 24):
        return 'Night'
    else:
        return 'Late Night'


# In[152]:


data['Dep_Time_hour'].apply(flight_dep_time)


# In[153]:


data['Dep_Time_hour'].apply(flight_dep_time).value_counts().plot(kind = 'bar')


# In[154]:


get_ipython().system('pip install plotly')
get_ipython().system('pip install cufflinks')


# In[155]:



import plotly
import cufflinks as cf
from cufflinks.offline import go_offline
from plotly.offline import download_plotlyjs , init_notebook_mode , plot , iplot


# In[156]:


get_ipython().system('pip install chart_studio')


# In[157]:


# data['Dep_Time_hour'].apply(flight_dep_time).value_counts().iplot(kind = 'bar')


# In[158]:


def preprocessing_duration(x):
    if 'h' not in x:
        x = '0h '+ x
    elif 'm' not in x:
        x =  x +' 0m'
    return x


# In[159]:


data['Duration'] = data['Duration'].apply(preprocessing_duration)


# In[160]:


data['Duration_hours'] = data['Duration'].apply(lambda x :int(x.split(' ')[0][0:-1]))


# In[161]:


data['Duration_minutes'] = data['Duration'].apply(lambda x :int(x.split(' ')[1][0:-1]))


# In[162]:


data.head(3)


# In[163]:


data['Duration'].str.replace('h','*60').str.replace(' ','+').str.replace('m' , '*1').apply(eval)


# In[164]:


# Eval takes numbers in string format and return it in value


# In[165]:


data['Duration_total_mins'] = data['Duration'].str.replace('h','*60').str.replace(' ','+').str.replace('m' , '*1').apply(eval)


# In[166]:


data.head(3)


# In[167]:


sns.lmplot(x= "Duration_total_mins" , y = "Price" , data = data)


# In[168]:


data['Destination'].unique()


# In[169]:


data['Destination'].value_counts().plot(kind = 'pie')


# In[170]:


data['Route']


# In[171]:


plt.figure(figsize = (15,5))
sns.boxplot(y='Price' , x = 'Airline' , data=data)
plt.xticks(rotation = 'vertical')


# In[172]:


plt.figure(figsize = (15,5))
sns.violinplot(y='Price' , x = 'Airline' , data=data)
plt.xticks(rotation = 'vertical')


# In[173]:


np.round(data['Additional_Info'].value_counts()/len(data)*100  , 2 )


# In[174]:


#We can think of to drop additional info as 78 % data has no info so 


# In[175]:


data.drop(columns = ['Additional_Info' , 'Route' ,'Duration_total_mins' , 'Journey_Year' ] , axis  = 1 , inplace = True)


# In[176]:


cat_col = [col for col in data.columns if data[col].dtype == 'object']


# In[177]:


num_col = [col for col in data.columns if data[col].dtype != 'object']


# In[178]:


cat_col


# In[179]:


for category in data['Source'].unique():
    data['Source_'+ category] = data['Source'].apply(lambda x:1 if x == category else 0)


# In[180]:


data.head(3)


# In[181]:


data.drop(columns = 'Source' , axis= 1 , inplace  = True)


# In[182]:


data.head(3)


# In[183]:


airlines = data.groupby(['Airline'])['Price'].mean().sort_values().index


# In[184]:


Dict1 = {Key:Index for Index , Key in enumerate(airlines,0)}


# In[185]:


Dict1


# In[186]:


data['Airline'] = data['Airline'].map(Dict1)


# In[187]:


data.head(2)


# In[188]:


data['Destination'].replace('New Delhi' , 'Delhi' , inplace = True)


# In[189]:


data['Destination']


# In[190]:


dest = data.groupby(['Destination'])['Price'].mean().sort_values().index


# In[191]:


dest


# In[192]:


Dict2 = {Key:Index for Index, Key in enumerate(dest,0)}


# In[193]:


data['Destination'] = data['Destination'].map(Dict2)


# In[194]:


data['Destination']


# In[195]:


data.head()


# In[196]:


data['Total_Stops'].unique()


# In[197]:


stops = {'non-stop' : 0 , '2 stops':2 , '1 stop':1 , '3 stops':3 , '4 stops':4}


# In[198]:


data['Total_Stops']=data['Total_Stops'].map(stops)


# In[199]:


def plot(df , col):
    fig , (ax1 , ax2 , ax3) = plt.subplots(3,1)
    sns.distplot(df[col] , ax = ax1)
    sns.boxplot(df[col] , ax = ax2)
    sns.distplot(df[col] , ax = ax3 , kde = False)


# In[200]:


plot(data , 'Price')


# In[201]:


data['Price'] = np.where(data['Price']>=35000 , data['Price'].median() , data['Price'])


# In[202]:


plot(data , 'Price')


# In[203]:


data.drop(columns = ["Duration"] , axis = 1 , inplace = True)


# In[204]:


data.head(2)


# In[205]:


data.dtypes


# In[206]:


from sklearn.feature_selection import mutual_info_regression


# In[230]:


X = data.drop(['Price'] , axis = 1)


# In[208]:


y = data['Price']


# In[209]:


mutual_info_regression(X , y)


# In[210]:


imp = pd.DataFrame(mutual_info_regression(X, y ) , index = X.columns)


# In[211]:


imp.columns = ['Importance']


# In[212]:


imp.sort_values(by = 'Importance' , ascending = False)


# In[213]:


from sklearn.model_selection import train_test_split


# In[234]:


X_train ,X_test , y_train , y_test = train_test_split(X , y, test_size = 0.25 , random_state = 42)


# In[235]:


from sklearn.ensemble import RandomForestRegressor


# In[236]:


ml_model = RandomForestRegressor()


# In[237]:


model = ml_model.fit(X_train , y_train)


# In[241]:


y_pred = model.predict(X_test)


# In[242]:


get_ipython().system('pip install pickle')


# In[243]:


import pickle


# In[244]:


file  = open(r'C:\Users\Ameen\ML PROJECTS\Predict Prices of Airline Tickets/rf_random.pkl','wb')


# In[245]:


pickle.dump(model , file)


# In[246]:


model = open(r'C:\Users\Ameen\ML PROJECTS\Predict Prices of Airline Tickets/rf_random.pkl','rb')


# In[252]:


forest = pickle.load(model)


# In[253]:


forest.predict(X_test)


# In[256]:


def mape(y_true , y_pred):
    y_true , y_pred = np.array(y_true) , np.array(y_pred)
    return np.mean(np.abs((y_true-y_pred)/y_true))*100


# In[257]:


mape(y_test , forest.predict(X_test))


# In[277]:


def predict(ml_model):
    model = ml_model.fit(X_train,y_train)
    print('training_score: {}'.format(model.score(X_train,y_train)))
    y_prediction = model.predict(X_test)
    print("Prediction are:{}".format(y_prediction))
    print('\n')
    
    from sklearn import metrics
    r2_score = metrics.r2_score(y_test , y_prediction)
    print('r2_score: {}'.format(r2_score))
    print('MSE :' , metrics.mean_squared_error(y_test , y_prediction))
    print('MAE :' , metrics.mean_absolute_error(y_test , y_prediction))
    print('RMSE :' , np.sqrt(metrics.mean_squared_error(y_test , y_prediction)))
    print('MAPE: ' , mape(y_test , y_prediction))
    sns.distplot(y_test-y_prediction)


# In[278]:


predict(RandomForestRegressor())


# In[ ]:





# In[ ]:




