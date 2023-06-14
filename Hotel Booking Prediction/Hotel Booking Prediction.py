#!/usr/bin/env python
# coding: utf-8

# In[392]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[393]:


df = pd.read_csv(r"C:\Users\Ameen\ML PROJECTS\Hotel Booking Prediction/hotel_bookings.csv")


# In[394]:


df.head()


# In[395]:


df.shape


# In[396]:


df.isna().sum()


# In[397]:


def data_clean(df):
    df.fillna(0 , inplace = True)
    print(df.isnull().sum())


# In[398]:


data_clean(df)


# In[399]:


df.columns


# In[400]:


list = ['adults', 'children', 'babies']
for i in list:
    print('{} has unique values as {}'.format(i,df[i].unique()))


# Adults , Children and Babies would not be zero at same time so we have to filter it.

# In[401]:


filter = (df['adults'] == 0) & (df['children'] == 0) & (df['babies'] == 0 )
df[filter]


# In[402]:


pd.set_option('display.max_columns', 32)
filter = (df['adults'] == 0) & (df['children'] == 0) & (df['babies'] == 0 )
df[filter]


# In[403]:


data  = df[~filter]
data.head()


#  ##### Problem 1 Where do the guests come from ? 
# 

# In[404]:


country_wise_data = data[data['is_canceled'] == 0 ]['country'].value_counts().reset_index()


# In[405]:


country_wise_data


# In[406]:


country_wise_data.columns = ['country' , 'No. of Guests']


# In[407]:


country_wise_data


# In[408]:


get_ipython().system('pip install folium')


# In[409]:


import folium
from folium.plugins import HeatMap
basemap = folium.Map()
basemap


# In[410]:


get_ipython().system('pip install plotly')


# In[411]:


import plotly.express as px
map_guests = px.choropleth(country_wise_data , 
                          locations = country_wise_data['country'],
                          color = country_wise_data['No. of Guests'],
                          hover_name = country_wise_data['country'],
                          title = "Home country of guests")
map_guests.show()


# In[412]:


data2 = data[data['is_canceled'] == 0]
data2


# In[413]:


data2.columns


# In[414]:


plt.figure(figsize = (12,8))
sns.boxplot(x = 'reserved_room_type' , y = "adr" , hue = "hotel" , data = data2)
plt.title("Price of room type per person and per night")
plt.xlabel("Room type")
plt.ylabel("Price(Euro)")
plt.legend()
plt.show()


# In[415]:


data_resort = data[(data['hotel'] == 'Resort Hotel') & (data['is_canceled'] == 0 )]
data_city = data[(data['hotel'] == 'City Hotel') & (data['is_canceled'] == 0 )]
data_resort.head()


# In[416]:


data_city.head()


# In[417]:


resort_hotel = data_resort.groupby(['arrival_date_month'])['adr'].mean().reset_index()


# In[418]:


resort_hotel


# In[419]:


city_hotel = data_city.groupby(['arrival_date_month'])['adr'].mean().reset_index()
city_hotel


# In[420]:


final = resort_hotel.merge(city_hotel , on = "arrival_date_month")
final.columns  = ["month" , "Price_for_resort" , "Price_for_city"]
final


# In[421]:


get_ipython().system('pip install sorted-months-weekdays')
get_ipython().system('pip install sort-dataframeby-monthorweek')


# In[422]:


import sort_dataframeby_monthorweek as sd
def sort_data(df , colname):
    return sd.Sort_Dataframeby_Month(df,colname)


# In[423]:


final = sort_data(final , 'month')
final


# In[424]:


final.columns


# In[425]:


px.line(final , x = "month" , y = ["Price_for_resort" , "Price_for_city"] , title = "Room Price per night over the month")


# In[426]:


data_resort.head()
rush_resort = data_resort['arrival_date_month'].value_counts().reset_index()
rush_resort.columns = ['month' , 'no. of guests']
rush_resort


# In[427]:


data_city.head()


# In[428]:


rush_city = data_city['arrival_date_month'].value_counts().reset_index()
rush_city.columns = ['month' , 'no. of guests']
rush_city


# In[429]:


final_rush = rush_resort.merge(rush_city , on = 'month')


# In[430]:


final_rush


# In[431]:


final_rush.columns = ["Month" , "No. of Guests in Resort Hotel" , "No. of Guests in City Hotel"]
final_rush


# In[432]:


final_rush = sort_data(final_rush , 'Month')
final_rush


# In[433]:


final_rush.columns


# In[434]:


px.line(final_rush , x = 'Month' , y = [ 'No. of Guests in Resort Hotel',
       'No. of Guests in City Hotel'] , title = "Total No. of Guests month")


# In[435]:


data.head()


# In[436]:


data.corr()


# In[437]:


co_relation = data.corr()['is_canceled']
co_relation


# In[438]:


co_relation.abs().sort_values(ascending = False)


# In[439]:


data.groupby("is_canceled")['reservation_status'].value_counts()


# In[440]:


list_not = ['days_in_waiting_list',"arrival_date_year"]
num_features = [col for col in data.columns if data[col].dtypes != 'O' and col not in list_not]
num_features


# In[441]:


data.columns


# In[442]:


cat_not = ["arrival_date_year" , "reserved_room_type" , "assigned_room_type" , "country" , "booking_changes","days_in_waiting_list"]
cat_not


# In[443]:


cat_features = [col for col in data.columns if data[col].dtype == 'O' and col not in cat_not]
cat_features


# In[444]:


data_cat = data[cat_features]
data_cat.head()


# In[445]:


data_cat.dtypes


# In[446]:


import warnings
from warnings import filterwarnings
filterwarnings("ignore")


# In[447]:


data_cat['reservation_status_date'] = pd.to_datetime(data_cat['reservation_status_date'])


# In[448]:


data_cat['year'] = data_cat['reservation_status_date'].dt.year
data_cat['day'] = data_cat['reservation_status_date'].dt.day
data_cat['month'] = data_cat['reservation_status_date'].dt.month


# In[449]:


data_cat.head()


# In[450]:


data_cat.drop("reservation_status_date" , axis = 1 , inplace = True)
data_cat['cancellation'] = data['is_canceled']
data_cat.head()


# In[451]:


cols = data_cat.columns[0:8]
cols


# In[452]:


data_cat.groupby(["hotel"])['cancellation'].mean()


# For all the columns we will do:

# In[453]:


for col in cols:
    dict  = (data_cat.groupby([col])["cancellation"].mean().to_dict())
    data_cat[col] = data_cat[col].map(dict)


# In[454]:


data_cat.head()


# In[455]:


dataframe = pd.concat([data_cat , data[num_features]] , axis =1)


# In[456]:


dataframe.head()


# In[457]:


dataframe.drop('cancellation' , axis = 1 , inplace = True)
dataframe


# In[458]:


dataframe.head()


# In[485]:


dataframe.shape


# In[459]:


sns.distplot(dataframe['lead_time'])


# In[460]:


dataframe.shape


# In[461]:


import numpy as np
def handle_outlier(col):
    dataframe[col] = np.log1p(dataframe[col])


# In[462]:


handle_outlier('lead_time')
sns.distplot(dataframe['lead_time'])


# In[463]:


handle_outlier('adr')
sns.distplot(dataframe['adr'].dropna())


# In[464]:


dataframe.isnull().sum()


# In[465]:


dataframe.dropna(inplace = True)
dataframe.columns


# In[486]:


y = dataframe['is_canceled']
x = dataframe.drop('is_canceled' , axis = 1)


# In[487]:


from sklearn.linear_model import Lasso
from sklearn.feature_selection import SelectFromModel


# In[510]:


feature_sel_model = SelectFromModel(Lasso(alpha = 0.095 , random_state = 0))


# In[511]:


feature_sel_model.fit(x, y)


# In[512]:


feature_sel_model.get_support()


# In[515]:


cols = x.columns
cols


# In[516]:


selected_feature = ['year', 'day', 'month','lead_time','deposit_type','arrival_date_week_number','adults', 'children','previous_cancellations','booking_changes', 'company',
       'adr', 'required_car_parking_spaces', 'total_of_special_requests' ]


# In[519]:


x = x[selected_feature]


# In[524]:


from sklearn.model_selection import train_test_split
X_train , X_test , y_train , y_test = train_test_split(x , y , test_size = 0.25  , random_state = 0)


# In[527]:


from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
logreg.fit(X_train , y_train)


# In[531]:


y_pred = logreg.predict(X_test)
y_pred


# In[536]:


from sklearn.metrics import confusion_matrix
confusion_matrix(y_test , y_pred)


# In[540]:


from sklearn.metrics import accuracy_score
accuracy_score(y_test , y_pred)


# In[542]:


from sklearn.model_selection import cross_val_score


# In[546]:


score = cross_val_score(logreg , x , y , cv =10)
score.mean()


# In[550]:


from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier


# In[567]:


models = []
models.append(("LogisticRegression" , LogisticRegression()))
models.append(("Naive Bayas" , GaussianNB()))
models.append(("RandomForest" , RandomForestClassifier()))
models.append(("DecisionTree" , DecisionTreeClassifier()))
models.append(("KNN" , KNeighborsClassifier()))


# In[568]:


for name, model in models:
    print(name)
    model.fit(X_train,y_train)
    prediction  = model.predict(X_test)
    from sklearn.metrics import confusion_matrix
    print(confusion_matrix(prediction , y_test))
    print('\n')
    print(accuracy_score(prediction , y_test))
    print('\n')


# In[ ]:





# In[ ]:




