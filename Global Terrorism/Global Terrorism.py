#!/usr/bin/env python
# coding: utf-8

# In[72]:


import math
import plotly.offline as py
import plotly.graph_objs as go
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 
import pandas as pd
import warnings
warnings.filterwarnings('ignore')


# In[73]:


terror = pd.read_csv(r"C:\Users\Ameen\ML PROJECTS\Global Terrorism\globalterrorism.csv")


# In[74]:


pd.set_option('display.max_columns', None)
terror.head()


# In[75]:


terror.columns


# In[76]:


terror.rename(columns={'iyear':'Year','imonth':'Month','iday':'Day','country_txt':'Country','provstate':'state',
                       'region_txt':'Region','attacktype1_txt':'AttackType','target1':'Target','nkill':'Killed',
                       'nwound':'Wounded','summary':'Summary','gname':'Group','targtype1_txt':'Target_type',
                       'weaptype1_txt':'Weapon_type','motive':'Motive'},inplace=True)


# In[77]:


terror.head()


# In[78]:


# Extracting important columns
terror=terror[['Year','Month','Day','Country','state','Region','city','latitude','longitude','AttackType','Killed',
               'Wounded','Target','Summary','Group','Target_type','Weapon_type','Motive']]


# In[79]:


terror.head()


# In[80]:


terror.isnull().sum()


# In[81]:


terror.info()


# In[82]:


print("Country with the most attacks :" , terror['Country'].value_counts().idxmax())
print("City with the most attacks :" , terror['city'].value_counts().index[1])
print("Region with the most attacks :" , terror['Region'].value_counts().idxmax())
print("Year with the most attacks :" , terror['Year'].value_counts().idxmax())
print("Month with the most attacks :" , terror['Month'].value_counts().idxmax())
print("Group with the most attacks :" , terror['Group'].value_counts().index[1])
print("Most Attack Type :" , terror['AttackType'].value_counts().idxmax())


# In[83]:


get_ipython().system('pip install --upgrade pip')


# In[84]:


get_ipython().system('pip install wordcloud')
from wordcloud import WordCloud
from scipy import signal
cities = terror.state.dropna(False)
plt.subplots(figsize=(10,10))
wordcloud = WordCloud(background_color = 'white',
                     width = 512,
                     height = 384).generate(' '.join(cities))
plt.axis('off')
plt.imshow(wordcloud)
plt.show()


# In[85]:


x_year = terror['Year'].unique()
y_count_years = terror['Year'].value_counts(dropna = False).sort_index()
plt.figure(figsize = (18,10))
sns.barplot(x = x_year,
           y = y_count_years,
           palette = 'rocket')
plt.xticks(rotation = 45)
plt.xlabel('Attack Year')
plt.ylabel('Number of Attacks each year')
plt.title('Attack_of_Years')
plt.show()


# In[86]:


terror['Wounded'] = terror['Wounded'].fillna(0).astype(int)
terror['Killed'] = terror['Killed'].fillna(0).astype(int)
terror['casualities'] = terror['Killed'] + terror['Wounded']


# In[87]:


terror1 = terror.sort_values(by='casualities',ascending=False)[:40]


# In[88]:


heat=terror1.pivot_table(index='Country',columns='Year',values='casualities')
heat.fillna(0,inplace=True)


# In[89]:


import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
colorscale = [[0, '#edf8fb'], [.3, '#00BFFF'],  [.6, '#8856a7'],  [1, '#810f7c']]
heatmap = go.Heatmap(z=heat.values, x=heat.columns, y=heat.index, colorscale=colorscale)
data = [heatmap]
layout = go.Layout(
    title='Top 40 Worst Terror Attacks in History from 1982 to 2016',
    xaxis = dict(ticks='', nticks=20),
    yaxis = dict(ticks='')
)
fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='heatmap',show_link=False)


# In[90]:


plt.figure(figsize = (15,6))
sns.barplot(terror['Country'].value_counts()[:15].index , terror['Country'].value_counts()[:15].values)
plt.title("Top Countries Affected")
plt.xlabel("Countries")
plt.ylabel("Count")
plt.xticks(rotation = 90)
plt.show()


# In[91]:


terror.Group.value_counts()[1:15]


# In[92]:


plt.figure(figsize = (15,6))
sns.barplot(terror.Group.value_counts()[1:15].index , terror.Group.value_counts()[1:15].values)
plt.title("Terrorist Organizations ")
plt.xticks(rotation = 90)
plt.xlabel("Organization Name")
plt.ylabel("Attacks carried by terrorist Organization")
plt.show()


# In[93]:


# Total Number of people killed in terror attack
killData = terror.loc[:,'Killed']
print('Number of people killed by terror attack:', int(sum(killData.dropna())))# drop the NaN values


# In[94]:


# Let's look at what types of attacks these deaths were made of.
attackData = terror.loc[:,'AttackType']
# attackData
typeKillData = pd.concat([attackData, killData], axis=1)


# In[95]:


typeKillData.head()


# In[96]:


typeKillFormatData = typeKillData.pivot_table(columns='AttackType', values='Killed', aggfunc='sum')
typeKillFormatData


# In[97]:


typeKillFormatData.info()


# In[98]:


#Number of Killed in Terrorist Attacks by Countries
countryData = terror.loc[:,'Country']
# countyData
countryKillData = pd.concat([countryData, killData], axis=1)


# In[99]:


countryKillFormatData = countryKillData.pivot_table(columns='Country', values='Killed', aggfunc='sum')
countryKillFormatData


# In[100]:


plt.figure(figsize = (15,6))
labels = countryKillFormatData.columns.tolist()
labels = labels[50:101]
index = np.arange(len(labels))
transpoze = countryKillFormatData.T
values = transpoze.values.tolist()
values = values[50:101]
values = [int(i[0]) for i in values]
colors = ['red', 'green', 'blue', 'purple', 'yellow', 'brown', 'black', 'gray', 'magenta', 'orange']
plt.bar(index, values, color = colors, width = 0.9)
plt.ylabel('Killed People', fontsize=20)
plt.xlabel('Countries', fontsize = 20)
plt.xticks(index, labels, fontsize=18, rotation=90)
plt.title('Number of people killed by countries', fontsize = 20)
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:




