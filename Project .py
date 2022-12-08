#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import plotly.express as px
import seaborn as sns
from sklearn import linear_model
import matplotlib.pyplot as plt


# In[2]:


df = pd.DataFrame()
df = pd.read_csv("/Users/phonemyatkyaw/Downloads/world_population2022.csv")


# In[3]:


df.head()


# In[4]:


df.info()


# In[5]:


df.columns


# In[6]:


df.describe()


# In[7]:


df.isna().sum()


# In[8]:


df.rename(columns={"Country/Other":"Country", "Population (2020)":"Population"}, inplace=True)
df.rename(columns={"Urban Pop %": "Urban Pop"},inplace=True)


# In[9]:


avg_urban = df["Urban Pop"].mean()
avg_urban


# In[10]:


df["Urban Pop"] = df["Urban Pop"].fillna(avg_urban)


# In[11]:


df.isna().sum()


# In[12]:


df.head()


# In[13]:


highestpop = df.groupby("Country")["Population"].sum()


# In[14]:


highestpop= highestpop.sort_values(ascending=False)
highestpop


# In[15]:


highestpop = highestpop.to_frame()


# In[16]:


highestpop = highestpop.iloc[:5]
highestpop


# In[17]:


l = highestpop.index
e = [0.2,0,0,0,0]
c= ["#16F3AC","#F71C47","#D9F71C","#1E3DE8","#0BEAF5"]
plt.figure(figsize=(10,16))
plt.pie(highestpop["Population"],explode=e, colors= c, labels=l,autopct="%.2f%%")
plt.legend(title = "Top 5 Countries")
plt.title("Top 5 countries with highest population")
plt.savefig('highpop.png')


# In[18]:


lowpop = df.groupby("Country")["Population"].sum()
lowpop


# In[19]:


lowpop = lowpop.sort_values(ascending=True)
lowpop


# In[20]:


lowpop = lowpop.to_frame()


# In[21]:


lowpop = lowpop.iloc[:5]
lowpop


# In[22]:


l = lowpop.index
e = [0.2,0.2,0,0,0]
c= ["#16F3AC","#F71C47","#D9F71C","#1E3DE8","#0BEAF5"]
plt.figure(figsize=(10,16))
plt.pie(lowpop["Population"],colors=c, explode=e, labels=l,autopct="%.2f%%")
plt.legend(title = "Top 5 Countries with lowest population")
plt.title("Top 5 coutries with lowest population")
plt.savefig('lowpop.png')


# In[23]:


fig = px.choropleth(df, locations='Country',color='Yearly Change',locationmode='country names',color_continuous_scale='Electric',template='plotly_dark',title='Yearly Change')
fig.show()
plt.savefig('Yealy Change Map.png')


# In[24]:


fig = px.choropleth(df, locations='Country',color='Fert. Rate',locationmode='country names',color_continuous_scale='Electric',template='plotly_dark',title='World Map Based On Fert Rate')
fig.show()


# In[25]:


fig = px.choropleth(df, locations='Country',color='Migrants (net)',locationmode='country names',color_continuous_scale='Electric',template='plotly_dark',title='World Map Based on Migrants rate(net)')
fig.show()


# In[26]:


df.corr()


# In[27]:


plt.figure(figsize=(12,4))
sns.heatmap(df.corr(),annot=True, cmap="plasma", vmin=-1, vmax=1,linewidths=4)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[28]:


subset = df[['Population', 'Yearly Change','Land Area (Km²)', 'Migrants (net)', 'Fert. Rate', 'World Share']]
mask = np.zeros_like(subset.corr())
triangle_indicies = np.triu_indices_from(mask)
mask[triangle_indicies] = True
plt.figure(figsize=(16,10))
sns.heatmap(subset.corr(), mask=mask, annot=True, annot_kws={"size": 10})
plt.style.use('dark_background')
plt.savefig('corr.png')


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[29]:


plt.figure(figsize=(10,5))
plt.subplot(3,1,1)
plt.title("Positive correlation between population & Net Change ")
sns.regplot(x="Net Change", y= "Population",data=df)
plt.savefig('pcorr.png')


# In[30]:


plt.figure(figsize=(10,5))
plt.subplot(3,1,1)
plt.title("Positive correlation between population & Land Area (Km²) ")
sns.regplot(x="Land Area (Km²)", y= "Population",data=df)
plt.savefig('pcorrl.png')


# In[31]:


sns.pairplot(df, vars=['Population', 'Land Area (Km²)', 'Urban Pop', 'World Share'], 
             height=2.5)
plt.savefig('variable.png')


# In[32]:


df.columns


# In[33]:


df["Country"].count


# In[34]:


highmigrant = df.groupby("Country")["Migrants (net)"].sum()
highmigrant 


# In[35]:


highmigrant = highmigrant.sort_values(ascending=True)
highmigrant


# In[36]:


highmigrant.to_frame()


# In[37]:


highmigrant = highmigrant.iloc[:5]
highmigrant


# In[38]:


l = highmigrant.index
data = highmigrant.values
plt.figure(figsize=(10,8))
plt.title("Top 5 Countries with highest migrant", color="Red", fontsize=22)
plt.ylabel("Country", color="Red", fontsize=22)
plt.xlabel("Migrants (net)", color="Red", fontsize=22)
plt.barh(l, width=data,);
plt.savefig('h5.png')


# In[39]:


lowmigrant = df.groupby("Country")["Migrants (net)"].sum()
lowmigrant


# In[40]:


lowmigrant = lowmigrant.sort_values(ascending=True)
lowmigrant 


# In[41]:


lowmigrant.to_frame()
lowmigrant = lowmigrant[["Canada","United Kingdom","Turkey","Germany","United States"]]
lowmigrant


# In[42]:


l = lowmigrant.index
data = lowmigrant.values
plt.figure(figsize=(10,8))
plt.title("Top 5 Countries with lowest migrant", color="Red", fontsize=22)
plt.ylabel("Country", color="Red", fontsize=22)
plt.xlabel("Migrants (net)", color="Red", fontsize=22)
plt.barh(l, width=data)
plt.savefig('l5.png');


# In[43]:


df.columns


# In[44]:


df["Yearly Change"]


# In[45]:


plt.ylabel("Fert.Rate")
plt.xlabel("Med Age")
plt.scatter(df["Med. Age"],df["Fert. Rate"], color="g", marker="x")
plt.savefig('scatter for Fert and Med Age.png')


# In[46]:


X = df[["Med. Age"]]
y = df["Fert. Rate"]
model = linear_model.LinearRegression()
model.fit(X,y)
m = c.coef_[0]
model.predict([[30]])
b = model.intercept_
a= m* 30 +b
a


# In[ ]:


df.columns


# In[ ]:


plt.ylabel("Fert. Rate")
plt.xlabel("Med. Age")
plt.scatter(df["Med. Age"], df["Fert. Rate"], color="g", marker="x")
plt.plot(df["Med. Age"], model.predict(X))
plt.scatter (30, a,color = 'r', marker = "+", s=200)
plt.savefig("predict.png");


# In[ ]:


fig = px.choropleth(df, locations='Country',color='Migrants (net)',locationmode='country names',color_continuous_scale='Electric',template='plotly_dark',title='World Map Based on Migrants rate(net)')
fig.show()


# In[ ]:


fig = px.choropleth(df, locations='Country',color='Fert. Rate',locationmode='country names',color_continuous_scale='Electric',template='plotly_dark',title='World Map Based On Fert Rate')
fig.show()


# In[ ]:





# In[ ]:


fig = px.choropleth(df, locations='Country',color='Density (P/Km²)',locationmode='country names',color_continuous_scale='Electric',template='plotly_dark',title='Yearly Change')
fig.show()

