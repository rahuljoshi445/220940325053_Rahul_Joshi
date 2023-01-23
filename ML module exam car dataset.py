#!/usr/bin/env python
# coding: utf-8

# In[7]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[24]:


df = pd.read_csv('D:/car.csv')
df


# In[25]:


df.shape


# In[26]:


df.head()


# In[27]:


df.info()


# In[28]:


df.describe()


# In[30]:


df.isnull().sum()


# In[31]:


sns.pairplot(df)


# In[33]:


#Indepenndent and dependent varaiable
X = df.iloc[:,:-1].values

y = df.iloc[:,-1].values


# In[35]:


X


# In[36]:


y


# In[37]:


df.head()


# In[38]:


df.describe(include='all')


# In[47]:


df.drop('Car_Name', axis=1 ,inplace=True)


# In[52]:


maximum =df['Year'].max()
Age=df['Year'].apply(lambda x:(maximun+1)-X)
df.drop('Year',axis=1,inplace=True)
df.insert(0,'Age');df


# In[55]:


fig=plt.figure(figuresize=(10,10))
sns.replot('Year','Selling_Price',data=df,kind='line')


# In[56]:


df['Selling_Price'].describe()


# In[57]:


#All the figures are in lacs so minimum selling price is 10000 and maximum selling price is 3500000
sns.catplot('Selling_Price',data=df)


# In[58]:


#so we are plotting relationship betweeen selling price and present price
df.plot.scatter(x='Selling_Price',y='Present_Price')


# In[59]:


#As we can see tht for every yerar petrol cars and bike are more for selling than diesel and CNG
plt.figure(figsize=(7,7))
sns.countplot('Year',hue='Fuel_Type',data=df)


# In[60]:


plt.figure(figsize=(4,4))
sns.catplot(data=df,kind='swarm',x='Transmission',y='Selling_Price',split=True)


# In[61]:


df.plot.scatter('Selling_Price','Kms_Driven')


# In[62]:


sns.jointplot(data=df, x="Year", y="Fuel_Type")


# In[64]:


#we are converting Year column into number of years selling car is old
df['curr_Year']=2020
df['Years']=df.curr_Year - df.Year
df.drop(['Year','curr_Year'],axis=1,inplace=True)
df.head()


# In[66]:


categorical_columns=df.select_dtypes(include='object')
for i in categorical_columns:
    print('column name {} -> {} : {}'.format(i,df[i].nunique(),df[i].unique()))


# In[67]:


df=pd.get_dummies(df,drop_first=True)
df.head()


# In[68]:


sns.pairplot(data=df)


# In[69]:


#after which plotting corelatio between dataset column by using heatmap
plt.figure(figsize=(10,10))


# In[70]:


X=df.drop('Selling_Price',axis=1)
y=df['Selling_Price']


# In[73]:


print(X.shape)
print(y.shape)


# In[74]:


from sklearn.ensemble import ExtraTreesRegressor
model=ExtraTreesRegressor()
model.fit(X,y)
feat_imp=pd.Series(model.feature_importances_,index=X.columns).sort_values(ascending=True)
feat_imp.plot(kind='barh')
plt.show()


# In[75]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)


# In[76]:


from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
X_train=scaler.fit_transform(X_train)
X_test=scaler.transform(X_test)


# In[ ]:




