#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[8]:


df = pd.read_excel('D:/data_final.xlsx')
df


# In[9]:


df.shape


# In[10]:


df.describe


# In[11]:


df.head()


# In[12]:


df.info()


# In[13]:


df.isnull().sum()


# In[14]:


# Preprocessing

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df = df.apply(le.fit_transform)
df


# In[16]:


#Independent and Dependent variables

X = df.iloc[:,:-1].values

y = df.iloc[:,-1].values


# In[18]:


X


# In[19]:


y


# In[20]:


#Train and Test Data

from sklearn.model_selection import train_test_split
X_train , X_test , y_train , y_test = train_test_split(X,y,train_size=0.3,random_state=0)


# In[21]:


X_train.shape , X_test.shape , y_train.shape , y_test.shape


# In[22]:


#Model Building

from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion='entropy' , random_state=0)
classifier.fit(X_train,y_train)


# In[23]:


y_pred = classifier.predict(X_test)
y_pred


# In[24]:


from sklearn.metrics import confusion_matrix , accuracy_score , classification_report

#Confusion Matrix
cm = confusion_matrix(y_test , y_pred)
cm


# In[25]:


print(classification_report(y_test , y_pred))


# In[26]:


classifier.predict([[0,0,1,1]])


# In[27]:


classifier.predict([[1,1,0,0]])


# In[28]:


import seaborn as sns
plt.figure(figsize=(40, 4))
sns.countplot(x = 'observation', data = df)


# In[29]:


df.describe()


# In[30]:


def null_values(df):
    return round((df.isnull().sum()*100/len(df)).sort_values(ascending = False),2)


# In[31]:


df[] = df[].fillna(value = df.median())


# In[32]:


df[]=df[].replace(np.nan,df.median())


# In[33]:


df1['Size'] = df1['Size'].apply(lambda x: x.replace('nan',df.median())


# In[35]:


#Scatter plot 
plt.scatter(X,y)


# In[36]:


plt.xlabel('X axis'


# In[37]:


sns.countplot(df['x'])


# In[38]:


label=LabelEncoder()
label.fit(x[:,0])
x[:,0]=label.transform(x[:,0])


# In[40]:


onehotEncoder=OneHotEncoder()
encode=pd.DataFrame(onehot.fit_transform(df1[['Country']]).toarray())
encode


# In[41]:


from sklearn.linear_model import LinearRegression
model=LinearRegression()
model.fit(x_train,y_train)


# In[42]:


from sklearn.linear_model import LinearRegression
model=LinearRegression()
model.fit(x_train,y_train)


# In[43]:


y_pred=model.predict(x_test)


# In[ ]:




