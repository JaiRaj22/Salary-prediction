#!/usr/bin/env python
# coding: utf-8

# # Simple Linear Regression

# ## Importing the libraries

# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle


# ## Importing the dataset

# In[3]:


data = pd.read_csv('Salary_Data.csv')
data


# In[6]:


x = data.iloc[:,:-1].values
y = data.iloc[:,-1].values


# In[7]:


x


# In[8]:


y


# ## Splitting the dataset into the Training set and Test set

# In[9]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)


# ## Training the Simple Linear Regression model on the Training set

# In[10]:


from sklearn.linear_model import LinearRegression
lr = LinearRegression()
reg = lr.fit(x_train, y_train)


# ## Predicting the Test set results

# In[11]:


yped = lr.predict(x_test)


# In[12]:


yped


# ## Visualising the Training set results

# In[13]:


plt.scatter(x_train, y_train, color='red')
plt.plot(x_train, lr.predict(x_train), color='blue')
plt.xlabel('salary')
plt.ylabel('years')
plt.show()


# ## Visualising the Test set results

# In[14]:


plt.scatter(x_test, y_test, color='red')
plt.plot(x_train, lr.predict(x_train), color='blue')
plt.xlabel('salary')
plt.ylabel('years')
plt.show()


# In[15]:


pickle.dump(reg, open('salary.pkl','wb'))


# In[ ]:




