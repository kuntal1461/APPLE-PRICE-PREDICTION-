#!/usr/bin/env python
# coding: utf-8

# # Liberay import

# In[117]:


import pandas as pd 
import numpy as np 
from sklearn import linear_model
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.metrics import confusion_matrix,accuracy_score
import math


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# #  IMPORT FILE 

# In[118]:


df=pd.read_csv("AAPL.csv")


# In[119]:


df.head()


# In[120]:


df.shape


# In[121]:


df.drop("Adj Close", axis=1)


# In[122]:


df.isnull()


# In[123]:


df.isnull().sum()


# In[124]:


df.isna().any()


# In[125]:


df.info()


# In[126]:


df.describe()


# In[127]:


df["Open"].plot(figsize=(16,6))


# In[128]:


plt.scatter(df.Open,df.Close,marker='o',color='g')
plt.xlabel("Open Price")
plt.ylabel("Close Price")
plt.title("Open price  Vs Close price graph")


# In[129]:


plt.scatter(df.High,df.Close,marker="*",color='c')
plt.xlabel('High prce')
plt.ylabel("close Price")
plt.title("High price vs Close Price Graph")


# In[130]:


plt.scatter(df.Close,df.Low,color="m")
plt.xlabel("Low Price")
plt.ylabel("Close Price")
plt.title("Low price vs Close Price graph")


# #  Select the data set 

# In[ ]:





# In[131]:


x=df[["Open",'Low','High','Volume']]
y=df['Close']


# # Split your data set to train data and test data  

# In[ ]:





# In[132]:


from sklearn.model_selection import train_test_split
X_train, X_test,Y_train,Y_test=train_test_split(x,y,random_state=0)


# In[133]:


X_train.shape


# In[134]:


X_test.shape


# In[135]:


Y_train.shape


# In[136]:


Y_test.shape


# #  Apply the Model 
# Here the model is linear regression model .  we apply the linear regression model to predict the close price of the apple stock

# In[137]:


reg=linear_model.LinearRegression()


# In[138]:


reg.fit(X_train,Y_train)


# In[139]:


predicted=reg.predict(X_test)


# In[ ]:





# In[140]:


print(X_test)


# In[141]:


predict.shape


# In[142]:


dataframe=pd.DataFrame(Y_test,predicted)


# In[143]:


dataframe.head()


# In[144]:


dfr=pd.DataFrame({"Actual_price":Y_test,"Prediced_Price":predicted})


# In[145]:


dfr.head()


# 

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




