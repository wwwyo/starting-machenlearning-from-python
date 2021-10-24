#!/usr/bin/env python
# coding: utf-8

# In[5]:


import scipy


# In[6]:


scipy.__version__


# In[3]:


import sklearn


# In[4]:


sklearn.__version__


# In[8]:


from sklearn.datasets import load_iris
iris_dataset = load_iris()


# In[9]:


iris_dataset


# In[10]:


iris_dataset.keys()


# In[12]:


print(iris_dataset['DESCR'])


# In[17]:


iris_dataset['target_names']

iris_dataset['feature_names']


# In[18]:


type(iris_dataset['data'])


# In[20]:


iris_dataset['data'].shape


# In[21]:


iris_dataset['data'][:5]


# In[22]:


iris_dataset['target'].shape


# In[23]:


iris_dataset['target']


# In[24]:


from sklearn.model_selection import train_test_split


# In[25]:


X_train,X_test, y_train, y_test = train_test_split(iris_dataset['data'], iris_dataset['target'],random_state=0)
# random_stateはseedを渡しているので再現が可能


# X_train.shape
# # 75%

# y_train.shape

# In[30]:


print(X_test.shape)
print(y_test.shape)


# In[31]:


import pandas as pd


# In[35]:


iris_dataframe = pd.DataFrame(X_train, columns = iris_dataset.feature_names)


# In[46]:


from pandas.plotting import scatter_matrix
import mglearn


# In[47]:


grr = scatter_matrix(iris_dataframe,c=y_train,figsize=(15,15),marker='o',hist_kwds={'bins':20},s=60,alpha=.8,cmap=mglearn.cm3)


# In[48]:


grr


# In[49]:


# k-最近傍法
# 予測データの最も近い点を採用する
# 今回は近傍点の数を1とする
from sklearn.neighbors import KNeighborsClassifier as KNN
knn = KNN(n_neighbors=1)


# In[50]:


knn.fit(X_train,y_train)


# In[55]:


import numpy as np
X_new = np.array([[5,2.9,1,0.2]])


# In[56]:


prediction = knn.predict(X_new)


# In[57]:


X_new.shape


# In[61]:


print(prediction)
print(iris_dataset['target_names'][prediction])


# In[63]:


y_pred = knn.predict(X_test)


# In[64]:


y/pred


# In[65]:


y_pred


# In[66]:


np.mean(y_pred == y_test)


# In[67]:


knn.score(X_test,y_test)


# In[ ]:




