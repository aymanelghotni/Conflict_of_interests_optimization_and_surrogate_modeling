#!/usr/bin/env python
# coding: utf-8

# <font size=6>Task 1: Creating the Surrogate Model</font>

# In[373]:


import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Input,Dense
from tensorflow.keras.layers.experimental import preprocessing
import pandas as pd
import seaborn as sns


# In[105]:


dataset=pd.read_csv("./Desktop/ds_opt/ds1.csv")


# In[106]:


dataset['Res']=dataset['Width']*dataset['Height']
dataset


# In[107]:


train_data=dataset.sample(frac=0.6,random_state=0)
test_data=dataset.drop(train_data.index)


# <font size=5>Sanity check that train and test set are unique</font>

# In[108]:


print(test_data.merge(train_data).empty)


# <font size=5>As we can see below, the time and quality are functions of Resolution</font>

# In[109]:


sns.pairplot(train_data[['Res','time(Seconds)','PPI(Normalized)']],diag_kind='kde')


# <font size=5>Getting rid of useless features, because we are interested in the Width and Height only</font>

# In[110]:


train_data.pop('PPI')
train_data.pop('size(MB)')
train_data.pop('Resolution')
train_data.pop('Res')
test_data.pop('PPI')
test_data.pop('size(MB)')
test_data.pop('Resolution')
test_data.pop('Res')


# <font size=5>Separate features from labels in both Training and Test set</font>

# In[112]:


train_features=train_data.copy()
test_features=test_data.copy()
train_labels1=train_features.pop('time(Seconds)').tolist()
train_labels2=train_features.pop('PPI(Normalized)').tolist()
test_labels1=test_features.pop('time(Seconds)').tolist()
test_labels2=test_features.pop('PPI(Normalized)').tolist()

train_labels=np.concatenate((np.array(train_labels1).reshape((len(train_features),1)),np.array(train_labels2).reshape(len(train_features),1)),axis=1)
test_labels=np.concatenate((np.array(test_labels1).reshape((len(test_features),1)),np.array(test_labels2).reshape(len(test_features),1)),axis=1)


# <font size=5>Creating the model</font>

# In[359]:


normalizer = preprocessing.Normalization()
normalizer.adapt(np.array(train_features))


# In[368]:


model=keras.Sequential([
    normalizer,
    Dense(256,activation='relu'),
    Dense(512,activation='relu'),
    Dense(256,activation='relu'),
    Dense(128,activation='relu'),
    Dense(2)
])
model.compile(optimizer='adam',loss='mse')


# In[369]:


hist_train=model.fit(train_features,train_labels,validation_split=0.2,epochs=1000)


# In[374]:


hist_acc=model.evaluate(test_features,test_labels)


# In[375]:


plt.plot(hist_train.history['val_loss'],range(len(hist_train.history['val_loss'])))


# In[ ]:




