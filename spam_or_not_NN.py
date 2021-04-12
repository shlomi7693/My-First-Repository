#!/usr/bin/env python
# coding: utf-8

# ### import

# In[575]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression


# In[576]:


data = pd.read_csv('spambase_csv.csv')
data.head()


# In[546]:





# ### shuffeling the dataset:

# In[577]:


rand_permutations = np.random.permutation(data.shape[0])
shuffeled_data = data.loc[rand_permutations,:]
shuffeled_data.head()


# ### balancing the dataset:

# In[578]:


num_of_spams = np.sum(data['class'])
num_of_zeroes = data.shape[0]-num_of_spams
print('number of spam emails: '+ str(num_of_spams) + ' . number_of_regular_emails: ' +str(num_of_zeroes))
num_examples_to_remove  = num_of_zeroes-num_of_spams


# we need to remove 975 records of class=0 

# In[579]:


index_to_remove = shuffeled_data[shuffeled_data['class'] == 0].index.values[0:975]
balanced_data = shuffeled_data.drop(index_to_remove[0:num_examples_to_remove],axis=0)


# now the data is equally balanced:

# In[580]:


num_of_spams = np.sum(balanced_data['class'])
num_of_zeroes = balanced_data.shape[0]-num_of_spams
print('number of spam emails: '+ str(num_of_spams) + ' . number_of_regular_emails: ' +str(num_of_zeroes))


# ### checkpoint:

# In[581]:


data = balanced_data.copy()


# ### splitting the dataset into train, val and test:

# In[582]:


samples_count = data.shape[0]
train_samples_count = int(0.85*samples_count)
val_samples_count = int(0.15*train_samples_count)
test_samples_count = samples_count-train_samples_count


# In[583]:


val = data.iloc[0:val_samples_count,:]
train = data.iloc[val_samples_count:train_samples_count,:]
test = data.iloc[train_samples_count:,:]


# ### creating inputs and targets:

# In[584]:


## train:
train_inputs = train.drop(['class'],axis=1)
train_targets = train['class']

## val:
val_inputs = val.drop(['class'],axis=1)
val_targets = val['class']

## test:
test_inputs = test.drop(['class'],axis=1)
test_targets = test['class']


# ### standardizing the inputs by the data of train only!

# In[585]:


## creating scaler from the train dataset only:
scaler = StandardScaler()

scaler.fit(train_inputs)
scaled_train_inputs = scaler.transform(train_inputs)

## using the scaler to standardize the val inputs and test inputs:
scaled_val_inputs = scaler.transform(val_inputs)
scaled_test_inputs = scaler.transform(test_inputs)


# ### modeling:

# setting parameters:

# In[680]:


input_size = train_inputs.shape[1]
output_size = 2 ## sincewe have only 2 classes:
hidden_layer_size = 50
batch_size =20
early_stopping = tf.keras.callbacks.EarlyStopping(patience=2)
num_of_epochs = 100


# model:

# In[681]:


model = tf.keras.Sequential([tf.keras.layers.Dense(hidden_layer_size,activation='relu'),
                            tf.keras.layers.Dense(hidden_layer_size,activation='relu'),
                            tf.keras.layers.Dense(output_size,activation='softmax'),])


# In[682]:


model.compile(optimizer='adam' , loss = 'sparse_categorical_crossentropy', metrics=['accuracy'])


# In[683]:


model.fit(x=scaled_train_inputs, y=train_targets,batch_size=batch_size,epochs=num_of_epochs,
         validation_data=(scaled_val_inputs,val_targets), verbose=2, callbacks=[early_stopping])


# ### test the model:

# In[679]:


predictions = model.evaluate(scaled_test_inputs,test_targets)
#accuracy_score(test_targets,predictions)


# In[612]:


reg = LogisticRegression()
reg.fit(scaled_train_inputs,train_targets)
reg.score(scaled_test_inputs,test_targets)

