#!/usr/bin/env python
# coding: utf-8

#
# # CODE TO PREDICT CAR PURCHASING DOLLAR AMOUNT USING ANNs (REGRESSION TASK)
#
#

# # PROBLEM STATEMENT

# You are working as a car salesman and you would like to develop a model
# to predict the total dollar amount that customers are willing to pay given the following attributes:
# - Customer Name
# - Customer e-mail
# - Country
# - Gender
# - Age
# - Annual Salary
# - Credit Card Debt
# - Net Worth
#
# The model should predict:
# - Car Purchase Amount

# # STEP #0: LIBRARIES IMPORT
#

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# # STEP #1: IMPORT DATASET

# In[2]:


car_df = pd.read_csv('Car_Purchasing_Data.csv', encoding='ISO-8859-1')


# In[3]:


car_df.head(10)


# In[4]:


car_df.tail(10)


# # STEP #2: VISUALIZE DATASET

# In[5]:


sns.pairplot(car_df)


# # STEP #3: CREATE TESTING AND TRAINING DATASET/DATA CLEANING

# In[6]:


X = car_df.drop(['Customer Name', 'Customer e-mail', 'Country', 'Car Purchase Amount'], axis = 1)


# In[ ]:





# In[7]:


X


# In[8]:


y = car_df['Car Purchase Amount']
y.shape
# X.shape


# In[9]:


from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)


# In[10]:


X_scaled


# In[11]:


scaler.data_max_


# In[12]:


scaler.data_min_


# In[13]:


print(X_scaled[:,0])


# In[14]:


y.shape


# In[15]:


y = y.values.reshape(-1,1)


# In[16]:


y.shape


# In[17]:


y_scaled = scaler.fit_transform(y)


# In[18]:


y_scaled


# # STEP#4: TRAINING THE MODEL

# In[ ]:


#Import model selection and divide data into Taaining and Test data


# In[41]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size = 0.25)


# In[42]:


#You can determine the training data percentage in ....test_size = 0.15....above


# In[43]:


X_test.shape


# In[44]:


X_train.shape


# In[45]:


import tensorflow.keras
from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import MinMaxScaler

#creation of the model
model = Sequential()
model.add(Dense(40, input_dim=5, activation='relu'))
model.add(Dense(40, activation='relu'))
model.add(Dense(1, activation='linear'))
#if you want to check up on your model
model.summary()


# In[ ]:


#Train the model


# In[21]:


model.compile(optimizer='adam', loss='mean_squared_error')


# In[ ]:


#Fit the model


# In[22]:


epochs_hist = model.fit(X_train, y_train, epochs=20, batch_size=25,  verbose=1, validation_split=0.2)


# # STEP#5: EVALUATING THE MODEL

# In[47]:


print(epochs_hist.history.keys())


# In[24]:


plt.plot(epochs_hist.history['loss'])
plt.plot(epochs_hist.history['val_loss'])

plt.title('Model Loss Progression During Training/Validation')
plt.ylabel('Training and Validation Losses')
plt.xlabel('Epoch Number')
plt.legend(['Training Loss', 'Validation Loss'])


# In[25]:


# Gender, Age, Annual Salary, Credit Card Debt, Net Worth

X_Testing = np.array([[1, 50, 50000, 10985, 629312]])


# In[26]:


y_predict = model.predict(X_Testing)
y_predict.shape


# In[27]:


print('Expected Purchase Amount=', y_predict[:,0])


# # EXCELLENT JOB! NOW YOU'VE MASTERED THE USE OF ANNs FOR REGRESSION TASKS!
