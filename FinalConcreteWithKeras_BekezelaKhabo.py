#!/usr/bin/env python
# coding: utf-8

# ## Business Understanding
# Since circa 150BC, when the Roman Empire implemented the use of concrete "opus caementicium" in the majority of their impressive construction projects, some still standing today, concrete is now ubuquitous.
# 
# Global construction work done is expected to increase to US13.9 trillion in 15years time, up from US$9.7 trillion in 2022. This will be pushed by superpower construction markets in China, the United States and India. Further, the Philippines, Vietnam, Malaysia and Indonesia are anticiptaed to be the fastest growing construction markets during that period [Oxford Economics](https://www.oxfordeconomics.com/resource/global-construction-futures/)
# 
# Selecting the correct amounts of materials for making concrete of the required compressive strength not only has implications on the durability of projects, like the Roman Empire projects, but also on the cost of the projects, or indeed their feasiblity before construction has even begun.
# 
# In this project, Bekezela Bobbie Khabo, an AI Engineering student with IBM, aims to harness the power of Deep Learning to address this very challenge through use of the Keras Python API running on Tensorflow to perform Regression on the following [dataset](https://cocl.us/concrete_data
# ). 

# # The data
# Depending on the different quantities of the seven constituent  materials below, the resulting concrete will have different compressive stregths. Results of compressive strengths depending different compositions of the seven materials are compiled in a CSV format referenced above. The seven materials, called predictors moving forward, are, 
# ### 1. Cement: in cubic metres
# ### 2. Blast Furnace Slag: in cubic metres
# ### 3. Fly Ash: in cubic metres
# ### 4. Water: in cubic metres
# ### 5. Superplasticizer: in cubic metres
# ### 6. Coarse Aggregate: in cubic metres
# ### 7. Fine Aggregate: in cubic metres
# 
# An additonal predictor, which is not a material per se, is
# ### 8. Age: in days
# 
# 
# The target variable is 
# ### Compressive strength: in megapascals

# In[1]:


import numpy as np
import pandas as pd
import tensorflow as tf 


# Importing the data to a Pandas dataframe

# In[3]:


concrete_data=pd.read_csv("https://cocl.us/concrete_data")

concrete_data


# In[7]:


#Checking shape of data in terms of rows and columns
concrete_data.shape


# As expected, the dataframe has 8 columns- eight predictor columns, and one target column.
# 
# It has 1030 different combinations of the materials and ages, each with its own compressive strength.

# In[4]:


concrete_data.describe()


# As noted under "count"- ech column has 1030 entries, a clean dataset. Ranges are as follows,
# 
# Cement from 102.0 to 540.0, with median 272.9
# 
# Blast Furnace Slag from 0 to 359.4, with median 22.0
# 
# Fly ash from 0 to 200.1, with median 0.
# 
# Water from 121.8 to 247.0 with median 185.0
# 
# Superplasticiser from 0.0 to 32.2, with median 6.4
# 
# Coarse Agreegate from 801.0 to 1145.0, with median 968.0
# 
# Fine Agreegate from 594.0 to 992.6, with median 779.5
# 
# Age from 1day to 365days, median duration 28days.
# 
# The range of Compressive Strengths are from 2.3 to 82.6, with median strength of 34.4

# In[6]:


concrete_data.isnull().sum()


# In[8]:


#Defining the predictors as x, and target as y
concrete_data_columns=concrete_data.columns

x=concrete_data[concrete_data_columns[concrete_data_columns != 'Strength']]
y=concrete_data['Strength'] 


# In[9]:


x.head()


# In[10]:


y.head()


# In[12]:


# the eight independent variables
n_cols = x.shape[1]


# In[13]:


#importing scikitlearn and keras before baseline model
import keras
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

def data_split_random (x,y,seed):
    x_train,x_test, y_train, y_test=train_test_split(x, y, test_size=0.3, random_state=0)
    return x_train, x_test, y_train, y_test 


# In[21]:


#Defining a function to return one iteration of model
def build_baseline_model():
    baseline_model= Sequential()   #defining the model
    baseline_model.add(Dense(10, activation='relu', input_shape=(n_cols,)))
    baseline_model.add(Dense(10, activation='relu'))
    baseline_model.add(Dense(1))
    
    
    baseline_model.compile(optimizer='adam', loss='mean_squared_error')
    return baseline_model


#Repeating steps fifty times using a for loop
mse_list=[]
predicted_list={}

for i in range(50):
    if (i + 1) % 5 == 0:
        print ("Computing the {}th iteration".format(i +1))
        
    model=build_baseline_model()
    x_train, x_test, y_train, y_test=data_split_random(x, y, i)
    model.fit(x_train, y_train, epochs=50, verbose=0)
    y_hats=model.predict(x_test)
    mse=mean_squared_error(y_test, y_hats)
    mse_list.append(mse)
    predicted_list[i]={"y_test":y_test, "Y_hats": y_hats}
    
    

#Outputting the mean MSE, and its std dev
mse_mean=np.mean(mse_list)
mse_std=np.std(mse_list)
print("Mean MSE:{:.2f}, and Std Dev of MSE:{:.2f}".format(mse_mean, mse_std))


# In[22]:


#Repeated steps but this time with normalised predictors

x_norm=(x-x.mean())/x.std()

x_norm.head()


# In[23]:


#Repeating steps fifty times using a for loop for normalised predictors
mse_list_norm=[]
predicted_list_norm={}

for i in range(50):
    if (i + 1) % 5 == 0:
        print ("Computing the {}th iteration".format(i +1))
        
    model=build_baseline_model()
    x_train, x_test, y_train, y_test=data_split_random(x_norm, y, i)
    model.fit(x_train, y_train, epochs=50, verbose=0)
    y_hats=model.predict(x_test)
    mse=mean_squared_error(y_test, y_hats)
    mse_list_norm.append(mse)
    predicted_list_norm[i]={"y_test":y_test, "Y_hats": y_hats}
    
    

#Outputting the mean MSE, and its std dev
mse_mean_norm=np.mean(mse_list_norm)
mse_std_norm=np.std(mse_list_norm)
print("Mean of Normaslised MSE:{:.2f}, and Std Dev of Normalised MSE:{:.2f}".format(mse_mean_norm, mse_std_norm))


# In[24]:


#Repeating steps fifty times using a for loop for normalised predictors with hundred epochs
mse_list_hundy=[]
predicted_list_hundy={}

for i in range(50):
    if (i + 1) % 5 == 0:
        print ("Computing the {}th iteration".format(i +1))
        
    model=build_baseline_model()
    x_train, x_test, y_train, y_test=data_split_random(x_norm, y, i)
    model.fit(x_train, y_train, epochs=100, verbose=0)
    y_hats=model.predict(x_test)
    mse=mean_squared_error(y_test, y_hats)
    mse_list_hundy.append(mse)
    predicted_list_hundy[i]={"y_test":y_test, "Y_hats": y_hats}
    
    

#Outputting the mean MSE, and its std dev
mse_mean_hundy=np.mean(mse_list_hundy)
mse_std_hundy=np.std(mse_list_hundy)
print("Mean of Normaslised MSE with Hundred Epochs :{:.2f}, and Std Dev of Normalised MSE with Hundred Epochs:{:.2f}".format(mse_mean_hundy, mse_std_hundy))


# In[25]:


#Repeat with normalised predictors but with three hidden nodes
def build_three_layer_model():
    baseline_model= Sequential()   #defining the model
    baseline_model.add(Dense(10, activation='relu', input_shape=(n_cols,)))
    baseline_model.add(Dense(10, activation='relu'))
    baseline_model.add(Dense(10, activation='relu'))
    baseline_model.add(Dense(10, activation='relu'))
    baseline_model.add(Dense(1))
    
    
    baseline_model.compile(optimizer='adam', loss='mean_squared_error')
    return baseline_model


#Repeating steps fifty times using a for loop
mse_list_three_layer=[]
predicted_list_three_layer={}

for i in range(50):
    if (i + 1) % 5 == 0:
        print ("Computing the {}th iteration".format(i +1))
        
    model=build_three_layer_model()
    x_train, x_test, y_train, y_test=data_split_random(x_norm, y, i)
    model.fit(x_train, y_train, epochs=50, verbose=0)
    y_hats=model.predict(x_test)
    mse=mean_squared_error(y_test, y_hats)
    mse_list_three_layer.append(mse)
    predicted_list_three_layer[i]={"y_test":y_test, "Y_hats": y_hats}
    
    

#Outputting the mean MSE, and its std dev
mse_mean_three_layer=np.mean(mse_list_three_layer)
mse_std=np.std(mse_list_three_layer)
print("Mean MSE of Three Layer Model:{:.2f}, and Std Dev of MSE of Three Layer Model:{:.2f}".format(mse_mean, mse_std))


# # Results before Normalisation of predictor variables
# ## Mean MSE:174.99, and Std Dev of MSE:222.74  
# ### There is so much variation in the 50 computed MSEs before normalising the data emphasizing the need to normalize data prior to compiling, running, and evaluating the model 
# 
# # Results after Normalisation of predictor variables
# ## Mean of Normalised MSE:135.24, and Std Dev of Normalised MSE:6.36  
# ### After the data has been normalised, the MSE has decreased, which is ideal. More importantly, the variation in the computed 50MSEs decreased dramatically, further attesting to the need to normalise predictor variables so that they carry equal weight before modeling using neural networks. This version of the regression model is better in all aspects than the previous. 
# 
# # Results after running 100 epochs for each iteration
# ## Mean of Normalised MSE with Hundred Epochs :102.78, and Std Dev of Normalised MSE with Hundred Epochs:9.20 
# ### Afterrunning 100 epochs each iteration, the MSE has decreased even further, again, preferable. However, variation in the MSEs has increased. There is more variation in the MSEs. Its is more accurate, but becoming less precise. 
#     
#     
# # Results after adding hidden layers to three total
# ## Mean MSE of Three Layer Model:174.99, and Std Dev of MSE of Three Layer Model:16.67
# ### After adding some hidden layers to a total of three, the error in the model has increased to almost the level prior to normalisation, albeit with less variation in the error. I attribute this to the regression model being overtrained on the training data, resulting in overfitting. This means it performs less well on "unseen" testing portion of the data. Training of the regression model to this point is less than ideal, and should have stopped when the error of the model on "unseen" test data began to get worse. 
# 
# 
# 

# In[ ]:




