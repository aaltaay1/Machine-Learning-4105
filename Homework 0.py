#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#Homework 0
#Name : Abrar Altaay
#ID: 801166376


# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[2]:


df=pd.read_csv('P:\STORAGE\Downloads\D3.csv')
df.head()# To get first n rows from the dataset default value of n is 5
M=len(df)
M #Print M


# In[3]:


X0=df.values[:,0]# get input values from first column
X1=df.values[:,1]# get input values from second column
X2=df.values[:,2]# get input values from third column
y =df.values[:,3]# get output values from fourt column
m=len(y)# Number of training examples
print('X0 = ',X0[:5])# Show only first 5 records
print('X1 = ',X1[:5])
print('X2 = ',X2[:5])
print('y = ',y[:5])
print('m = ',m)


# In[4]:


#Lets create a matrix with single column of ones
X_Ones = np.ones((m,1))
X_Ones [:5]


# In[5]:


# Using reshape function convert X 1D array to 2D array of dimension 97x1
X_0=X0.reshape(m,1)
X_0[:10]


# In[6]:


# Lets use hstack() function from numpy to stack X_0 and X_1 horizontally (i.e. column
# This will be our final X matrix (feature matrix)
X0 = np.hstack((X_Ones, X_0))


# In[7]:


theta=np.zeros(2)
theta


# In[8]:


def compute_cost(X,y,theta):
    
    predictions=X.dot(theta)
    errors=np.subtract(predictions,y)
    sqrErrors=np.square(errors)
    J=1/(2*m)*np.sum(sqrErrors)
    
    return J


# In[9]:


# Lets compute the cost for theta values
cost=compute_cost(X0,y,theta)
print('The cost for given values of theta_0 and theta_1 =',cost)


# In[10]:


def gradient_descent(X,y,theta,alpha,iterations):
    cost_history=np.zeros(iterations)
    for i in range(iterations):
        predictions=X.dot(theta)
        errors=np.subtract(predictions,y)
        sum_delta=(alpha/m)*X.transpose().dot(errors);
        theta=theta-sum_delta;
        cost_history[i]=compute_cost(X,y,theta)
    return theta,cost_history


# In[11]:


theta=[0.,0.]
iterations=1500;
alpha=0.01;


# In[12]:


theta,cost_history=gradient_descent(X0,y,theta,alpha,iterations)
print('Final value of theta =',theta)
print('cost_history =',cost_history)


# In[13]:


# Since X is list of list (feature matrix) lets take values of column of index 1 only
plt.scatter(X0[:,1],y,color='red',marker='+',label='Training Data')
plt.plot(X0[:,1],X0.dot(theta),color='green',label='Linear Regression')

plt.rcParams["figure.figsize"]=(10,6)
plt.grid()
plt.legend()


# In[14]:


plt.plot(range(1,iterations+1),cost_history,color='blue')
plt.rcParams["figure.figsize"]=(10,6)
plt.grid()
plt.xlabel('Number of iterations')
plt.ylabel('Cost (J)')
plt.title('Convergence of gradient descent - Graph 1')


# In[15]:


# Using reshape function convert X 1D array to 2D array of dimension 97x1
X_1=X1.reshape(m,1)
X_1[:10]


# In[16]:


# Lets use hstack() function from numpy to stack X_0 and X_1 horizontally (i.e. column
# This will be our final X matrix (feature matrix)
X1 = np.hstack((X_Ones, X_1))


# In[17]:


# Lets compute the cost for theta values
cost=compute_cost(X1,y,theta)
print('The cost for given values of theta_0 and theta_1 =',cost)


# In[18]:


theta,cost_history=gradient_descent(X1,y,theta,alpha,iterations)
print('Final value of theta =',theta)
print('cost_history =',cost_history)


# In[19]:


# Since X is list of list (feature matrix) lets take values of column of index 1 only
plt.scatter(X1[:,1],y,color='red',marker='+',label='Training Data')
plt.plot(X1[:,1],X1.dot(theta),color='green',label='Linear Regression')

plt.rcParams["figure.figsize"]=(10,6)
plt.grid()
plt.legend()


# In[20]:


plt.plot(range(1,iterations+1),cost_history,color='blue')
plt.rcParams["figure.figsize"]=(10,6)
plt.grid()
plt.xlabel('Number of iterations')
plt.ylabel('Cost (J)')
plt.title('Convergence of gradient descent - Graph 2')


# In[21]:


# Using reshape function convert X 1D array to 2D array of dimension 97x1
X_2=X2.reshape(m,1)
X_2[:10]


# In[22]:


# Lets use hstack() function from numpy to stack X_0 and X_1 horizontally (i.e. column
# This will be our final X matrix (feature matrix)
X2 = np.hstack((X_Ones, X_2))


# In[23]:


# Lets compute the cost for theta values
cost=compute_cost(X2,y,theta)
print('The cost for given values of theta_0 and theta_1 =',cost)


# In[24]:


theta,cost_history=gradient_descent(X2,y,theta,alpha,iterations)
print('Final value of theta =',theta)
print('cost_history =',cost_history)


# In[25]:


# Since X is list of list (feature matrix) lets take values of column of index 1 only
plt.scatter(X2[:,1],y,color='red',marker='+',label='Training Data')
plt.plot(X2[:,1],X2.dot(theta),color='green',label='Linear Regression')

plt.rcParams["figure.figsize"]=(10,6)
plt.grid()
plt.legend()


# In[26]:


plt.plot(range(1,iterations+1),cost_history,color='blue')
plt.rcParams["figure.figsize"]=(10,6)
plt.grid()
plt.xlabel('Number of iterations')
plt.ylabel('Cost (J)')
plt.title('Convergence of gradient descent - Graph 3')


# In[27]:


#Problem 1C
#It was observed that the first input, X0, had the lowest cost function by far then compared to X1 & X2. 


# In[28]:


#Problem 1 D
#The higher the alpha point to a certain point, the lower the cost function will become. 
#A lower alpha number is producing  a higher cost. 
#Too high of an alpha, the program will crash. Such as 0.5


# In[29]:



############### 
## Problem 2 ##
############### 


# In[30]:


theta=np.zeros(4)
theta


# In[31]:


X3 = np.hstack((X_Ones, X_0, X_1, X_2))
X3[:5]


# In[32]:


# Lets compute the cost for theta values
cost=compute_cost(X3,y,theta)
print('The cost for given values of theta_0 and theta_1 =',cost)


# In[33]:


theta,cost_history=gradient_descent(X3,y,theta,alpha,iterations)
print('Final value of theta =',theta)
print('cost_history =',cost_history)


# In[34]:


#2-A The model with all 3 collumes graphed had the lowest cost 


# In[35]:


plt.plot(range(1,iterations+1),cost_history,color='blue')
plt.rcParams["figure.figsize"]=(10,6)
plt.grid()
plt.xlabel('Number of iterations')
plt.ylabel('Cost (J)')
plt.title('Convergence of gradient descent - Graph 3')


# In[36]:


#2-C 
#The higher the alpha point to a certain point, the lower the cost function will become. 
#A lower alpha number is produces a higher cost. 
#Too high of an alpha, the program will crash. Such as 0.5


# In[37]:


print(theta)


# In[38]:



################ 
## Problem 2D ##
################ 


# In[39]:


def predict(X1, X2, X3): # New function 
    Input = [1, X1, X2, X3] 
    Y_predict = 0
    for i in range(4):
        j = Input[i]*theta[i]  #multiple variable input with calcualted theta
        Y_predict = Y_predict + j 
    print('Predicted Y Value: ', Y_predict)


# In[40]:


predict(1, 1, 1)


# In[41]:


predict(2, 0, 4)


# In[42]:


predict(3, 2, 1)


# In[43]:


print(y)


# In[ ]:




