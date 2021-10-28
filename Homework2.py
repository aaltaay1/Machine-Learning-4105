#!/usr/bin/env python
# coding: utf-8

# In[1]:


#github: https://github.com/aaltaay1/Machine-Learning-4105
#Homework 1
#Name : Abrar Altaay
#ID: 801166376


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# In[2]:


#Reading training data
dataset = pd.read_csv('P:\STORAGE\Desktop\School\ML\Homework 2\diabetes.csv')
dataset.head()


# In[3]:


#problem #1 -
# Lines to find the data: 
# confusion_matrix ln[10]
# Evaluation Results ln [11]
# Heat Map ln[11]


# In[4]:


#Show values without scietfici notation
#np.set_printoptions(suppress=True)

X = dataset.iloc[:, [0,1,2,3,4,5,6,7]].values
Y = dataset.iloc[:, 8].values
X[0:5]


# In[5]:


#Train our data set with a 80 / 20 split model
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)


# In[6]:


#Standardization
#feature scaling data between 0 and 1 for better reading
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)


# In[7]:


#Make an instance classifier of the object LogisticRegression and give random_state =
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state=0)
classifier.fit(X_train, Y_train)


# In[8]:


Y_pred = classifier.predict(X_test)


# In[9]:


Y_pred[0:9]


# In[10]:


#Model evualtion metrics to get more accurate results
from sklearn.metrics import confusion_matrix
cnf_matrix = confusion_matrix(Y_test, Y_pred)
cnf_matrix


# In[11]:


#Let's evaluate the model using model evaluation metrics such as accuracy, precision, a
from sklearn import metrics
print("Accuracy:",metrics.accuracy_score(Y_test, Y_pred))
print("Precision:",metrics.precision_score(Y_test, Y_pred))
print("Recall:",metrics.recall_score(Y_test, Y_pred))


# In[12]:


#matrix using matp
#visualize the confusion matrix using Heatmap.
import seaborn as sns
class_names=[0,1] # name of classes
fig, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)
# create heatmap
sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu" ,fmt='g')
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title('Confusion matrix', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')


# In[13]:


#For #1 the data was imported then split to an 80 / 20 training and testing data set
# then the data was scaled from 0 to 1 for better accuracy
# Then using using confusion matrix we were able to evaluate the data 
# We oupted the Accuracy, Precision, and Recall as shown in ln[10]

#After the data was outputed, we were able to create a 'heat map' to 
# represent the accuracy results


# In[ ]:





# In[ ]:





# In[ ]:





# In[14]:


#Part #2 - Naive Gausian Bayes 

#The logistis regression is more accurate than the Naive Bays from ln 22 and 23
# Confusion Matrix = ln 17
# classifier ln 16


# In[15]:


from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB() #setting classifier as log regression
classifier.fit(X_train, Y_train) #fitting the training data for log regression


# In[16]:


Y2_pred = classifier.predict(X_test) 

Y2_pred


# In[17]:


from sklearn.metrics import confusion_matrix,accuracy_score
cm_2 = confusion_matrix(Y_test, Y2_pred)
ac_2 = accuracy_score(Y_test, Y2_pred)

cm_2 #display the confusion_matrix


# In[18]:


ac_2 # display accuracy score


# In[19]:


#matrix using matp
#visualize the confusion matrix using Heatmap.
import seaborn as sns
class_names=[0,1] # name of classes
fig, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)
# create heatmap

sns.heatmap(pd.DataFrame(cm_2), annot=True, cmap="YlGnBu" ,fmt='g')
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title('Confusion matrix', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')


# In[ ]:





# In[ ]:





# In[ ]:





# In[20]:


#Problem #3 - Logistic regression for problem 1 is more accurate than k=5
#           - The logistic regression is ALSO more accurate k = 10 shown in cell 26


# In[21]:


from sklearn.model_selection import RepeatedKFold
k = RepeatedKFold(n_splits=5, n_repeats=5, random_state=1)

classifier = LogisticRegression(random_state=0) 
from sklearn.model_selection import cross_validate
Y3_pred = cross_validate(classifier, X, Y, scoring={'accuracy', 'precision', 'recall'}, cv=k, n_jobs=-1)


from numpy import mean
from numpy import std
print("For Kfold where K = 5 \n")
print("Accuracy: ", np.mean(Y3_pred['test_accuracy']))
print("Precision: ", np.mean(Y3_pred['test_precision']))
print("Recall: ", np.mean(Y3_pred['test_recall']))


# In[22]:


from sklearn.model_selection import RepeatedKFold
k = RepeatedKFold(n_splits=10, n_repeats=10, random_state=1)

classifier = LogisticRegression(random_state=0) 
from sklearn.model_selection import cross_validate
Y3_pred = cross_validate(classifier, X, Y, scoring={'accuracy', 'precision', 'recall'}, cv=k, n_jobs=-1)


print("For Kfold where K = 10 \n")
print("Accuracy: ", np.mean(Y3_pred['test_accuracy']))
print("Precision: ", np.mean(Y3_pred['test_precision']))
print("Recall: ", np.mean(Y3_pred['test_recall']))


# In[ ]:





# In[ ]:





# In[ ]:





# In[23]:


# Problem 4 - k=5

# When looping Naive bays, you get seprate testing and training data fro each run, 
# and will average the accuracies together. which is not smart to do as the k-fold is for 
# Naive bays is not needed
# for k=5 our accuracy compared to #2 is less accurate, this is becuase k-fold is not needed
#when splitting the same data a different ammount of times, then the data wont work with each other.


# In[24]:


k = RepeatedKFold(n_splits=5, n_repeats=5, random_state=1)

classifier = GaussianNB()
Y3_pred = cross_validate(classifier, X, Y, scoring={'accuracy', 'precision', 'recall'}, cv=k, n_jobs=-1)

print("For Kfold where K = 5 \n")
print("Accuracy: ", np.mean(Y3_pred['test_accuracy']))
print("Precision: ", np.mean(Y3_pred['test_precision']))
print("Recall: ", np.mean(Y3_pred['test_recall']))


# In[25]:


k = RepeatedKFold(n_splits=10, n_repeats=10, random_state=1)

classifier = GaussianNB()
Y3_pred = cross_validate(classifier, X, Y, scoring={'accuracy', 'precision', 'recall'}, cv=k, n_jobs=-1)


print("For Kfold where K = 10 \n")
print("Accuracy: ", np.mean(Y3_pred['test_accuracy']))
print("Precision: ", np.mean(Y3_pred['test_precision']))
print("Recall: ", np.mean(Y3_pred['test_recall']))

