#!/usr/bin/env python
# coding: utf-8

# In[66]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
data=pd.read_csv("D:/422codes/Melanoma TFRecords 256x256.csv")
#data  #print dataframe


# In[67]:


data_subset=data.dropna(axis = 0, subset = ['sex'])


# In[68]:


from sklearn.impute import SimpleImputer
impute = SimpleImputer(missing_values=np.nan, strategy='mean')
impute.fit(data_subset[['age_approx']])
data_subset.age_approx= impute.transform(data_subset[['age_approx']])


# In[69]:


from sklearn.preprocessing import LabelEncoder
enc = LabelEncoder()
data_subset['sex'] = enc.fit_transform(data_subset['sex'])
#data_subset[['sex']]


# In[70]:


category_enc = pd.get_dummies(data_subset['anatom_site_general_challenge'])
data_total_subset = pd.concat([data_subset,category_enc], axis=1)


# In[71]:


data_total_subset=data_total_subset.drop(['anatom_site_general_challenge'], axis = 1)


# In[72]:


from sklearn.model_selection import train_test_split
y = pd.DataFrame(data_total_subset['target'])
X_train, X_test, y_train, y_test = train_test_split(data_total_subset[['sex','age_approx','tfrecord' ,'width','height','head/neck','lower extremity','oral/genital','palms/soles','torso','upper extremity']], 
                                                    data_total_subset['target'], 
                                                    test_size = 0.20, 
                                                    random_state=0,stratify = y)


# In[73]:


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)


# In[74]:


import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score

#Train the model
logisticModel = LogisticRegression()
logisticModel.fit(X_train, y_train) #Training the model
predictions = logisticModel.predict(X_test)
print(predictions)# printing predictions


# In[75]:


logistic_accuracy=accuracy_score(y_test, predictions)
print(logistic_accuracy)


# In[1]:


from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier(criterion='entropy',random_state=1)
dtc.fit(X_train,y_train)
y_pred = dtc.predict(X_test)
decision_accuracy=accuracy_score(y_pred,y_test)
print(decision_accuracy)


# In[77]:


x=decision_accuracy
y=logistic_accuracy
z=np.array([y,x])
q=np.array(["Logistic Regression", "Decision Tree"])
plt.bar(q, z, color = "#4CAF90")
plt.show()


# In[ ]:




