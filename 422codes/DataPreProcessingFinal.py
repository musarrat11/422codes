#!/usr/bin/env python
# coding: utf-8

# In[191]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
data=pd.read_csv("D:/422codes/Melanoma TFRecords 256x256.csv")
data


# In[192]:


data.keys()


# In[193]:


fig,ax = plt.subplots()
ax.hist(data['tfrecord'])
ax.set_title('Data Arrangement')
ax.set_xlabel('age')
ax.set_ylabel('no. of patient')


# In[194]:


fig,ax = plt.subplots()
ax.hist(data['target'])
ax.set_title('Frequency')
ax.set_xlabel('benign_malignant')
ax.set_ylabel('no. of patient')


# In[195]:


data['anatom_site_general_challenge'].unique()


# In[196]:


#data_subset=data.loc[data.sex.notnull()]
data_subset=data.dropna(axis = 0, subset = ['sex'])
data_subset


# In[197]:


data_subset


# In[198]:


from sklearn.impute import SimpleImputer
impute = SimpleImputer(missing_values=np.nan, strategy='mean')
impute.fit(data_subset[['age_approx']])
data_subset['age_approx'] = impute.transform(data_subset[['age_approx']])
data_subset.isnull().sum()


# In[199]:


from sklearn.preprocessing import LabelEncoder

# Set up the LabelEncoder object
enc = LabelEncoder()

# Apply the encoding to the "Accessible" column
data_subset['sex'] = enc.fit_transform(data_subset['sex'])

# Compare the two columns
data_subset[['sex']]


# In[200]:


# Transform the category_desc column
category_enc = pd.get_dummies(data_subset['anatom_site_general_challenge'])

# Take a look at the encoded columns
data_total_subset = pd.concat([data_subset,category_enc], axis=1)


# In[201]:


data_total_subset


# In[202]:


data_total_subset=data_total_subset.drop(['anatom_site_general_challenge'], axis = 1)


# In[203]:


data_total_subset.isnull().sum()


# In[212]:


from sklearn.model_selection import train_test_split
y = pd.DataFrame(data_total_subset['target'])
X_train, X_test, y_train, y_test = train_test_split(data_total_subset[['sex','age_approx','tfrecord' ,'width','height','head/neck','lower extremity','oral/genital','palms/soles','torso','upper extremity']], 
                                                    data_total_subset['target'], 
                                                    test_size = 0.28, 
                                                    random_state=0,stratify = y)


# In[210]:


X_train


# In[215]:


from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

scaler.fit(X_train)

X_train_scaled = scaler.transform(X_train)

print("per-feature minimum before scaling:\n {}".format(X_train.min(axis=0)))
print("per-feature maximum before scaling:\n {}".format(X_train.max(axis=0)))


# In[216]:


print("per-feature minimum after scaling:\n {}".format(
    X_train_scaled.min(axis=0)))
print("per-feature maximum after scaling:\n {}".format(
    X_train_scaled.max(axis=0)))


# In[ ]:




