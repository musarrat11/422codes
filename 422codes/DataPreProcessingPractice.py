#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[2]:


import pandas as pd
import numpy as np
data=pd.read_csv("D:/422codes/Melanoma TFRecords 256x256.csv")
data


# In[17]:


data.head(10)


# In[19]:


data.target


# In[25]:


data['target']


# In[27]:


data.iloc[1]


# In[33]:


data.iloc[:10, 5]


# In[37]:


data.iloc[1:4]#full row
#only column data.iloc[1:4, 0] it will give the value of 0th column from row 1to3


# In[38]:


data.loc[0, 'width']# 0->row width-> feature name of the specific column


# In[39]:


data.loc[:, ['benign_malignant', 'anatom_site_general_challenge', 'patient_code']]
# : -> all rows , [....] -> specific features


# # When choosing or transitioning between loc and iloc, there is one "gotcha" worth keeping in mind, which is that the two methods use slightly different indexing schemes.
# 
# iloc uses the Python stdlib indexing scheme, where the first element of the range is included and the last one excluded. So 0:10 will select entries 0,...,9. loc, meanwhile, indexes inclusively. So 0:10 will select entries 0,...,10.
# 
# Why the change? Remember that loc can index any stdlib type: strings, for example. If we have a DataFrame with index values Apples, ..., Potatoes, ..., and we want to select "all the alphabetical fruit choices between Apples and Potatoes", then it's a lot more convenient to index df.loc['Apples':'Potatoes'] than it is to index something like df.loc['Apples', 'Potatoet'] (t coming after s in the alphabet).This is particularly confusing when the DataFrame index is a simple numerical list, e.g. 0,...,1000. In this case df.iloc[0:1000] will return 1000 entries, while df.loc[0:1000] return 1001 of them! To get 1000 elements using loc, you will need to go one lower and ask for df.loc[0:999].
# 
# Otherwise, the semantics of using loc are the same as those for iloc.

# In[47]:


data.benign_malignant == 'malignant'


# In[10]:


data.loc[data.benign_malignant == 'malignant'] #bujhinai
#reviews.loc[(reviews.country == 'Italy') & (reviews.points >= 90)]


# In[13]:


data.loc[data.anatom_site_general_challenge.isin(['upper extremity', 'torso'])]
#The first is isin. isin is lets you select data whose value "is in" a list of values.
# For example, here's how we can use it to select wines only from Italy or France:
#reviews.loc[reviews.country.isin(['Italy', 'France'])]
#The second is isnull (and its companion notnull). 
#These methods let you highlight values which are (or are not) empty (NaN). 
#For example, to filter out wines lacking a price tag in the dataset, here's what we would do:
#reviews.loc[reviews.price.notnull()]


# In[16]:


data.loc[:, data.notnull().any()] # Select columns without NaNs


# # The second is isnull (and its companion notnull). These methods let you highlight values which are (or are not) empty (NaN). For example, to filter out wines lacking a price tag in the dataset, here's what we would do:
# 
# reviews.loc[reviews.price.notnull()]

# In[51]:


data.describe()


# In[53]:


data.diagnosis.describe()


# In[54]:


data.diagnosis.unique()


# In[55]:


data.diagnosis.value_counts()


# In[56]:


data[pd.isnull(data.sex)]


# In[57]:


data[pd.isnull(data.age_approx)]


# In[11]:


data[pd.isnull(data.anatom_site_general_challenge)]


# Replacing missing values is a common operation. Pandas provides a really handy method for this problem: fillna(). fillna() provides a few different strategies for mitigating such data. For example, we can simply replace each NaN with an "Unknown":
# 
# reviews.region_2.fillna("Unknown")

# In[12]:


data.isnull().sum()


# In[24]:


#data_subset=data.dropna(how='any') # delete all rows withh null value
data_subset=data.dropna(axis = 0, subset = ['sex'])
data_subset.isnull().sum()


# In[25]:


data_subset.shape


# # Check how many values are missing in the category_desc column
# print("Number of rows with null values in category_desc column: ", volunteer['category_desc'].isnull().sum())
# 
# # Subset the volunteer dataset
# 
# volunteer_subset = volunteer[volunteer['category_desc'].notnull()]
# 
# # Print out the shape of the subset
# print("Shape after removing null values: ", volunteer_subset.shape)
# 
# print("Shape of dataframe before dropping:", volunteer.shape)
# volunteer = volunteer.dropna(axis = 0, subset = ['category_desc'])
# print("Shape after dropping:", volunteer.shape)

# In[28]:


data_subset.age_approx.fillna(50)


# In[29]:


data_subset.isnull().sum()


# In[31]:


#data.isnull().sum()
#data[['sex','age_approx']]
data.loc[(data.diagnosis=='unknown') & (data.benign_malignant == 'malignant')]


# In[ ]:


from sklearn.impute import SimpleImputer
impute = SimpleImputer(missing_values=np.nan, strategy='mean')
impute.fit(data[['age_approx']])
data['age_approx'] = impute.transform(data[['age_approx']])
data.isnull().sum()

