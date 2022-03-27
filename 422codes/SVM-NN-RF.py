#!/usr/bin/env python
# coding: utf-8

# In[198]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
data=pd.read_csv("D:/422codes/Melanoma TFRecords 256x256.csv")
#data  #print dataframe


# In[199]:


data_subset=data.dropna(axis = 0, subset = ['sex'])


# In[200]:


from sklearn.impute import SimpleImputer
impute = SimpleImputer(missing_values=np.nan, strategy='mean')
impute.fit(data_subset[['age_approx']])
data_subset.age_approx= impute.transform(data_subset[['age_approx']])


# In[201]:


from sklearn.preprocessing import LabelEncoder
enc = LabelEncoder()
data_subset['sex'] = enc.fit_transform(data_subset['sex'])
#data_subset[['sex']]


# In[202]:


category_enc = pd.get_dummies(data_subset['anatom_site_general_challenge'])
data_total_subset = pd.concat([data_subset,category_enc], axis=1)


# In[203]:


data_total_subset=data_total_subset.drop(['anatom_site_general_challenge'], axis = 1)


# In[204]:


X=data_total_subset[['sex','age_approx','tfrecord' ,'width','height','head/neck','lower extremity','oral/genital','palms/soles','torso','upper extremity']]
X.head()


# In[205]:


from sklearn.model_selection import train_test_split
y = pd.DataFrame(data_total_subset['target'])
X_train, X_test, y_train, y_test = train_test_split(X,data_total_subset['target'], 
                                                    test_size = 0.20, 
                                                    random_state=0,stratify = y)


# In[206]:


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)


# In[207]:


data_total_subset


# In[208]:


from sklearn.svm import SVC
svc = SVC(kernel="linear")
svc.fit(X_train_scaled, y_train)


# In[209]:


print("Training accuracy of the model is {:.2f}".format(svc.score(X_train_scaled, y_train)))
print("Testing accuracy of the model is {:.2f}".format(svc.score(X_test_scaled, y_test)))
pre_svc=svc.score(X_test_scaled, y_test)


# In[210]:


from sklearn.neural_network import MLPClassifier
nnc=MLPClassifier(hidden_layer_sizes=(7), activation="relu", max_iter=100000)


# In[211]:


nnc.fit(X_train_scaled, y_train)


# In[212]:


print("The Training accuracy of the model is {:.2f}".format(nnc.score(X_train_scaled, y_train)))
print("The Testing accuracy of the model is {:.2f}".format(nnc.score(X_test_scaled, y_test)))
pre_nnc=nnc.score(X_test_scaled, y_test)


# In[213]:


from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=50)
rfc.fit(X_train_scaled, y_train)


# In[214]:


print("The Training accuracy of the model is {:.2f}".format(rfc.score(X_train_scaled, y_train)))
print("The Testing accuracy of the model is {:.2f}".format(rfc.score(X_test_scaled, y_test)))
pre_rfc=rfc.score(X_test_scaled, y_test)
##random


# In[215]:


from sklearn.decomposition import PCA 
pca = PCA(n_components=6)


# In[216]:


principal_components= pca.fit_transform(X)
print(principal_components)


# In[217]:


#sum(pca.explained_variance_ratio_)
principal_df = pd.DataFrame(data=principal_components ,columns=["pc1", "pc2","pc3", "pc4","pc5", "pc6"])
#principal_df.head()
principal_df.head


# In[218]:


xn_train, xn_test, yn_train, yn_test = train_test_split(X,data_total_subset['target'], 
                                                    test_size = 0.20, 
                                                    random_state=0,stratify = y)


# In[219]:


scaler.fit(xn_train)
xn_train_scaled = scaler.transform(xn_train)
xn_test_scaled = scaler.transform(xn_test)


# In[220]:


from sklearn.svm import SVC
svc = SVC(kernel="linear")
svc.fit(xn_train_scaled, yn_train)


# In[221]:


print("Training accuracy of the model is {:.2f}".format(svc.score(xn_train_scaled, yn_train)))
print("Testing accuracy of the model is {:.2f}".format(svc.score(xn_test_scaled, yn_test)))
post_svc=svc.score(X_test_scaled, y_test)


# In[222]:


from sklearn.neural_network import MLPClassifier
nnc=MLPClassifier(hidden_layer_sizes=(7), activation="relu", max_iter=100000)


# In[223]:


nnc.fit(xn_train, yn_train)
print("The Training accuracy of the model is {:.2f}".format(nnc.score(xn_train_scaled, yn_train)))
print("The Testing accuracy of the model is {:.2f}".format(nnc.score(xn_test_scaled, yn_test)))
post_nnc=nnc.score(X_test_scaled, y_test)


# In[224]:


from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=50)
rfc.fit(xn_train_scaled, yn_train)


# In[227]:


print("The Training accuracy of the model is {:.2f}".format(rfc.score(xn_train_scaled, yn_train)))
print("The Testing accuracy of the model is {:.2f}".format(rfc.score(xn_test_scaled, yn_test)))
post_rfc=rfc.score(X_test_scaled, y_test)
##random


# In[231]:



position=['svc','nnc','rfc']
position=np.arange(3)
pre_pca = [pre_svc, pre_nnc, pre_rfc]
post_pca = [post_svc, post_nnc, post_rfc]
score= [pre_pca,post_pca]
plt.bar(position+0.00,score[0],color='r',width=0.35)
plt.bar(position+0.25,score[1],color='b',width=0.35)
plt.title('Score of SVC,NNC and RFC before and after pca')
addressed=['SVC', 'NNC', 'RFC']
plt.xticks(position,addressed)
plt.show()


# In[ ]:




