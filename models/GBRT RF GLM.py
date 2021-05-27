#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import pandas as pd
import datetime as dt
import urllib.request, json
import os
import numpy as np
import tensorflow as tf # This code has been tested with TensorFlow 1.6
from sklearn.preprocessing import MinMaxScaler


# ### Read Data

# In[2]:


df = pd.read_csv(r"/Users/georgemao/Library/Mobile Documents/com~apple~CloudDocs/HKUST/HK/Portfolio/data.csv")
#/Users/georgemao/Desktop/Portfolio Project/data_GroupProject.csv
df.head()

df.next_ret = pd.to_numeric(df.next_ret,errors='coerce')
df = df[~np.isnan(df.next_ret)].reset_index(drop=True)

df['divyield'] = df['divyield'].str.replace('%', '').astype('float')

df.columns = [name.lower() for name in df.columns]

ind_train = df[df.year.isin(range(1980,2000))].index # 1980 to 1999
ind_test = df[df.year.isin(range(2000,2020))].index # 2000 to 2019

df_train = df.loc[ind_train,:].copy().reset_index(drop=True)
df_test = df.loc[ind_test,:].copy().reset_index(drop=True)
  
feats_not_to_use=['next_ret']
feats_to_use = [feat for feat in df.columns if feat not in feats_not_to_use]
target = 'next_ret'


# ### Normalize Data

# In[3]:


feats = feats_to_use
def normalize(series):
  return (series-series.mean(axis=0))/series.std(axis=0)

mean = df_train[feats].mean(axis=0)
df_train[feats] = df_train[feats].fillna(mean)      

data_train = df_train[feats].apply(normalize).values

mean = df_test[feats].mean(axis=0)
df_test[feats] = df_test[feats].fillna(mean)      

data_test = df_test[feats].apply(normalize).values


# ### GBRT

# In[4]:


from sklearn import metrics
from sklearn.ensemble import GradientBoostingRegressor

GBDT = GradientBoostingRegressor(loss='huber')
GBDT.fit(data_train,df_train[target])
GBDT_score = GBDT.score(data_test,df_test[target])


# In[5]:


def importance_plot(features,feature_importance):
    #feature_importance = 100.0 * (feature_importance/feature_importance.max())
    sorted_idx = np.argsort(feature_importance)
    sorted_feats = []
    for i in sorted_idx:
        sorted_feats.append(features[i])

    pos = np.arange(sorted_idx.shape[0])+.5
    plt.figure(figsize=(20,20))
    plt.barh(pos,feature_importance[sorted_idx],align='center')
    plt.yticks(pos,sorted_feats)
    plt.xlabel('Relative Importance')
    plt.title('Variable Importance')
    plt.show()
    
importance_plot(feats,GBDT.feature_importances_)


# ### RF

# In[57]:


from sklearn.ensemble import RandomForestRegressor

RF = RandomForestRegressor()
RF.fit(data_train,df_train[target])
RF_score = RF.score(data_test,df_test[target])
print(RF_score)


# In[58]:


importance_plot(feats,RF.feature_importances_)


# ### GLM

# In[129]:


from sklearn.linear_model import LassoCV

GLM = LassoCV()
GLM.fit(data_train, df_train[target])
GLM.score(data_test,df_test[target])

from eli5.permutation_importance import get_score_importances

# ... load data, define score function
def score(X, y):
    y_pred = GLM.predict(X)
    return r2_score(y, y_pred)

base_score, score_decreases = get_score_importances(score, data_test, df_test[target])
GLM_feature_importances = abs(np.mean(score_decreases, axis=0))


# In[130]:


importance_plot(feats,GLM_feature_importances)


# from group_lasso import GroupLasso
# from sklearn.metrics import r2_score
# from eli5.sklearn import PermutationImportance
# 
# GL = GroupLasso()
# GL.fit(data_train, df_train[target])
# yhat = GL.predict(data_test)
# GL_score = r2_score(df_test[target],yhat)
# 
# from eli5.permutation_importance import get_score_importances
# 
# 
# def score(X, y):
#     y_pred = GL.predict(X)
#     return r2_score(y, y_pred)
# 
# base_score, score_decreases = get_score_importances(score, data_test, df_test[target])
# feature_importances = np.mean(score_decreases, axis=0)

# In[135]:


Limp = np.repeat(0,len(feats))
GBDT_imp = pd.DataFrame({'GBDT':GBDT.feature_importances_/sum(GBDT.feature_importances_),'RF':RF.feature_importances_/sum(RF.feature_importances_),'GLM':GLM_feature_importances/sum(GLM_feature_importances)})
GBDT_imp.index = feats

import seaborn as sns

# Draw a heatmap with the numeric values in each cell
f, ax = plt.subplots(figsize=(15, 20))
sns.heatmap(GBDT_imp, annot=False, linewidths=0.2, cmap="ocean_r",ax=ax)


# In[120]:





# In[ ]:




