#!/usr/bin/env python
# coding: utf-8

# In[1]:


#加载词向量
import jieba
import gensim
import gc
import numpy as np


VEC_PATH = '/home/admin/jupyter/Demo/DataSets/merge_sgns_bigram_char300/merge_sgns_bigram_char300.txt'
TRAIN_PATH='/home/admin/jupyter/Demo/DataSets/oppo_data_ronud2_20181107/data_train.txt'
TEST_PATH='/home/admin/jupyter/Demo/DataSets/oppo_round2_test_B/oppo_round2_test_B.txt'
VAL_PATH='/home/admin/jupyter/Demo/DataSets/oppo_data_ronud2_20181107/data_vali.txt'


# In[2]:


import pandas as pd
import gc
import time
from time import  strftime

start_time=time.localtime()
str_stime=strftime("%Y_%m_%d %H:%M:%S",start_time)
print(str_stime)




train_data = pd.read_table(TRAIN_PATH,quoting=3,
        names= ['prefix','query_prediction','title','tag','label'], header= None, encoding='utf-8').astype(str)
val_data = pd.read_table(VAL_PATH,quoting=3 ,
        names = ['prefix','query_prediction','title','tag','label'], header = None, encoding='utf-8').astype(str)



test_data = pd.read_table(TEST_PATH,quoting=3 ,
        names = ['prefix','query_prediction','title','tag'],header = None, encoding='utf-8').astype(str)


#不能使用data.index会有问题 当tain和val合并的时候
test_data['index']=[i for i in range(len(test_data))]

train_data = train_data[train_data['label'].isin(['0','1']) ]
val_data = val_data[val_data['label'].isin(['0','1']) ]


train_data['label'] = train_data['label'].apply(lambda x: int(x))
val_data['label'] = val_data['label'].apply(lambda x: int(x))

print(len(train_data),len(val_data),len(test_data))


data=pd.concat([train_data,val_data])
data=pd.concat([data,test_data])
data['index']=[i for i in range(len(data))]


print(len(data))

del train_data
del val_data
del test_data

gc.collect()


# In[3]:



# nunique

fea_list=['tag','query_prediction','title']

for fea in fea_list:
    gp1=data.groupby('prefix')[fea].nunique().reset_index().rename(columns={fea:"prefix_%s_nuq_num"%fea})
    gp2=data.groupby(fea)['prefix'].nunique().reset_index().rename(columns={'prefix':"%s_prefix_nuq_num"%fea})
    data=pd.merge(data,gp1,how='left',on=['prefix'])
    data=pd.merge(data,gp2,how='left',on=[fea])   
    gc.collect()
    
fea_list=['tag','query_prediction'] 

for fea in fea_list:
    gp1=data.groupby('title')[fea].nunique().reset_index().rename(columns={fea:"title_%s_nuq_num"%fea})
    gp2=data.groupby(fea)['title'].nunique().reset_index().rename(columns={'title':"%s_title_nuq_num"%fea})
    data=pd.merge(data,gp1,how='left',on=['title'])
    data=pd.merge(data,gp2,how='left',on=[fea])   
    gc.collect()
    


# In[4]:


fea_list=[['tag','title']]

for fea in fea_list:
    gp2=data.groupby(fea)['prefix'].nunique().reset_index().rename(columns={'prefix':"%s_prefix_nuq_num"%                                                                                  "_".join(fea)})
    data=pd.merge(data,gp2,how='left',on=fea)   
    gc.collect()
    


# In[5]:


end_time=time.localtime()
str_stime=strftime("%Y_%m_%d %H:%M:%S",end_time)
print(str_stime)
feature=[i for i in data.columns if '_num' in i  or'ratio' in i  or '_ca' in i]


data.sort_values(by='index',inplace=True)
data[feature].to_csv("./CSV/data_unique_fea.csv", index=False,compression='bz2')


# In[ ]:




