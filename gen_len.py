#!/usr/bin/env python
# coding: utf-8

# In[7]:


#加载词向量
import jieba
import gensim
import gc
import numpy as np


VEC_PATH = '/home/admin/jupyter/Demo/DataSets/merge_sgns_bigram_char300/merge_sgns_bigram_char300.txt'
TRAIN_PATH='/home/admin/jupyter/Demo/DataSets/oppo_data_ronud2_20181107/data_train.txt'
TEST_PATH='/home/admin/jupyter/Demo/DataSets/oppo_round2_test_B/oppo_round2_test_B.txt'
VAL_PATH='/home/admin/jupyter/Demo/DataSets/oppo_data_ronud2_20181107/data_vali.txt'



# In[8]:


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


# In[9]:



import gc
import re 
import json

data['prefix_equal_title_ca']=data.apply(lambda x: 1 if x.prefix.lower()==x.title.lower() else 0,axis=1)
data['title_in_query_ca']=data.apply(lambda x:1 if x.title.lower() in x.query_prediction.lower() else 0,axis=1)
data['prefix_in_title_head_ca']=data.apply(lambda x: 1 if x.prefix.lower()==x.title[:len(x.prefix)].lower() else 0, axis=1)
data['title_in_query_ca']=data.apply(lambda x:1 if x.title.lower() in x.query_prediction.lower() else 0,axis=1)
data['prefix_in_title_ca']=data.apply(lambda x:1 if x.prefix.lower() in x.title.lower() else 0,axis=1)


data['title_len_num']=data['title'].map(lambda x:len(x))
data['prefix_len_num']=data['prefix'].map(lambda x:len(x))
data['prefix_diff_title_len_num']=data['title_len_num']-data['prefix_len_num']
data['query_prediction_len_num']=data['query_prediction'].map(lambda x:len(str(x)))
data['prefix_dev_title_num']=data['prefix_len_num']/data['title_len_num']
data['query_prediction_dev_title_num']=data['query_prediction_len_num']/data['title_len_num']
gc.collect()



def include_num(prefix):
    
    s=re.search("\d",str(prefix))
    if s!=None:
        return 1
    else:
        return 0

def include_zm(prefix):
    s=re.search("[a-zA-Z]",str(prefix))
    if s!=None:
        return 1
    else:
        return 0
    
def include_ch(prefix):
    s=re.search("[\u4e00-\u9fa5]+",str(prefix))
    
    if s!=None:
        return 1
    else:
        return 0
    

data['prefix_inc_sz_ca']=data.apply(lambda x:include_num(x.prefix),axis=1)
data['prefix_inc_zm_ca']=data.apply(lambda x:include_zm(x.prefix),axis=1)
data['prefix_inc_ch_ca']=data.apply(lambda x:include_ch(x.prefix),axis=1)


# In[10]:


import Levenshtein

gc.collect()

def extract_key(pred):
    try:
        pred = eval(pred)
    except:
        return [pred]
    pred = sorted(pred.items(), key=lambda x: x[1], reverse=True)
    key_lst=[]
    for i in range(10):
        if len(pred)<i+1:
#             pred_prob_lst.append(0)
            break
        else:
            key_lst.append(pred[i][0])
    return key_lst

data['q_key_list']=data.query_prediction.map(lambda x:extract_key(x))

data['max_title_ca']=data.q_key_list.map(lambda x:x[0] if len(x)>0 else 'nan')


# In[11]:


import numpy as np

data['q_keys_max_len_num']=data.q_key_list.map(lambda x:max([len(i) for i in x]) if  len(x)>0  else np.nan)
data['q_keys_min_len_num']=data.q_key_list.map(lambda x:min([len(i) for i in x]) if  len(x)>0  else np.nan)
data['q_keys_mean_len_num']=data.q_key_list.map(lambda x:sum([len(i) for i in x])/len(x)if  len(x)>0  else np.nan)

data['qkmaxl_title_num']=data['q_keys_max_len_num']-data['title_len_num']
data['qkminl_title_num']=data['title_len_num']-data['q_keys_min_len_num']
data['qkmeanl_title_num']=data['q_keys_mean_len_num']-data['title_len_num']

print("长度特征提取完毕")
gc.collect()


# In[12]:



feature=[i for i in data.columns if '_num' in i  or'ratio' in i  or '_ca' in i]


data.sort_values(by='index',inplace=True)
data[feature].to_csv("./CSV/data_len_fea.csv", index=False,compression='bz2')


# In[ ]:




