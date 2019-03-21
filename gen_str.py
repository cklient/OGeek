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


def extract_value(pred):
    try:
        pred = eval(pred)
    except:
        return []
    pred = sorted(pred.items(), key=lambda x: x[1], reverse=True)
    pred_prob_lst=[]
    for i in range(10):
        if len(pred)<i+1:
#             pred_prob_lst.append(0)
            break
        else:
            pred_prob_lst.append(float(pred[i][1]))
    return pred_prob_lst

# def parse_mutlti_task(data):
#     data['q_key_list']=data.query_prediction.map(lambda x:extract_key(x))
#     data['q_value_list']=data.query_prediction.map(lambda x:extract_value(x))
#     return data
    
# data=apply_mul_core(data,parse_mutlti_task)
data['q_key_list']=data.query_prediction.map(lambda x:extract_key(x))
data['q_value_list']=data.query_prediction.map(lambda x:extract_value(x))
print("query 解析完毕")

gc.collect()


# In[4]:


import numpy as np


data['q_value_max_num']=data.q_value_list.map(lambda x:max(x) if len(x)>0 else np.nan)

data['q_value_mean_num']=data.q_value_list.map(lambda x:sum(x)/len(x) if len(x)>0 else np.nan)

data['q_value_sum_num']=data.q_value_list.map(lambda x:sum(x) if len(x)>0 else np.nan)

gc.collect()


# In[5]:


import Levenshtein



def levenshtein_similarity(str1,str2):
    return Levenshtein.ratio(str1,str2)



def get_equal_rate(ss,q_key_list):
    if len(q_key_list)<1:
        return []
    equal_rate=[]
    for i in q_key_list:
        equal_rate.append(levenshtein_similarity(ss,i))
    return equal_rate


def get_weight_equal_rate(equal_rate_list,q_value_list):
    if len(equal_rate_list)<1:
        return []
    s=np.array(equal_rate_list)*np.array(q_value_list)
    return s

data['t_q_sm_rate_list']=data.apply(lambda x:get_equal_rate(x.title,x.q_key_list),axis=1)
data['t_q_sm_weight_rate_list']=data.apply(lambda x:get_weight_equal_rate(x.t_q_sm_rate_list,x.q_value_list),axis=1)
                          
                          
data['p_t_sm_rate_num']=data.apply(lambda x :levenshtein_similarity(x.prefix,x.title),axis=1)

data['sum_t_q_sm_rate_list_num']=data.t_q_sm_rate_list.map(lambda x:sum(x) if len(x)>0 else np.nan)
data['mean_t_q_sm_rate_list_num']=data.t_q_sm_rate_list.map(lambda x:sum(x)/len(x) if len(x)>0 else np.nan)
                          
data['sum_t_q_sm_weight_rate_list_num']=data.t_q_sm_weight_rate_list.map(lambda x:sum(x) if len(x)>0 else np.nan)
data['mean_t_q_sm_weight_rate_list_num']=data.t_q_sm_weight_rate_list.map(lambda x:sum(x)/len(x) if len(x)>0 else np.nan)

print("levenshtein_similarity计算完毕")
gc.collect()


# In[6]:


import numpy as np

def add_pred_similarity_feat(data):
    for i in range(10):
        data['max_t_q_sm_rate'+str(i)+"_num"]=data.t_q_sm_rate_list.apply(lambda x:float(x[i])                                                                           if len(x)>i else np.nan)
        data['max_t_q_sm_weight_rate' + str(i)+"_num"] = data.t_q_sm_weight_rate_list. apply(lambda x:float(x[i]) if len(x)>i else np.nan)
    return data
data=add_pred_similarity_feat(data)


# In[7]:


end_time=time.localtime()
str_stime=strftime("%Y_%m_%d %H:%M:%S",end_time)
print(str_stime)
feature=[i for i in data.columns if '_num' in i  or'ratio' in i  or '_ca' in i]


data.sort_values(by='index',inplace=True)
data[feature].to_csv("./CSV/data_strsma_fea.csv",index=False, compression='bz2')


# In[ ]:




