#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#加载词向量
import jieba
import gensim
import gc
import numpy as np

VEC_PATH = '/home/admin/jupyter/Demo/DataSets/merge_sgns_bigram_char300/merge_sgns_bigram_char300.txt'
TRAIN_PATH='/home/admin/jupyter/Demo/DataSets/oppo_data_ronud2_20181107/data_train.txt'
TEST_PATH='/home/admin/jupyter/Demo/DataSets/oppo_round2_test_B/oppo_round2_test_B.txt'
VAL_PATH='/home/admin/jupyter/Demo/DataSets/oppo_data_ronud2_20181107/data_vali.txt'


model= gensim.models.KeyedVectors.load_word2vec_format(VEC_PATH,binary=False,unicode_errors='ignore')
keys=model.vocab.keys()
jieba.load_userdict(keys)
print(len(keys))
del keys
del model
gc.collect()


# In[ ]:


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


# In[ ]:


#解析query里面出现次数最多的动名词 ，并返回他出现的次数
import re
from collections import Counter

def  get_query_kwd(query):
    s=str(query)
    s=re.sub('\d+',' ',s)
    kwds=[i for i in  jieba.cut(s)  if len(i)>1]
    word_counts = Counter(kwds)
    top_3 = word_counts.most_common(3)

    return top_3
    

data['query_kwd_list']=data.query_prediction.map(lambda x:get_query_kwd(x))
data['kwds_num']=data.query_kwd_list.map(lambda x:len(x))
data['kwds_1_word_ca']=data.query_kwd_list.map(lambda x:x[0][0] if len(x)>0 else 'nan')
data['kwds_1_word_count_num']=data.query_kwd_list.map(lambda x:x[0][1] if len(x)>0 else 0)
data['kwds_2_word_ca']=data.query_kwd_list.map(lambda x:x[1][0] if len(x)>1 else 'nan')
data['kwds_2_word_count_num']=data.query_kwd_list.map(lambda x:x[1][1] if len(x)>1 else 0)
data['kwds_3_word_ca']=data.query_kwd_list.map(lambda x:x[2][0] if len(x)>2 else 'nan')
data['kwds_3_word_count_num']=data.query_kwd_list.map(lambda x:x[2][1] if len(x)>2 else 0)
data['kwds_1_word_in_title_ca']=data.apply(lambda x:1 if x.kwds_1_word_ca  in x.title else 0,                                          axis=1)
data['kwds_2_word_in_title_ca']=data.apply(lambda x:1 if x.kwds_2_word_ca  in x.title else 0,                                          axis=1)

data['kwds_3_word_in_title_ca']=data.apply(lambda x:1 if x.kwds_3_word_ca  in x.title else 0,                                          axis=1)




# In[ ]:



items = ['tag','prefix','query_prediction','title']

for item in items:
    temp = data.groupby(item).agg('size').reset_index().rename(columns={0:'%s_count_num'%item})
    data = pd.merge(data, temp, on=item, how='left')
    data['%s_count_rate_num'%item]=data['%s_count_num'%item]/len(data)


# In[ ]:


def feature_count(data,col1,col2):
    if col1+'_'+col2 in data.columns:
        dd = 5
    else :
        data[col1+'_'+col2] = data[col1] + data[col2]
        tf1=data.groupby([col1+'_'+col2],as_index=False)[col1+'_'+col2].agg({col1+'_'+col2+'_counts':'count'})
        data=pd.merge(data,tf1,how='left',on=[col1+'_'+col2])
    if col1+'_counts' in data.columns:
        dd = 5
    else:
        tf2=data.groupby([col1],as_index=False)[col1].agg({col1+'_counts':'count'})
        data=pd.merge(data,tf2,how='left',on=[col1])    
    if col2+'_counts' in data.columns:
        dd = 5
    else:
        tf3=data.groupby([col2],as_index=False)[col2].agg({col2+'_counts':'count'})
        data=pd.merge(data,tf3,how='left',on=[col2])
    return data 
ratio_feature_list = []
for i in ['prefix']:  
    for j in ['title','tag','query_prediction']:
        data = feature_count(data,i,j)
        data['ratio_'+j+'_of_'+i] = data[i+'_'+j+'_counts'] / data[i+'_counts']
        data['ratio_'+i+'_of_'+j] = data[i+'_'+j+'_counts'] / data[j+'_counts']
        ratio_feature_list.append('ratio_'+j+'_of_'+i)
        ratio_feature_list.append('ratio_'+i+'_of_'+j)   
for i in ['title']:  
    for j in ['prefix','tag','query_prediction']:
        data = feature_count(data,i,j)
        data['ratio_'+j+'_of_'+i] = data[i+'_'+j+'_counts'] / data[i+'_counts']
        data['ratio_'+i+'_of_'+j] = data[i+'_'+j+'_counts'] / data[j+'_counts']
        ratio_feature_list.append('ratio_'+j+'_of_'+i)
        ratio_feature_list.append('ratio_'+i+'_of_'+j) 
        
for i in ['tag']:  
    for j in ['prefix','query_prediction','title']:
        data = feature_count(data,i,j)
        data['ratio_'+j+'_of_'+i] = data[i+'_'+j+'_counts'] / data[i+'_counts']
        data['ratio_'+i+'_of_'+j] = data[i+'_'+j+'_counts'] / data[j+'_counts']
        ratio_feature_list.append('ratio_'+j+'_of_'+i)
        ratio_feature_list.append('ratio_'+i+'_of_'+j) 
        
        
for i in ['query_prediction']:  
    for j in ['prefix','tag','title']:
        data = feature_count(data,i,j)
        data['ratio_'+j+'_of_'+i] = data[i+'_'+j+'_counts'] / data[i+'_counts']
        data['ratio_'+i+'_of_'+j] = data[i+'_'+j+'_counts'] / data[j+'_counts']
        ratio_feature_list.append('ratio_'+j+'_of_'+i)
        ratio_feature_list.append('ratio_'+i+'_of_'+j) 
        
print("feature  count 特征提取完毕")


# In[ ]:


temp1=data.groupby(['prefix','title']).agg('size').reset_index().rename(columns={                                                                    0:'pt_count'})

temp2=data.groupby(['prefix']).agg('size').reset_index().rename(columns={                                                                    0:'prefix_count'})
temp=pd.merge(temp1,temp2,how='left',on=['prefix'])
temp['t_in_p_bgl_rate_num']=(temp['pt_count'])/(temp['prefix_count'])
data=pd.merge(data,temp,how='left',on=['prefix','title'])
gc.collect()


# In[ ]:


temp1=data.groupby(['prefix','title','tag']).agg('size').reset_index().rename(columns={                                                                    0:'ptg_count_num'})

temp2=data.groupby(['prefix']).agg('size').reset_index().rename(columns={                                                                    0:'prefix_count'})
temp=pd.merge(temp1,temp2,how='left',on=['prefix'])
temp['tg_in_p_bgl_rate_num']=(temp['ptg_count_num'])/(temp['prefix_count'])
data=pd.merge(data,temp,how='left',on=['prefix','title','tag'])
gc.collect()


# In[ ]:


temp1=data.groupby(['prefix','tag']).agg('size').reset_index().rename(columns={                                                                    0:'pg_count'})

temp2=data.groupby(['prefix']).agg('size').reset_index().rename(columns={                                                                    0:'prefix_count'})
temp=pd.merge(temp1,temp2,how='left',on=['prefix'])
temp['g_in_p_bgl_rate_num']=(temp['pg_count'])/(temp['prefix_count'])
data=pd.merge(data,temp,how='left',on=['prefix','tag'])
gc.collect()


# In[ ]:


temp1=data.groupby(['prefix','query_prediction','tag','title']).agg('size').reset_index().rename(columns={                                                                    0:'pqtt_count'})

temp2=data.groupby(['prefix']).agg('size').reset_index().rename(columns={                                                                    0:'prefix_count'})
temp=pd.merge(temp1,temp2,how='left',on=['prefix'])
temp['qtt_in_p_bgl_rate_num']=(temp['pqtt_count'])/(temp['prefix_count'])
data=pd.merge(data,temp,how='left',on=['prefix','query_prediction','tag','title'])
gc.collect()


# In[ ]:


end_time=time.localtime()
str_stime=strftime("%Y_%m_%d %H:%M:%S",end_time)
print(str_stime)

feature=[i for i in data.columns if '_num' in i  or'ratio' in i  or '_ca' in i]


data.sort_values(by='index',inplace=True)
data[feature].to_csv("./CSV/data_count_fea.csv",index=False, compression='bz2')


# In[ ]:




