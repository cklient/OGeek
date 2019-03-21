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





model= gensim.models.KeyedVectors.load_word2vec_format(VEC_PATH,binary=False,unicode_errors='ignore')
keys=model.vocab.keys()
jieba.load_userdict(keys)
print(len(keys))
del keys
gc.collect()


# In[2]:


print(model.similarity('陌陌','app'))
print(model.similarity('京东','app'))
print(model.similarity('陌陌','应用'))
print(model.similarity('京东','应用'))
print(model.similarity('陌陌','百科'))
print(model.similarity('京东','百科'))


# In[3]:


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


# In[4]:


data.tag.value_counts()


# In[ ]:


all_tags=list(set(data.tag.tolist()))
print(all_tags)
all_tags.remove('快应用')
all_tags.remove('应用')
all_tags.remove("影视")
all_tags.remove('阅读')
all_tags.remove('网页')

all_tags=['app','电影','电视剧','小说']+all_tags
all_tags=list(set(all_tags))
print(all_tags)


# In[ ]:


get_ipython().run_cell_magic('time', '', 'def cut_wd(x):\n    s=[ i for i in jieba.cut(x)]\n    return s\n\n\ndata["prefix"]=data.prefix.map(lambda x:cut_wd(x))\ndata[\'title\']=data.title.map(lambda x:cut_wd(x))\n# data[\'tag\']=data.tag.map(lambda x:cut_wd(x))\nprint("分词结束",len(data))\ngc.collect()')


# In[ ]:


import tqdm

def get_max_sm_distance_num(title_cut,tag):
    dist=[0]
    for t in  title_cut:
            
        try:
            d=model.similarity(t,tag)
        except:
            d=0
        dist.append(d)
    return max(dist)


for  sub_tag in  tqdm.tqdm(all_tags):
    data['title_dist_%s_num'%sub_tag]=data.title.map(lambda x: get_max_sm_distance_num(x,sub_tag))


# In[ ]:


data.columns


# In[ ]:


data.head()


# In[ ]:


data['title_wds_num']=data.title.map(lambda x:len(x))


# In[ ]:


end_time=time.localtime()
str_stime=strftime("%Y_%m_%d %H:%M:%S",end_time)
print(str_stime)

feature=[i for i in data.columns if '_num' in i  or'ratio' in i  or '_ca' in i]


data.sort_values(by='index',inplace=True)
data[feature].to_csv("./CSV/data_t2q_fea.csv",index=False, compression='bz2')


# In[ ]:




