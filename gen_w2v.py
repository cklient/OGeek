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


data['tag']=data.tag.map(lambda x:'app' if x in ['应用','快应用'] else x)


# In[4]:


get_ipython().run_cell_magic('time', '', 'def cut_wd(x):\n    s=" ".join(jieba.cut(x))\n    return s\n\n\ndata["prefix"]=data.prefix.map(lambda x:cut_wd(x))\ndata[\'title\']=data.title.map(lambda x:cut_wd(x))\ndata[\'tag\']=data.tag.map(lambda x:cut_wd(x))\nprint("分词结束",len(data))\ngc.collect()')


# In[5]:


get_ipython().run_cell_magic('time', '', '\nimport pandas as pd\nimport numpy as np\nimport gensim\nfrom tqdm import tqdm\nfrom scipy.stats import skew, kurtosis\nfrom scipy.spatial.distance import cosine, cityblock, jaccard, canberra, euclidean, minkowski, braycurtis\n\n\ndef sent2vec(s):\n    words=str(s).split()\n    M=[]\n    for w in words:\n        try:\n            M.append(model[w])\n        except:\n            continue\n    M=np.array(M)\n    v=M.sum(axis=0)\n    return v/np.sqrt((v**2).sum())\n\n\n\ndef get_s2s_dist(s1,s2):\n    s1_v=sent2vec(s1)\n    s2_v=sent2vec(s2)\n    dist=cosine(s1_v,s2_v)\n    return dist\n')


# In[8]:


data['tag']=data.tag.map(lambda x:'app' if x in ['应用','快应用'] else x)

def cut_wd(x):
    s=" ".join(jieba.cut(x))
    return s


data["prefix"]=data.prefix.map(lambda x:cut_wd(x))
data['title']=data.title.map(lambda x:cut_wd(x))
data['tag']=data.tag.map(lambda x:cut_wd(x))
print("分词结束",len(data))
gc.collect()


# In[9]:



data['w2v1_prefix_2_title_distance_num']=data.apply(lambda x:get_s2s_dist(x.prefix,x.title),axis=1)
data['w2v1_tag_2_title_distance_num']=data.apply(lambda x:get_s2s_dist(x.tag,x.title),axis=1)
data['w2v1_prefix_2_tag_distance_num']=data.apply(lambda x:get_s2s_dist(x.prefix,x.tag),axis=1)

print("空间距离计算完毕")

gc.collect()


# In[10]:




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


# In[11]:


def cut_key_list(x):
    if len(x)<1:
        return x
    seg_key_list=[]
    for k in  x:
        s=" ".join(jieba.cut(k))
        seg_key_list.append(s)
        
    return seg_key_list


# def cutkey_mutlti_task(data):
#     data['seg_q_key_list']=data.q_key_list.map(lambda x:cut_key_list(x))
#     return data

# data=apply_mul_core(data,cutkey_mutlti_task)
data['q_key_list']=data.q_key_list.map(lambda x:cut_key_list(x))
print("q_key_list 解析完毕")

gc.collect()


# In[ ]:


def get_distance_list(title,q_key_list):
    if len(q_key_list)<1:
        return []
    w2v_distance_list=[]
    for k in q_key_list:
        d=get_s2s_dist(title,k)
        w2v_distance_list.append(d)

    return w2v_distance_list

# def w2v_dist_list_mutlti_task(data):
data['w2v1_title_q_key_distance_list']=data.apply(lambda x: get_distance_list(x.title,x.q_key_list),axis=1)
data['w2v1_tag_q_key_distance_list']=data.apply(lambda x: get_distance_list(x.tag,x.q_key_list),axis=1)

#     return data
print("title与q_key空间距离计算完毕")
print("tag与q_key空间距离计算完毕")

gc.collect()


# In[ ]:


def get_weight_distance_list(w2v_title_q_key_distance_list,q_value_list):
    if len(w2v_title_q_key_distance_list)<1:
        return []
    s=np.array(w2v_title_q_key_distance_list)*np.array(q_value_list)
    return s


data['w2v1_title_q_key_weight_distance_list']=data.apply(lambda x: get_weight_distance_list(x.w2v1_title_q_key_distance_list,x.q_value_list),axis=1)
print("title与q_key空间权重距离计算完毕")

data['w2v1_tag_q_key_weight_distance_list']=data.apply(lambda x: get_weight_distance_list(x.w2v1_tag_q_key_distance_list,x.q_value_list),axis=1)
print("tag与q_key空间权重距离计算完毕")

gc.collect()


# In[ ]:


def add_pred_similarity_feat(data):
    for i in range(10):
        data['w2v1_title_q_key_weight'+str(i)+"_num"]=data.w2v1_title_q_key_distance_list.apply(lambda x:float(x[i]) if len(x)>i  else np.nan)
        data['w2v1_title_q_key'+str(i)+"_num"]=data.w2v1_title_q_key_weight_distance_list.apply(lambda x:float(x[i]) if len(x)>i  else np.nan)
        data['w2v1_tag_q_key_weight'+str(i)+"_num"]=data.w2v1_tag_q_key_weight_distance_list.apply(lambda x:float(x[i]) if len(x)>i  else np.nan)
        data['w2v1_tag_q_key'+str(i)+"_num"]=data.w2v1_tag_q_key_distance_list.apply(lambda x:float(x[i]) if len(x)>i  else np.nan)

    return data


data=add_pred_similarity_feat(data)


# In[ ]:


data['w2v1_title_key_max_distance_num']=data.w2v1_title_q_key_distance_list.map(lambda x: max(x) if len(x)>0  else np.nan)
data['w2v1_title_key_mean_distance_num']=data.w2v1_title_q_key_distance_list.map(lambda x: sum(x)/len(x) if len(x)>0  else np.nan)
data['w2v1_title_key_sum_distance_num']=data.w2v1_title_q_key_distance_list.map(lambda x: sum(x) if len(x)>0  else np.nan)
gc.collect()


# In[ ]:


data['w2v1_tag_key_max_distance_num']=data.w2v1_tag_q_key_distance_list.map(lambda x: max(x) if len(x)>0  else np.nan)
data['w2v1_tag_key_mean_distance_num']=data.w2v1_tag_q_key_distance_list.map(lambda x: sum(x)/len(x) if len(x)>0  else np.nan)
data['w2v1_tag_key_sum_distance_num']=data.w2v1_tag_q_key_distance_list.map(lambda x: sum(x) if len(x)>0  else np.nan)
gc.collect()


# In[ ]:


data['w2v1_title_key_max_weight_distance_num']=data.w2v1_title_q_key_weight_distance_list.map(lambda x: max(x) if len(x)>0  else np.nan)
data['w2v1_title_key_mean_weight_distance_num']=data.w2v1_title_q_key_weight_distance_list.map(lambda x: sum(x)/len(x) if len(x)>0  else np.nan)
data['w2v1_title_key_sum_weight_distance_num']=data.w2v1_title_q_key_weight_distance_list.map(lambda x: sum(x) if len(x)>0  else np.nan)
gc.collect()


# In[ ]:


data['w2v1_tag_key_max_weight_distance_num']=data.w2v1_tag_q_key_weight_distance_list.map(lambda x: max(x) if len(x)>0  else np.nan)
data['w2v1_tag_key_mean_weight_distance_num']=data.w2v1_tag_q_key_weight_distance_list.map(lambda x: sum(x)/len(x) if len(x)>0  else np.nan)
data['w2v1_tag_key_sum_weight_distance_num']=data.w2v1_tag_q_key_weight_distance_list.map(lambda x: sum(x) if len(x)>0  else np.nan)
gc.collect()


# In[ ]:


end_time=time.localtime()
str_stime=strftime("%Y_%m_%d %H:%M:%S",end_time)
print(str_stime)

feature=[i for i in data.columns if '_num' in i  or'ratio' in i  or '_ca' in i]


data.sort_values(by='index',inplace=True)
data[feature].to_csv("./CSV/data_w2v_fea.csv",index=False, compression='bz2')


# In[ ]:


len(feature)


# In[ ]:




