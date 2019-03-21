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


# model= gensim.models.KeyedVectors.load_word2vec_format(VEC_PATH,binary=False,unicode_errors='ignore')
# keys=model.vocab.keys()
# jieba.load_userdict(keys)
# print(len(keys))
# del keys
# del model
# gc.collect()


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


# In[4]:


data['prefix']=data.prefix.map(lambda x:x.lower())
data['query_prediction']=data.query_prediction.map(lambda x:x.lower())
data['title']=data.title.map(lambda x:x.lower())


# In[5]:


#
features=['./CSV/data_count_fea.csv','./CSV/data_len_fea.csv','./CSV/data_strsma_fea.csv'        './CSV/data_unique_fea.csv','./CSV/data_w2v_fea.csv']

count_feature = pd.read_csv('./CSV/data_count_fea.csv', compression='bz2')
len_feature = pd.read_csv('./CSV/data_len_fea.csv', compression='bz2')
strsma_feature = pd.read_csv('./CSV/data_strsma_fea.csv', compression='bz2')
unique_feature = pd.read_csv('./CSV/data_unique_fea.csv', compression='bz2')
w2v_feature1 = pd.read_csv('./CSV/data_w2v_fea.csv', compression='bz2')
t_2_q_fea=pd.read_csv('./CSV/data_t2q_fea.csv',compression='bz2')

for df in [data, count_feature, len_feature, strsma_feature, unique_feature,           w2v_feature1,t_2_q_fea]:
    df.reset_index(inplace=True, drop=True)


data = pd.concat([data, count_feature, len_feature, strsma_feature, unique_feature,w2v_feature1,t_2_q_fea], axis=1)



del count_feature
del len_feature
del strsma_feature
del w2v_feature1
del t_2_q_fea

gc.collect()


# In[6]:


data['prefix_len']=data.prefix.map(lambda x:len(str(x)))
data['title_len']=data.title.map(lambda x:len(str(x)))
data['query_len']=data.query_prediction.map(lambda x:len(str(x)))
data['prefix_diff_title_len']=data.apply(lambda x:abs(x.title_len-x.prefix_len),axis=1)


# In[7]:


import tqdm
origin_cate_list=['prefix','query_prediction', 'title', 'tag']
ca_fea=origin_cate_list+[i for i in  data.columns if '_ca' in i ]



for i in tqdm.tqdm(ca_fea):

    data[i] = data[i].map(dict(zip(data[i].unique(), range(0, data[i].nunique()))))
    


# In[8]:


end_time=time.localtime()
str_stime=strftime("%Y_%m_%d %H:%M:%S",end_time)
print(str_stime)

train_data=data[:2000000]# 
val_data=data[2000000:2000000+50000]
test_data=data[2000000+50000:]
print(len(train_data),len(val_data),len(test_data))
del data
gc.collect()


# In[9]:


train_data.sort_values(by='index',inplace=True)
val_data.sort_values(by='index',inplace=True)
test_data.sort_values(by='index',inplace=True)


# In[10]:


def get_data_ctr_fea(tj_data,self_data):

    items = ['prefix', 'title', 'tag','query_prediction',['title', 'tag'],['prefix','tag'],
             ['prefix','title','tag']]
    len_items=['prefix_len','query_len','title_len','prefix_diff_title_len']
    
    oth_items=['max_title_ca',['max_title_ca','tag'],              ['max_title_ca','title','tag'],'kwds_1_word_ca','kwds_2_word_ca',              'kwds_3_word_ca',['kwds_1_word_ca','title'],              ['kwds_1_word_ca','tag']]
    
    items=items+len_items+oth_items
    tj_drop_columns=[i for i in  tj_data  if "_ctr" in i]
    if len(tj_drop_columns)>0:
        tj_data=tj_data.drop(columns=tj_drop_columns)
    
    self_drop_columns=[i for i in  self_data  if "_ctr" in i]
    if len(self_drop_columns)>0:
        print(self_drop_columns)
        self_data.drop(columns=self_drop_columns,inplace=True)
        
    
    for item in items:
        if type(item)==list:
            pr_name="_".join(item)
            merge_columns=item+[pr_name+'_ctr']
        else:
            pr_name=item
            merge_columns=[item,pr_name+'_ctr']
        temp = tj_data.groupby(item, as_index = False)['label'].agg({pr_name+'_click':'sum',pr_name+'_count':'count'})
        temp[pr_name+'_ctr'] =1000* (temp[pr_name+'_click']+3)/(temp[pr_name+'_count']+9)
        
        self_data = pd.merge(self_data, temp[merge_columns], on=item, how='left')
        
#     self_data['dd_ctr']=self_data.apply(lambda x:x.prefix_ctr*x.title_ctr*x.tag_ctr,axis=1)
    
    return self_data
            
test_data=get_data_ctr_fea(train_data,test_data)    
val_data=get_data_ctr_fea(train_data,val_data)    


# In[11]:


from scipy import sparse

origin_cate_list=['prefix','title','tag','query_prediction']

num_feature=[i for i in  test_data.columns if '_num' in i or 'ctr' in i  or'ratio' in  i]
ca_fea=origin_cate_list+[i for i in  test_data.columns if '_ca' in i ]
num_feature+=ca_fea

feature=num_feature
print(feature,len(feature))


# In[12]:


train_data.reset_index(inplace=True,drop=True)


# In[14]:


import gc


gc.collect()

import lightgbm as lgb
from sklearn.model_selection import  StratifiedKFold
from sklearn.metrics import  f1_score
import numpy as np
import logging


from sklearn.metrics import f1_score


logging.basicConfig(filename='lgb_log.log', level=logging.INFO)

THR_PROB=0.4

def lgb_f1_score(y_hat, data):
    y_true = data.get_label()
    y_hat = (y_hat>THR_PROB).astype(int)
    return 'f1', f1_score(y_true, y_hat), True

NFOLD=5

test_x=test_data[feature].values

print("test_x ",test_x.shape)

oof_train=np.zeros((train_data.shape[0],1))
oof_val=np.zeros((val_data.shape[0],1))
oof_val_skf=np.zeros((NFOLD,val_data.shape[0],1))
oof_test=np.zeros((test_data.shape[0],1))
oof_test_skf=np.zeros((NFOLD,test_data.shape[0],1))


cv_f1_csv=[]
val_f1_csv=[]


skf = StratifiedKFold(n_splits=NFOLD, random_state=2018, shuffle=True)

params={"boosting":'gbdt','objective':'binary','max_depth':-1,
            'lambda_l1':0,'lambda_l2':1,
#             
            'num_leaves':31,'max_bin':250,'min_data_in_leaf': 200,'learning_rate': 0.02,'feature_fraction': 0.9,
            
            'bagging_fraction': 0.8,'bagging_freq': 1,'nthread':-1}


##k折增量训练！！！！单模型
# 第一步，初始化模型为None，设置模型参数

gbm=None



for k, (train_in, test_in) in enumerate(skf.split(train_data.index.values.reshape(-1,1), train_data.label.values)):

    train_df=train_data[train_data.index.isin(train_in)]
    val_df=train_data[train_data.index.isin(test_in)]

    train_df=get_data_ctr_fea(val_df,train_df)
    val_df=get_data_ctr_fea(train_df,val_df)


        
    
    train_x=train_df[feature].values
    train_y=train_df.label.values
    
    print("train_x",train_x.shape)
    
    val_x=val_df[feature].values
    val_y=val_df.label.values
    
    print("cv_val_x",val_x.shape)
    
    val_data_x=val_data[feature].values
    val_data_y=val_data.label.values
    
    print("val_data_x",val_data_x.shape)


    lgb_train = lgb.Dataset(train_x, train_y,feature_name=feature)
    lgb_eval = lgb.Dataset(val_x,val_y)
    gbm = lgb.train(params,
                    lgb_train,
                    num_boost_round=50000,
                    valid_sets=(lgb_train,lgb_eval),
                    init_model=gbm,keep_training_booster=True,
                    early_stopping_rounds=200,verbose_eval=100
                    )
    PROB_THR=0.40
    cv_predict=gbm.predict(val_x, num_iteration=gbm.best_iteration).reshape(-1,1)
    oof_train[test_in]=cv_predict
    
    f1=f1_score(val_y, (cv_predict>PROB_THR).astype(int).reshape(-1))
    print("cv f1 %s"%f1)
    logging.info("第 %d折 cv f1 is %s"%(k,f1))
    cv_f1_csv.append(f1)
    val_data_predict=gbm.predict(val_data_x, num_iteration=gbm.best_iteration).reshape(-1,1)
    f1=f1_score(val_data_y, (val_data_predict>PROB_THR).astype(int).reshape(-1))
    
    print("val f1 %s"%f1)
    
    logging.info("第 %d折 val f1 is %s"%(k,f1))
    val_f1_csv.append(f1)

    gc.collect()

print('mean last f1 score:',f1)


# In[15]:


result_val=gbm.predict(val_data_x)
result_test=gbm.predict(test_x)
val_data = pd.read_table(VAL_PATH,quoting=3 ,
        names = ['prefix','query_prediction','title','tag','label'], header = None, encoding='utf-8').astype(str)


test_data = pd.read_table(TEST_PATH,quoting=3 ,
        names = ['prefix','query_prediction','title','tag'],header = None, encoding='utf-8').astype(str)

val_data['predict_proba']=result_val.reshape(-1)
test_data['predict_proba']=result_test.reshape(-1)


val_data.to_csv("val_bed_case_f1.csv",index=False)
test_data.to_csv("test_data_prob_f1.csv",index=False)

test_data['label']=(result_test> PROB_THR).astype(int).reshape(-1)
test_data['label'].astype(int).to_csv("submit_%f.csv"%val_f1_csv[-1],index=False)


# In[16]:


test_data['label'].value_counts()


# In[ ]:


# oof_val=gbm.predict(val_data_x)
# oof_test=gbm.predict(test_x)


# train_data = pd.read_table(TRAIN_PATH,quoting=3,
#         names= ['prefix','query_prediction','title','tag','label'], header= None, encoding='utf-8').astype(str)
# train_data['label'] = train_data['label'].apply(lambda x: int(x))

# val_data = pd.read_table(VAL_PATH,quoting=3 ,
#         names = ['prefix','query_prediction','title','tag','label'], header = None, encoding='utf-8').astype(str)

# val_data['label'] = val_data['label'].apply(lambda x: int(x))


# test_data = pd.read_table(TEST_PATH,quoting=3 ,
#         names = ['prefix','query_prediction','title','tag'],header = None, encoding='utf-8').astype(str)



# #不能使用data.index会有问题 当tain和val合并的时候
# test_data['index']=[i for i in range(len(test_data))]


# train_data['all_text']=train_data['prefix']+train_data['title']+train_data['tag']
# val_data['all_text']=val_data['prefix']+val_data['title']+val_data['tag']

# test_data['all_text']=test_data['prefix']+test_data['title']+test_data['tag']


# print(len(train_data),len(val_data),len(test_data))


# In[ ]:


# val_data['model_predict_prob']=oof_val.reshape(-1)
# test_data['model_predict_prob']=oof_test.reshape(-1)


# In[ ]:


# temp=train_data.groupby('all_text')['label'].agg({'text_ctr':'mean'}).reset_index()
# val_data=pd.merge(val_data,temp,how='left',on='all_text')
# test_data=pd.merge(test_data,temp,how='left',on='all_text')


# In[ ]:


# val_data.info()


# In[ ]:


# test_data.info()


# In[ ]:


# val_data1=val_data[val_data.text_ctr.notna()]
# val_data1['pre_label']=val_data1.text_ctr.map(lambda  x: 1  if x>0.378  else 0)

# test_data1=test_data[test_data.text_ctr.notna()]
# test_data1['pre_label']=test_data1.text_ctr.map(lambda  x: 1  if x>0.378  else 0)

# print(f1_score(val_data1.label.values, val_data1.pre_label.values))
# print(len(test_data1))


# In[ ]:


# val_data2=val_data[val_data.text_ctr.isna()]
# val_data2['pre_label']=val_data2.model_predict_prob.map(lambda  x: 1  if x>0.410 else 0)

# test_data2=test_data[test_data.text_ctr.isna()]
# test_data2['pre_label']=test_data2.model_predict_prob.map(lambda  x: 1  if x>0.410 else 0)


# print(f1_score(val_data2.label.values, val_data2.pre_label.values))
# print(len(test_data2))


# In[ ]:


# val_data_m=pd.concat([val_data1,val_data2])
# val_data_f1=f1_score(val_data_m.label.values, val_data_m.pre_label.values)
# print("val data f1 is %s"%val_data_f1)
# test_data_m=pd.concat([test_data1,test_data2])
# test_data_m.sort_values(by='index',inplace=True)
# test_data_m['pre_label'].astype(int).to_csv("submit_%s.csv"%val_data_f1,index=False)


# In[ ]:




