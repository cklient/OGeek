#加载词向量
import jieba
import gensim
import gc
import numpy as np

VEC_PATH = '../data/sgns.sogounews.bigram-char'
TRAIN_PATH='../data/data_train.txt'
TEST_PATH='../data/data_test.txt'
VAL_PATH='../data/data_vali.txt'

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

data['prefix']=data.prefix.map(lambda x:x.lower())
data['query_prediction']=data.query_prediction.map(lambda x:x.lower())
data['title']=data.title.map(lambda x:x.lower())

features=['./CSV/data_count_fea.csv','./CSV/data_len_fea.csv','./CSV/data_strsma_fea.csv'\
        './CSV/data_unique_fea.csv','./CSV/data_w2v_fea.csv']
fea_path='./CSV/'
count_feature = pd.read_csv(fea_path+'data_count_fea.csv', compression='bz2')
len_feature = pd.read_csv(fea_path+'data_len_fea.csv', compression='bz2')
strsma_feature = pd.read_csv(fea_path+'data_strsma_fea.csv', compression='bz2')
unique_feature = pd.read_csv(fea_path+'data_unique_fea.csv', compression='bz2')
w2v_feature1 = pd.read_csv(fea_path+'data_w2v_fea.csv', compression='bz2')
t_2_q_fea=pd.read_csv(fea_path+'data_t2q_fea.csv',compression='bz2')

for df in [data, count_feature, len_feature, strsma_feature, unique_feature, \
          w2v_feature1,t_2_q_fea]:
    df.reset_index(inplace=True, drop=True)


data = pd.concat([data, count_feature, len_feature, strsma_feature, unique_feature,w2v_feature1,t_2_q_fea], axis=1)



del count_feature
del len_feature
del strsma_feature
del w2v_feature1
del t_2_q_fea

gc.collect()

num_list=[]

data['prefix_len']=data.prefix.map(lambda x:len(str(x)))
data['title_len']=data.title.map(lambda x:len(str(x)))
data['query_len']=data.query_prediction.map(lambda x:len(str(x)))
data['prefix_diff_title_len']=data.apply(lambda x:abs(x.title_len-x.prefix_len),axis=1)

import tqdm
origin_cate_list=['prefix','query_prediction', 'title', 'tag']
ca_fea=origin_cate_list+[i for i in  data.columns if '_ca' in i ]



for i in tqdm.tqdm(ca_fea):

    data[i] = data[i].map(dict(zip(data[i].unique(), range(0, data[i].nunique()))))
	
	
end_time=time.localtime()
str_stime=strftime("%Y_%m_%d %H:%M:%S",end_time)
print(str_stime)

train_data=data[:2000000]# 
val_data=data[2000000:2000000+50000]
test_data=data[2000000+50000:]
print(len(train_data),len(val_data),len(test_data))
del data
gc.collect()

train_data.sort_values(by='index',inplace=True)
val_data.sort_values(by='index',inplace=True)
test_data.sort_values(by='index',inplace=True)

def get_data_ctr_fea(tj_data,self_data):

    items = ['prefix', 'title', 'tag','query_prediction',['title', 'tag'],['prefix','tag'],
             ['prefix','title','tag']]
    len_items=['prefix_len','query_len','title_len','prefix_diff_title_len']
    
    oth_items=['max_title_ca',['max_title_ca','tag'],\
              ['max_title_ca','title','tag'],'kwds_1_word_ca','kwds_2_word_ca',\
              'kwds_3_word_ca',['kwds_1_word_ca','title'],\
              ['kwds_1_word_ca','tag']]
    
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


from scipy import sparse

origin_cate_list=['prefix','title','tag','query_prediction']

num_feature=[i for i in  test_data.columns if 'score_' in i or'_num' in i or 'ctr' in i  or'ratio' in  i]
ca_fea=origin_cate_list+[i for i in  test_data.columns if '_ca' in i ]
num_feature+=ca_fea

feature=num_feature+num_list
print(feature,len(feature))

from sklearn.metrics import auc,roc_curve,mean_squared_error
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
import xgboost as xgb      
from sklearn.model_selection import KFold,StratifiedKFold
import lightgbm as lgb
from sklearn.metrics import auc,roc_curve,log_loss
from sklearn.metrics import roc_auc_score,precision_score,recall_score
from sklearn.model_selection import cross_val_score
import warnings
from hyperopt import fmin, tpe,anneal, hp,space_eval,rand,Trials,partial,STATUS_OK
import random
from sklearn.metrics  import log_loss
import numpy as np
import logging


from sklearn.metrics import f1_score

NFOLD=5

test_x=test_data[feature].values

print("test_x ",test_x.shape)

oof_train=np.zeros((train_data.shape[0],1))
oof_val=np.zeros((val_data.shape[0],1))
oof_val_skf=np.zeros((NFOLD,val_data.shape[0],1))
oof_test=np.zeros((test_data.shape[0],1))
oof_test_skf=np.zeros((NFOLD,test_data.shape[0],1))
logging.basicConfig(filename='para_log.log', level=logging.INFO)

count=0
mseforrec=0.7
#sub_preds = np.zeros((test.shape[0],6))
nbpara=0
def GBM(argsDict):
    global count  
    global mseforrec
    global nbpara
    argsDict['n_jobs']=-1
    argsDict['boosting']='gbdt'
    argsDict['objective']='binary'
        
    cv_f1_csv=[]
    val_f1_csv=[]
    skf = StratifiedKFold(n_splits=NFOLD, random_state=2018, shuffle=True)
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
        gbm = lgb.train(argsDict,
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
        print('step:{},fold:{},cv_f1:{}'.format(count,k,f1))
        logging.info('step:{},fold:{},cv_f1:{}'.format(count,k,f1))
        cv_f1_csv.append(f1)
        val_data_predict=gbm.predict(val_data_x, num_iteration=gbm.best_iteration).reshape(-1,1)
        f1=f1_score(val_data_y, (val_data_predict>PROB_THR).astype(int).reshape(-1))
    
        print('step:{},fold:{},val_f1:{}'.format(count,k,f1))
        logging.info('step:{},fold:{},val_f1:{}'.format(count,k,f1))
        val_f1_csv.append(f1)
        
        del train_df
        del val_df
        del train_x
        del val_x
        del val_data_x
        del lgb_train
        del lgb_eval
        gc.collect()
    
    if(val_f1_csv[-1]>mseforrec):
        print('param:',argsDict)
        mseforrec=val_f1_csv[-1]#np.mean(tmprmselist)
        logging.info('param:',argsDict)
        nbpara=argsDict
        print('improved!')
    count+=1
    print(" ")
    logging.info(' ')
    return -1*val_f1_csv[-1]
	
	
def hypersearch(num_boost_round,max_evals=100):

    max_depth=[-1]
    num_boost_round=num_boost_round
    learning_rate=np.linspace(0.01,0.12,num=40)
    feature_fraction=np.linspace(0.6,0.98,num=30)
    bagging_fraction=np.linspace(0.6,0.98,num=30)
    bagging_freq=np.arange(1,12,1)#[0]#
    #min_data_in_leaf=np.arange(1,120,2)
    num_leaves=np.arange(30,100,2)
    reg_lambda       =np.linspace(0.0,20.0,num=40)
    reg_alpha        =np.linspace(0.0,5.0,num=30)
    min_child_weight =np.linspace(0,6,num=30)
    drop_rate        =np.linspace(0.0,0.3,num=20)
    min_split_gain   =np.linspace(0.0,6.0,num=30)
    scale_pos_weight= [1.0]#np.linspace(1.0 ,3.0,num=20)
    threhold=         np.linspace(0.4,0.5,num=40)
    
    max_bin=np.arange(180,300,2)
    rands= np.arange(0,1000,4)
    min_data_in_bin= np.arange(3,15,1)
    space = {"max_depth":        hp.choice("max_depth",max_depth),
             #"num_boost_round":  hp.choice("num_boost_round",num_boost_round),  
             "learning_rate":    hp.choice("learning_rate",learning_rate),  
             "min_split_gain":   hp.choice("min_split_gain",min_split_gain),  
             "feature_fraction": hp.choice("feature_fraction",feature_fraction),
             #"min_data_in_leaf": hp.choice("min_data_in_leaf",min_data_in_leaf), #
             "num_leaves":       hp.choice("num_leaves",num_leaves),
             'bagging_fraction': hp.choice("bagging_fraction",bagging_fraction),
             'bagging_freq'    : hp.choice("bagging_freq",bagging_freq),
             "reg_lambda":       hp.choice("reg_lambda",reg_lambda),
             "reg_alpha":        hp.choice("reg_alpha",reg_alpha),
             "min_child_weight":   hp.choice("min_child_weight",min_child_weight),
             #"drop_rate":        hp.choice("drop_rate",drop_rate),
             #"threhold":        hp.choice("threhold",threhold),
            "scale_pos_weight": hp.choice("scale_pos_weight",scale_pos_weight),
             #"rands": hp.choice("rands",rands),
             "max_bin": hp.choice("max_bin",max_bin),
             "min_data_in_bin": hp.choice("min_data_in_bin",min_data_in_bin), 
        }

    trials = Trials()
    best = fmin(GBM,space,algo=anneal.suggest,#anneal
                max_evals=max_evals
                )
				
hypersearch([5000],max_evals=10000)

argsDict={'bagging_fraction': 0.796551724137931,
 'bagging_freq': 4,
 'feature_fraction': 0.6524137931034483,
 'learning_rate': 0.01564102564102564,
 'max_bin': 188,
 'max_depth': -1,
 'min_child_weight': 0.0,
 'min_data_in_bin': 7,
 'min_split_gain': 4.137931034482759,
 'num_leaves': 90,
 'reg_alpha': 1.7241379310344829,
 'reg_lambda': 14.87179487179487,
 'scale_pos_weight': 1.0,
 'n_jobs': -1,
 'boosting': 'gbdt',
 'objective': 'binary'}
 
 if True:
    skf = StratifiedKFold(n_splits=NFOLD, random_state=2018, shuffle=True)
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
        gbm = lgb.train(argsDict,
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
        print('step:{},fold:{},cv_f1:{}'.format(count,k,f1))
        logging.info('step:{},fold:{},cv_f1:{}'.format(count,k,f1))
        #cv_f1_csv.append(f1)
        val_data_predict=gbm.predict(val_data_x, num_iteration=gbm.best_iteration).reshape(-1,1)
        f1=f1_score(val_data_y, (val_data_predict>PROB_THR).astype(int).reshape(-1))
    
        print('step:{},fold:{},val_f1:{}'.format(count,k,f1))
        logging.info('step:{},fold:{},val_f1:{}'.format(count,k,f1))
        #val_f1_csv.append(f1)
        
        del train_df
        del val_df
        del train_x
        del val_x
        del val_data_x
        del lgb_train
        del lgb_eval
        gc.collect()