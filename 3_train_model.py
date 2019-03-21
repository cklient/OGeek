#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#加载词向量
# import jieba
# import gensim
import gc
import numpy as np
TRAIN_PATH='/home/admin/jupyter/Demo/DataSets/oppo_data_ronud2_20181107/data_train.txt'
#TEST_PATH='../Demo/DataSets/oppo_data_ronud2_20181107/data_test.txt'
VAL_PATH='/home/admin/jupyter/Demo/DataSets/oppo_data_ronud2_20181107/data_vali.txt'
TEST_PATH='/home/admin/jupyter/Demo/DataSets/oppo_round2_test_B/oppo_round2_test_B.txt'


# In[ ]:


import warnings
warnings.filterwarnings("ignore")
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
data['index']=[i for i in range(len(data))]
data=pd.concat([data,test_data])
print(len(data))

del train_data
del val_data
del test_data

gc.collect()


# In[ ]:


data['prefix_len']=data.prefix.map(lambda x:len(str(x)))
data['title_len']=data.title.map(lambda x:len(str(x)))
data['query_len']=data.query_prediction.map(lambda x:len(str(x)))
data['prefix_diff_title_len']=data.apply(lambda x:abs(x.title_len-x.prefix_len),axis=1)


# In[ ]:


import tqdm
origin_cate_list=['prefix', 'title', 'tag']
#ca_fea=origin_cate_list+[i for i in  data.columns if '_ca' in i ]

for i in tqdm.tqdm(origin_cate_list):
    data[i+"_label"] = data[i].map(dict(zip(data[i].unique(), range(0, data[i].nunique()))))


# In[ ]:


count_feature = pd.read_csv('../dy/CSV/data_count_fea.csv', compression='bz2')
len_feature = pd.read_csv('../dy/CSV/data_len_fea.csv', compression='bz2')
strsma_feature = pd.read_csv('../dy/CSV/data_strsma_fea.csv', compression='bz2')
unique_feature = pd.read_csv('../dy/CSV/data_unique_fea.csv', compression='bz2')
w2v_feature1 = pd.read_csv('../dy/CSV/data_w2v_fea.csv', compression='bz2')
t_2_q_fea=pd.read_csv('../dy/CSV/data_t2q_fea.csv',compression='bz2')

for df in [data, count_feature, len_feature, strsma_feature, unique_feature,w2v_feature1,t_2_q_fea]:
    df.reset_index(inplace=True, drop=True)


data = pd.concat([data, count_feature, len_feature, strsma_feature, unique_feature,w2v_feature1,t_2_q_fea], axis=1)



del count_feature
del len_feature
del strsma_feature
del w2v_feature1
del t_2_q_fea

gc.collect()


# In[ ]:


# %%time
# # napianlin
# def downcast_data(D):
#     for c, d in zip(D.columns, D.dtypes):
#         if d.kind == 'f':
#             D[c] = pd.to_numeric(D[c], downcast='float')
#         elif d.kind == 'i':
#             D[c] = pd.to_numeric(D[c], downcast='signed')
#     return D


# data = downcast_data(data)


# In[ ]:


end_time=time.localtime()
str_stime=strftime("%Y_%m_%d %H:%M:%S",end_time)
print(str_stime)

train_data=data[:2000000]# 
val_data=data[2000000:2000000+50000]
test_data=data[2000000+50000:]
print(len(train_data),len(val_data),len(test_data))

train_data.sort_values(by='index',inplace=True)
val_data.sort_values(by='index',inplace=True)
test_data.sort_values(by='index',inplace=True)

del data
gc.collect()


# In[ ]:


def get_data_ctr_fea(tj_data,self_data):

    items = ['prefix', 'title', 'tag','query_prediction',['title', 'tag'],['prefix','tag'],
             ['prefix','title','tag']]
    #len_items=['prefix_len','query_len','title_len','prefix_diff_title_len']
    len_items=['prefix_len','query_len','title_len','prefix_diff_title_len']
    
#     oth_items=['max_title',['max_title','tag'],\
#               ['max_title','title','tag']]
    
    oth_items=['max_title_ca',['max_title_ca','tag'],              ['max_title_ca','title','tag'],'kwds_1_word_ca','kwds_2_word_ca',              'kwds_3_word_ca',['kwds_1_word_ca','title'],              ['kwds_1_word_ca','tag']]
    
    items=items+len_items
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
    
    return self_data
        


# In[ ]:


print(test_data.shape)
test_data=get_data_ctr_fea(train_data,test_data)    
val_data=get_data_ctr_fea(train_data,val_data) 
print(test_data.shape)
print(test_data.columns)


# In[ ]:


origin_cate_list=['prefix_label', 'title_label', 'tag_label']
num_feature=[i for i in  test_data.columns if '_num' in i or 'ctr' in i  or'ratio' in  i]
ca_fea=origin_cate_list+[i for i in  test_data.columns if '_ca' in i ]
num_feature+=ca_fea

feature=num_feature
feature.remove("max_title_ca")
feature.remove("kwds_1_word_ca")
feature.remove("kwds_2_word_ca")
feature.remove("kwds_3_word_ca")
print(feature,len(feature))


# In[ ]:


get_ipython().run_cell_magic('time', '', "from sklearn.model_selection import KFold\nkfold = KFold(n_splits=5,shuffle=False,random_state=1024)\nt=pd.DataFrame()\nfor train_idx,val_idx in kfold.split(train_data):\n    train_X = train_data.iloc[train_idx] \n    val_X = train_data.iloc[val_idx]\n    val=get_data_ctr_fea(train_X,val_X)\n    t=pd.concat([t,val])\nt.sort_values(by='index',inplace=True)\ntrain_data=t\ntrain_data.fillna(-99,inplace=True)\ntest_data.fillna(-99,inplace=True)\nval_data.fillna(-99,inplace=True)")


# In[ ]:


train_feature=train_data[feature].values
val_feature=val_data[feature].values
test_feature=test_data[feature].values
print(train_feature.shape,val_feature.shape,test_feature.shape)

# train_data.drop(train_data.columns[feature], axis=1, inplace=True)
# val_data.drop(val_data.columns[feature], axis=1, inplace=True)
# test_data.drop(test_data.columns[feature], axis=1, inplace=True)
# gc.collect()
# print(train_feature.shape,val_feature.shape,test_feature.shape)


# In[ ]:


train_data.drop(feature, axis=1, inplace=True)
val_data.drop(feature, axis=1, inplace=True)
test_data.drop(feature, axis=1, inplace=True)
gc.collect()
print(train_data.shape,val_data.shape,test_data.shape)


# In[ ]:


from keras.layers import Dense, Input, LSTM, Embedding, Dropout,CuDNNLSTM,Bidirectional
from keras.layers.core import Lambda
from keras.layers.merge import concatenate, add, multiply
from keras.models import Model
from keras.layers.normalization import BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers.noise import GaussianNoise

def square_diff(sequence_1_input,sequence_2_input):
    Max_num = 200000
    embedding_layer = Embedding(Max_num,
                                embedding_matrix.shape[1],
                                weights=[embedding_matrix],
                                #input_length=MAX_SEQUENCE_LENGTH,
                                trainable=False)
    #lstm_layer = LSTM(75, recurrent_dropout=0.2)
    lstm_layer = Bidirectional(CuDNNLSTM(40))
    
    embedded_sequences_1 = embedding_layer(sequence_1_input)
    x1 = lstm_layer(embedded_sequences_1)
    
    embedded_sequences_2 = embedding_layer(sequence_2_input)
    y1 = lstm_layer(embedded_sequences_2)
    
    addition = add([x1, y1])
    minus_y1 = Lambda(lambda x: -x)(y1)
    merged = add([x1, minus_y1])
    merged = multiply([merged, merged])
    merged = concatenate([merged, addition])
    merged = Dropout(0.4)(merged)
    
    return merged
    

def build_model2():
    Max_num = 200000
    Q_length = 10
    A_length = 35
    MAX_SEQUENCE_LENGTH=10
   

    seq1 = Input(shape=(Q_length,))
    seq2 = Input(shape=(A_length,))
    
    p_a = []
    diff1 = square_diff(seq1,seq2)
    p_a.append(diff1)
    
    p_input = []
    for i in range(10):
        seq_p = Input(shape=(Q_length,))
        p_input.append(seq_p)
        
        merged_a=square_diff(seq_p,seq2)
        p_a.append(merged_a)
        
    merged = concatenate(p_a)
    merged = Dropout(0.4)(merged)
    
    features_input = Input(shape=(180,), dtype="float32")
    features_dense = BatchNormalization()(features_input)
    features_dense = Dense(300, activation="relu")(features_dense)
    features_dense = Dropout(0.2)(features_dense)

    
    merged = concatenate([merged, features_dense])
    merged = BatchNormalization()(merged)
    merged = GaussianNoise(0.1)(merged)

    merged = Dense(150, activation="relu")(merged)
    merged = Dropout(0.2)(merged)
    merged = BatchNormalization()(merged)

    out = Dense(1, activation="sigmoid")(merged)
    
    model= Model(inputs=[seq1, seq2,features_input]+p_input, outputs=out)

    model.compile(loss="binary_crossentropy",
                  optimizer="nadam", metrics=['accuracy'])
    #model.summary()
    return model
# model=build_model2()


# In[ ]:


get_ipython().run_cell_magic('time', '', 'ques=np.load("feature/train_ques.npy")\nans=np.load("feature/train_ans.npy")\nembedding_matrix=np.load("feature/embedding_matrix.npy")\nprediction_list=pd.read_pickle("./feature/train_prediction_list.pkl")\n\nval_ques=np.load("./feature/val_ques.npy")\nval_ans=np.load("./feature/val_ans.npy")\nval_prediction_list=pd.read_pickle("./feature/val_prediction_list.pkl")\nprint(val_ques.shape)\nprint(val_ans.shape)')


# In[ ]:


from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.callbacks import CSVLogger
csv_logger = CSVLogger('log.csv', append=True, separator=';')

early_stopping = EarlyStopping(monitor="val_loss", patience=3)
best_model_path = "model/best_t_model_rcnn_weight.h5"
model_checkpoint = ModelCheckpoint(best_model_path, save_best_only=True, save_weights_only=True)
model=build_model2()
model.fit([ques,ans,train_feature]+prediction_list, train_data["label"].values, epochs=100,
                   batch_size=512,
                   validation_split=0.05,
                   #validation_data=([val_ques,val_ans,val_feature]+val_prediction_list,val_data["label"].values),
                   callbacks=[model_checkpoint,early_stopping],
                  shuffle=True)


# In[ ]:


print("are you ok ?")


# In[ ]:


del ques,ans,prediction_list
gc.collect()


# In[ ]:


import numpy as np
from sklearn import metrics
model=build_model2()
best_model_path = "model/best_t_model_rcnn_weight.h5"
model.load_weights(best_model_path)
val=model.predict([val_ques,val_ans,val_feature]+val_prediction_list,batch_size=512)
for thresh in np.arange(0.25, 0.501, 0.01):
    thresh = np.round(thresh, 2)
    print("F1 score at threshold {0} is {1}".format(thresh, metrics.f1_score(val_data["label"].values, (val>thresh).astype(int))))


# In[ ]:


val_data["pre"]=val
val_data['pre'].to_csv("val_1.csv",index=False)


# In[ ]:



test_ques=np.load("feature/test_ques.npy")
test_ans=np.load("feature/test_ans.npy")

test_prediction_list=pd.read_pickle("./feature/test_prediction_list.pkl")

model.load_weights(best_model_path)
resu=model.predict([test_ques,test_ans,test_feature]+test_prediction_list,batch_size=1024)


# In[ ]:


test_data["pre"]=resu
test_data['pre'].to_csv("test_1.csv",index=False)


# In[ ]:




