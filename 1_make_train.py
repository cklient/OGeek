#!/usr/bin/env python
# coding: utf-8

# In[1]:


#加载词向量
import jieba
import gensim
import gc
import numpy as np

#VEC_PATH = '../Demo/DataSets/merge_sgns_bigram_char300/merge_sgns_bigram_char300.txt'
#VEC_PATH='../napianlin/data/300w2vModel.txt'
VEC_PATH='/home/admin/jupyter/Demo/DataSets/sgns.merge.word/sgns.merge.word'
TRAIN_PATH='/home/admin/jupyter/Demo/DataSets/oppo_data_ronud2_20181107/data_train.txt'
#TEST_PATH='../Demo/DataSets/oppo_data_ronud2_20181107/data_test.txt'
VAL_PATH='/home/admin/jupyter/Demo/DataSets/oppo_data_ronud2_20181107/data_vali.txt'
TEST_PATH='/home/admin/jupyter/Demo/DataSets/oppo_round2_test_B/oppo_round2_test_B.txt'


# In[ ]:


from gensim.models import KeyedVectors
#embeddings_index = KeyedVectors.load_word2vec_format(VEC_PATH, binary=False)
embeddings_index = KeyedVectors.load_word2vec_format(VEC_PATH, binary=False, unicode_errors='ignore')
jieba.load_userdict(embeddings_index.vocab.keys())


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


import json

#data["title"]=data["tag"]+data["title"]
data["prefix"]=data["tag"]+data["prefix"]


def extract_key(pred):
    try:
        pred = eval(pred)
    except:
        return [pred]
    pred = sorted(pred.items(), key=lambda x: x[1], reverse=True)
    key_lst=[]
    for i in range(10):
        if len(pred)<i+2:
#             pred_prob_lst.append(0)
            break
        else:
            key_lst.append(pred[i][0])
    return key_lst

def parse_prediction(data):
    data['prediction_texts'] = data.query_prediction.apply(extract_key)
    for i in range(10):
        data['text' + str(i)] = data.prediction_texts.apply(lambda x: x[i] if i < len(x) else "</s>")
    del data["prediction_texts"]
    return data
data=parse_prediction(data)


# In[ ]:


from multiprocessing import Pool
def df_cut_word(data, c):
    data[c] = data[c].map(lambda x: ' '.join(jieba.cut(x)))
    return data[[c]]

processor = 12
list_seg = ["prefix", "title", "text0", "text1", "text2", "text3", "text4", "text5", "text6", "text7", "text8",
                "text9"]
cut_word_args = list_seg
p = Pool(processor)
res = []

for i in range(len(cut_word_args)):
    print(cut_word_args[i])
    res.append(
        p.apply_async(df_cut_word, args=(
                data[[cut_word_args[i]]], cut_word_args[i]))
    )
    print(str(i) + ' processor started !')


p.close()
p.join()

res = [item.get() for item in res]
res = pd.concat(res, axis=1)

print(res.columns)

data = data[[c for c in data.columns if c not in res.columns]]
data = pd.concat([data, res], axis=1)


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


# In[7]:


# %%time
# end_time=time.localtime()
# str_stime=strftime("%Y_%m_%d %H:%M:%S",end_time)
# print(str_stime)

# def seg_sentence(sentence):
#     sentence_seged = jieba.cut(sentence.strip())
#     return " ".join(sentence_seged)

# list_seg = ["prefix", "title", "text0", "text1", "text2", "text3", "text4", "text5", "text6", "text7", "text8",
#                 "text9"]
# for seg in list_seg:
#     train_data[seg + "_list"] = train_data[seg].apply(lambda x: seg_sentence(x))
#     val_data[seg + "_list"] = val_data[seg].apply(lambda x: seg_sentence(x))
#     test_data[seg + "_list"] = test_data[seg].apply(lambda x: seg_sentence(x))


# In[ ]:



words=train_data["prefix"].values.tolist()

list_seg = ["title", "text0", "text1", "text2", "text3", "text4", "text5", "text6", "text7", "text8",
                "text9"]
for seg in list_seg:
    words.extend(train_data[seg].values.tolist())
    words.extend(val_data[seg].values.tolist())
    words.extend(test_data[seg].values.tolist()) 
words.extend(train_data["tag"].values.tolist())
from keras.preprocessing.text import Tokenizer
max_features=200000
tokenizer = Tokenizer(max_features, filters="~\t\n")
#tokenizer = Tokenizer(max_features)
tokenizer.fit_on_texts(words)
print(len(tokenizer.word_index))


# In[ ]:


import numpy as np
word_index = tokenizer.word_index
nb_words = min(max_features, len(word_index))
embedding_matrix = (np.random.rand(nb_words, 300) - 0.5) / 5.0
for word, i in word_index.items():
    if i >= max_features: continue
    if word in embeddings_index:
        embedding_vector = embeddings_index.get_vector(word)
        embedding_matrix[i] = embedding_vector


# In[9]:


#两个词向量拼接
# VEC_PATH='../napianlin/data/300w2vModel.txt'
# embeddings_index_two = KeyedVectors.load_word2vec_format(VEC_PATH, binary=False, unicode_errors='ignore')
# embedding_matrix_two = (np.random.rand(nb_words, 300) - 0.5) / 5.0
# for word, i in word_index.items():
#     if i >= max_features: continue
#     if word in embeddings_index_two:
#         embedding_vector = embeddings_index_two.get_vector(word)
#         embedding_matrix_two[i] = embedding_vector
# embedding_matrix=np.hstack((embedding_matrix,embedding_matrix_two))


# In[ ]:


def get_prediction_list(data):
    list_seg = ["text0", "text1", "text2", "text3", "text4", "text5", "text6", "text7", "text8",
                "text9"]
    prediction_list=[]
    for seg in list_seg:
        prediction_train = tokenizer.texts_to_sequences(data[seg])
        prediction = pad_sequences(prediction_train, maxlen=10,truncating="post")
        prediction_list.append(prediction)
    return prediction_list
from keras.preprocessing.sequence import pad_sequences
def get_que_ans(data):
    p_train = tokenizer.texts_to_sequences(data["prefix"])
    t_train = tokenizer.texts_to_sequences(data["title"])
    ques = pad_sequences(p_train, maxlen=10,truncating="post")
    ans = pad_sequences(t_train, maxlen=35,truncating="post")
    return ques,ans


# In[ ]:


ques,ans=get_que_ans(train_data)
prediction_list=get_prediction_list(train_data)


# In[ ]:


np.save("./feature/train_ques",ques)
np.save("./feature/train_ans",ans)
np.save("./feature/embedding_matrix",embedding_matrix)
pd.to_pickle(prediction_list,"./feature/train_prediction_list.pkl")


# In[ ]:



val_ques,val_ans=get_que_ans(val_data)
val_prediction_list=get_prediction_list(val_data)

np.save("./feature/val_ques",val_ques)
np.save("./feature/val_ans",val_ans)
pd.to_pickle(val_prediction_list,"./feature/val_prediction_list.pkl")


# In[ ]:


test_ques,test_ans=get_que_ans(test_data)
test_prediction_list=get_prediction_list(test_data)

np.save("./feature/test_ques",test_ques)
np.save("./feature/test_ans",test_ans)
pd.to_pickle(test_prediction_list,"./feature/test_prediction_list.pkl")


# In[ ]:




