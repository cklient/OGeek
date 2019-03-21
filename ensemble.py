#!/usr/bin/env python
# coding: utf-8

# In[31]:


import pandas as pd


val_dy=pd.read_csv("val_bed_case_f1.csv")
val_tsf=pd.read_csv("val_1.csv",header=None,names=['tsf_proba'])


# In[33]:


from  sklearn.metrics import f1_score
val_label=val_dy.label.values

best_dy_w=None
best_sf_w=None
best_f1=0

for i in  range(1,9):
    for j in  range(1,9):
        dy_w=i*0.1+j*0.01
        sf_w=1-dy_w
        stack_proba=val_dy.predict_proba.values*dy_w+val_tsf.tsf_proba.values*sf_w
        stack_label=(stack_proba>0.4).astype(int).reshape(-1)
        f1=f1_score(val_label,stack_label)
        if f1>best_f1:
            best_f1=f1
            best_dy_w=dy_w
            best_sf_w=sf_w
        print(dy_w,sf_w,f1)
    
print("best params",best_dy_w,best_sf_w,best_f1)


# In[42]:


test_dy=pd.read_csv("test_data_prob_f10.7516109844354119.csv")
test_tsf=pd.read_csv("test_0.749969.csv",header=None,names=['sf_proba'])


# In[43]:


print(best_dy_w,best_sf_w)
stack_proba=test_dy.predict_proba.values*best_dy_w+test_tsf.sf_proba*best_sf_w


# In[44]:


test_dy['stack_proba']=stack_proba


# In[45]:


test_dy['label']=test_dy.stack_proba.map(lambda x: 1 if x>0.4 else 0)




test_dy['label'].astype(int).to_csv("result.csv"%best_f1,index=False)





