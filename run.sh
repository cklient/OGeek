#!/bin/bash
mkdir CSV
mkdir feature
python3 gen_count_feature.py
python3 gen_len_feature.py
python3 gen_strsma_feature.py
python3 gen_title_dist_tag_feature.py
python3 gen_unique_feature.py
python3 gen_w2v_feature1.py
python3 lgb_model-dy.py


python3 1_make_traindata.py
python3 3_train_model_2.py

python3 Ensmble.py