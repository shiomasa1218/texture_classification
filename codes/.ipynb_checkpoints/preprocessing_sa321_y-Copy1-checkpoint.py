#!/usr/bin/env python
# coding: utf-8
# %%

# %%


import numpy as np
import os
import csv
import glob


# %%


preprocessed_name = "sa321_y"

# input_dir and out_dir
input_dir_path = "../rawdatas/*/*.csv"
input_dir = glob.glob(input_dir_path)
out_dir_path = "../preprocessed/train_"+preprocessed_name


# %%


for d in input_dir:
    print(d)
    if ".DS_Store" in d:
        os.remove(d)
        input_dir.remove(d)
        continue
    
    end_index = d.rfind('/')
    os.makedirs(out_dir_path+'/'+d[12:end_index], exist_ok=True)
        
    data = np.loadtxt(d, delimiter=",")
    
    axis_time = np.vstack(data[:,0])
    axis_y = np.vstack(data[:, 2])


     
    preprocessed = np.hstack([axis_time,axis_y])
    np.savetxt(out_dir_path+'/'+d[12:-4]+'_'+preprocessed_name+'.csv', preprocessed, delimiter=',')


# %%




