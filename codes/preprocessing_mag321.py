#!/usr/bin/env python
# coding: utf-8
# %%
import numpy as np
import os
import csv
import glob
import time


# %%
preprocessed_name = "mag321"

# input_dir and out_dir
input_dir_path = "../rawdatas/*/*.csv"
input_dir = glob.glob(input_dir_path)
out_dir_path = "../preprocessed/train_"+preprocessed_name


# %%
start_time = time.time()
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
    axis_x = np.vstack(data[:, 1])
    axis_y = np.vstack(data[:, 2])
    axis_z = np.vstack(data[:, 3])

    calc = np.sqrt(axis_x**2 + axis_y**2 + axis_z**2)
    
    preprocessed = np.hstack([axis_time,calc])
    np.savetxt(out_dir_path+'/'+d[12:-4]+'_'+preprocessed_name+'.csv', preprocessed, delimiter=',')

end_time = time.time()-start_time

f = open(out_dir_path+'/'+d[12:-4]+'_'+preprocessed_name+'_processtime.txt','w')
f.write(str(end_time))
f.close()

