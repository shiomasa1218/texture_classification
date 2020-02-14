# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.3.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

import numpy as np
import os
import csv
import glob
from scipy.signal import butter, filtfilt 
import scipy.fftpack
import time

# +
preprocessed_name = "dft321"

# input_dir and out_dir
input_dir_path = "../rawdatas/*/*.csv"
input_dir = glob.glob(input_dir_path)
out_dir_path = "../preprocessed/train_"+preprocessed_name

# +
fs = 1000.0 # サンプリング周波数 1000Hz => 1m秒間隔でサンプリング 

#通常用FFT
def ffta(data):
    return np.fft.fft(data)

def butter_lowpass(cutoff, fs, order=5): 
    nyq = 0.5 * fs 
    normal_cutoff = cutoff/nyq 
    b, a = butter(order, normal_cutoff, btype='low', analog=False) 
    return b, a 

def butter_lowpass_filtfilt(data, cutoff, fs, order=5): 
    b, a = butter_lowpass(cutoff, fs, order=order) 
    y = filtfilt(b, a, data) 
    return y 
  
def smtdA(csvdata):

    N = len(csvdata) # FFTのサンプル数 
    Ax = np.fft.fft(csvdata) 
    
    Ax_abs = np.abs(Ax) # 振幅スペクトル
   
    theta = np.angle(Ax)

    cutoff = 10
    condA = butter_lowpass_filtfilt(Ax_abs, cutoff, fs) #ゼロ位相遅延フィルタをかけて平滑化
    condA_real = condA*np.cos(theta)
    condA_imag = condA*np.sin(theta)
    
    return condA,condA_real,condA_imag


# +
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

    print(axis_x.shape)
    #smoothing
    Ax, real_x, imag_x = smtdA(axis_x.T)
    Ay, real_y, imag_y = smtdA(axis_y.T)
    Az, real_z, imag_z = smtdA(axis_z.T)
    
    #DFT321式（３）
    A_s = np.sqrt(np.abs(Ax)**2+np.abs(Ay)**2+np.abs(Az)**2)

    #
    real_sum = real_x+real_y+real_z
    imag_sum = imag_x+imag_y+imag_z

    theta = np.angle(real_sum+1j*imag_sum)

    ifftA = np.fft.ifft(A_s * np.exp(1j * theta))
    realA = ifftA.real.T
    print(realA.shape)
    preprocessed = np.hstack([axis_time,realA])
    np.savetxt(out_dir_path+'/'+d[12:-4]+'_'+preprocessed_name+'.csv', preprocessed, delimiter=',')

end_time = time.time()-start_time

f = open('../logs/'+IN_DIR_PATH+'/trainingtime.txt','w')
f.write(str(end_time))
f.close()
# -




