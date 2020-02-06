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
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout,Input
# %matplotlib inline
import matplotlib.pyplot as plt
from tensorflow.keras.layers.convolutional import Conv1D, UpSampling1D
from tensorflow.keras.layers.pooling import MaxPooling1D
import pandas as pd
import random
from sklearn.metrics import confusion_matrix, recall_score, precision_score, f1_score
import glob

# +
preprocessed_name = "sa321_y"

# input_dir and out_dir
input_dir_path = "../rawdatas/*/*.csv"
input_dir = glob.glob(input_dir_path)
out_dir_path = "../preprocessed/train_"+preprocessed_name

# +
#define
EPOCH = 3000
BATCH_SIZE = 600
VAL_RATE = 0.8
MINIBATCH = 256
DROP_RATE = 0.5

CLASS_NUM = 9

FC_SIZE = 1024
FILTER_SIZE = 64

IN_DIR_PATH = "train_sa321_x"
OUT_DIR_PATH = ""


# -

def Mynet():
    inputs = Input(shape=(MINIBATCH,1))
    # Due to memory limitation, images will resized on-the-fly.
    x = Conv1D(FILTER_SIZE, 5, padding='same', input_shape=(MINIBATCH, 1), activation=None)(inputs)
    x = Conv1D(FILTER_SIZE, 5, padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling1D(2, padding='same')(x)
    x = Dropout(DROP_RATE)(x)

    x = Conv1D(FILTER_SIZE*2, 5, padding='same')(x)
    x = Conv1D(FILTER_SIZE*2, 5, padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling1D(2, padding='same')(x)
    x = Dropout(DROP_RATE)(x)
    
    x = Conv1D(FILTER_SIZE*4, 5, padding='same')(x)
    x = Conv1D(FILTER_SIZE*4, 5, padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling1D(2, padding='same')(x)
    x = Dropout(DROP_RATE)(x)
    
    fc = Flatten()(x)
    fc = Dense(FC_SIZE, activation='relu')(fc)
    fc = BatchNormalization()(fc)
    fc = Dropout(DROP_RATE)(fc)
    
    fc = Flatten()(fc)
    fc = Dense(FC_SIZE, activation='relu')(fc)
    fc = BatchNormalization()(fc)
    fc = Dropout(DROP_RATE)(fc)
    
    softmax = Dense(CLASS_NUM, activation='softmax')(fc)
    model = Model(input=input, output=softmax)
    
    return model


model = Mynet()
model.summary()


