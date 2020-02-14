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

import gc
import numpy as np
import tensorflow as tf
from tensorflow import keras
#from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras import Input,Model,optimizers
# %matplotlib inline
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Conv1D, UpSampling1D,Activation
from tensorflow.keras.layers import MaxPooling1D, BatchNormalization, Flatten,GlobalMaxPooling1D
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping

from datetime import datetime
import time
from tensorflow.keras.callbacks import Callback, TensorBoard

# +
import pandas as pd
import random
from sklearn.metrics import confusion_matrix, recall_score, precision_score, f1_score, classification_report,plot_confusion_matrix
import glob
from tqdm import tqdm
import os
import itertools

physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    print('memory growth:', tf.config.experimental.get_memory_growth(physical_devices[0]))
else:
    print("Not enough GPU hardware devices available")
# -

VAL_RATE = 0.9

# +
#define
EPOCH = 3000
BATCH_SIZE = 128
MINIBATCH = 64
DROP_RATE = 0.5

FC_SIZE = 50
FILTER_SIZE = 64

DATA_START = 250
DATA_LEN = 4096

IN_DIR_PATH = "3ch_normalized"

os.makedirs('../logs/'+IN_DIR_PATH+'/events',exist_ok=True)

# +
# input_dir and out_dir
input_dir_path = "../preprocessed/train_"+IN_DIR_PATH+"/*/*.csv"
input_dir = glob.glob(input_dir_path)
input_num = len(input_dir)

class_list = np.array([])
all_data = np.array([])
all_label = np.array([])

init_flg = True

for d in tqdm(input_dir):
    # print(d)
    # delete DS_Sore 
    if ".DS_Store" in d:
        os.remove(d)
        input_dir.remove(d)
        continue
    
    # extract texture naem from path
    end_index = d.rfind('/')
    start_index = d[:end_index].rfind('/')
    class_name = d[start_index+1:end_index]
    # create classname list
    if class_name not in class_list:
        class_list = np.append(class_list,class_name)
        # print(class_name)
    # label classname list index
    label = np.where(class_list == class_name)
    all_label = np.append(all_label,label)
    
    data = np.loadtxt(d, delimiter=",")
    
    # axis_time = np.vstack(data[DATA_START:DATA_START+DATA_LEN,0])
    axis_value = np.hstack([data[DATA_START:DATA_START+DATA_LEN, 1:]])
    
    # preprocessed = np.hstack([axis_time,axis_value])
    if init_flg:
        all_data = axis_value
        init_flg = False
    else:
        all_data = np.dstack([all_data,axis_value])
# -

all_data_trans = np.transpose(all_data, (2,0,1))
print(all_data_trans.shape)

# +
one_hot_label = to_categorical(all_label,class_list.shape[0])

p = np.random.permutation(input_num)
shuffled_data = all_data_trans[p]
shuffled_label = one_hot_label[p]

trainX = shuffled_data[:int(input_num*VAL_RATE)]
valX = shuffled_data[int(input_num*VAL_RATE):]
trainY = shuffled_label[:int(input_num*VAL_RATE)]
valY = shuffled_label[int(input_num*VAL_RATE):]

trainX = np.reshape(trainX,(int(input_num*VAL_RATE),DATA_LEN,3)).astype(np.float32)
valX = np.reshape(valX,(input_num-int(input_num*VAL_RATE),DATA_LEN,3)).astype(np.float32)
# trainY = np.reshape(trainY,(int(input_num*VAL_RATE),class_list.shape[0],1))
# valY = np.reshape(valY,(input_num-int(input_num*VAL_RATE),class_list.shape[0],1))
# -

print(trainX.shape)
print(trainY.shape)


def Mynet():
    inputs = Input(shape=(DATA_LEN,3))
    # Due to memory limitation, images will resized on-the-fly.
    x = Conv1D(FILTER_SIZE, 5, padding='same', input_shape=(DATA_LEN,3), activation=None)(inputs)
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
    fc = Dense(FC_SIZE*2, activation='relu')(fc)
    fc = BatchNormalization()(fc)
    fc = Dropout(DROP_RATE)(fc)
    
    fc = Dense(FC_SIZE, activation='relu')(fc)
    fc = BatchNormalization()(fc)
    fc = Dropout(DROP_RATE)(fc)
    
    fc = Dense(class_list.shape[0])(fc)
    softmax = Activation('softmax')(fc)
    model = Model(inputs=inputs, outputs=softmax)
    
    return model


def Mynet_squeeze():
    inputs = Input(shape=(DATA_LEN,3))
    # Due to memory limitation, images will resized on-the-fly.
    x = Conv1D(FILTER_SIZE, 5, padding='same', input_shape=(DATA_LEN,3), activation='relu')(inputs)
    x = Conv1D(FILTER_SIZE, 5, padding='same',activation='relu')(x)
#     x = BatchNormalization()(x)
    x = MaxPooling1D(2, padding='same')(x)
    x = Conv1D(int(FILTER_SIZE*2), 5, padding='same',activation='relu')(x)
    x = MaxPooling1D(2, padding='same')(x)
    x = Dropout(DROP_RATE)(x)
    x = Conv1D(int(FILTER_SIZE), 5, padding='same', activation='relu')(x)
#     x = BatchNormalization()(x)
    
    fc = GlobalMaxPooling1D()(x)
    fc = Dropout(DROP_RATE)(fc)
    fc = Flatten()(fc)
    fc = Dense(class_list.shape[0])(fc)
    softmax = Activation('softmax')(fc)
    model = Model(inputs=inputs, outputs=softmax)
    
    return model


def plot_confusion_matrix(cm,
                          target_names,
                          title='Confusion matrix',
                          cmap=None,
                          normalize=False):
    """
    given a sklearn confusion matrix (cm), make a nice plot

    Arguments
    ---------
    cm:           confusion matrix from sklearn.metrics.confusion_matrix

    target_names: given classification classes such as [0, 1, 2]
                  the class names, for example: ['high', 'medium', 'low']

    title:        the text to display at the top of the matrix

    cmap:         the gradient of the values displayed from matplotlib.pyplot.cm
                  see http://matplotlib.org/examples/color/colormaps_reference.html
                  plt.get_cmap('jet') or plt.cm.Blues

    normalize:    If False, plot the raw numbers
                  If True, plot the proportions

    Usage
    -----
    plot_confusion_matrix(cm           = cm,                  # confusion matrix created by
                                                              # sklearn.metrics.confusion_matrix
                          normalize    = True,                # show proportions
                          target_names = y_labels_vals,       # list of names of the classes
                          title        = best_estimator_name) # title of graph

    Citiation
    ---------
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

    """
    import matplotlib.pyplot as plt
    import numpy as np
    import itertools

    accuracy = np.trace(cm) / np.sum(cm).astype('float')
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(16, 9))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig('../logs/'+IN_DIR_PATH+'/cm.png')

# +
model = Mynet_squeeze()
model.summary()
model.compile(loss='categorical_crossentropy',
             optimizer=optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08),
             metrics=['accuracy'])

start_time = time.time()
history = model.fit(trainX,trainY, 
                    epochs=3000, 
                    batch_size=BATCH_SIZE, 
                    verbose=1, 
                    validation_split=0.2,
                    callbacks=[tf.keras.callbacks.TensorBoard(log_dir='../logs/'+IN_DIR_PATH+'/events', histogram_freq=10, write_graph=True,),
#                                tf.keras.callbacks.EarlyStopping(monitor='val_loss',patience=700,verbose=0,mode='auto')
                              ]
                   )
end_time = time.time()-start_time
print('training_time : ',end_time)
score = model.evaluate(valX, valY, batch_size=BATCH_SIZE, verbose=0)


# +
fig, (axL, axR) = plt.subplots(ncols=2, figsize=(10,4))

# loss
def plot_history_loss(fit):
    # Plot the loss in the history
    axL.plot(fit.history['loss'],label="loss for training")
    axL.plot(fit.history['val_loss'],label="loss for validation")
    axL.set_title('model loss')
    axL.set_xlabel('epoch')
    axL.set_ylabel('loss')
    axL.set_ylim([0,3])
    axL.legend(loc='upper right')

# acc
def plot_history_acc(fit):
    # Plot the loss in the history
    axR.plot(fit.history['acc'],label="loss for training")
    axR.plot(fit.history['val_acc'],label="loss for validation")
    axR.set_title('model accuracy')
    axR.set_xlabel('epoch')
    axR.set_ylabel('accuracy')
    plt.legend(loc='lower right')

plot_history_loss(history)
plot_history_acc(history)
plt.show()
fig.savefig('../logs/'+IN_DIR_PATH+'/loss_acc.png')
plt.close()

# +
pred_y = model.predict(valX)
pred_y_c = np.argmax(pred_y,axis=1)
# pred_y_one_hot = np.identity(len(class_list))[pred_y_c]
true_y = np.argmax(valY,axis=1)
confusion_mtx = confusion_matrix(true_y, pred_y_c)
np.savetxt('../logs/'+IN_DIR_PATH+'/cm.csv', confusion_mtx)
plot_confusion_matrix(confusion_mtx, target_names=class_list)
plt.show()
cr = classification_report(true_y,pred_y_c,target_names=class_list)
f = open('../logs/'+IN_DIR_PATH+'/cr.txt','w')
f.write(cr)
f.close()
f = open('../logs/'+IN_DIR_PATH+'/trainingtime.txt','w')
f.write(str(end_time))
f.close()

keras.backend.clear_session()
gc.collect()
# -

