import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense,BatchNormalization,Bidirectional,LSTM,Dropout,Conv2D,AveragePooling2D,Flatten,Input
from tensorflow.keras.models import Model,load_model,save_model,Sequential
from tensorflow.keras.callbacks import ReduceLROnPlateau,ModelCheckpoint
from tensorflow.keras.optimizers import Adam
import glob
import tqdm
import pickle
from keras import backend as K
from tensorflow.keras import regularizers
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-d','--dataset',required=True,
                    help="path to dataset folder")
parser.add_argument('-s','--savingDir',required=True,
                    help="path to save model")

args = vars(parser.parse_args())
data_x = os.path.join(args['dataset'],'data_x_ext.npy')
data_y = os.path.join(args['dataset'],'data_y.npy')

x_train = np.load(data_x)
y_train = np.load(data_y)

''' unfortunately data shape is (2048,40) ie sequential feature is last so we need to take transpose'''

x_r = []
for i in range(x_train.shape[0]):
  x_temp = x_train[i,:,:]
  x_t = x_temp.T
  x_r.append(x_t)

x_train = np.array(x_r)

K.clear_session()

def rnn_model():
  x_input = Input(shape=(40,2048))
  x = LSTM(units=1024,return_sequences=True,dropout=0.4)(x_input)
  x = LSTM(units = 512,return_sequences=False,dropout=0.3)(x)
  x = Dense(256,activation='relu',kernel_regularizer=regularizers.l2(0.01))(x)
  x = BatchNormalization()(x)
  x = Dropout(0.3)(x)
  x = Dense(256,activation='relu',kernel_regularizer=regularizers.l2(0.01))(x)
  x = BatchNormalization()(x)
  x = Dense(1,activation='sigmoid')(x)

  model = Model(inputs=x_input,outputs = x)
  model.summary()
  adam = Adam(lr=0.005,decay=1e-6)

  model.compile(optimizer=adam,loss='binary_crossentropy',metrics=['accuracy'])
  return model

model = rnn_model()

cb = [ReduceLROnPlateau(patience=5,verbose=1),]
output = model.fit(x_train,y_train,epochs=50,batch_size=32,validation_split=0.2,verbose=1,callbacks=cb,shuffle=True)

model.save(os.path.join(args['savingDir'],'fd_model.h5'))

model.evaluate(x_train,y_train)

acc_val = output.history['acc']
val_acc = output.history['val_acc']
loss = output.history['loss']
val_loss = output.history['val_loss']

epochs = range(1,len(acc_val)+1)
plt.plot(epochs, acc_val, 'b', label='Training accuracy')
plt.plot(epochs, val_acc, 'r', label='Validation accuracy')
plt.title('Training and Validation accuracy')
plt.legend(loc='upper right')

plt.figure()

plt.plot(epochs, loss, 'b', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='validation loss')
plt.title('Training and Validation loss')
plt.legend(loc = 'upper right')
plt.show()