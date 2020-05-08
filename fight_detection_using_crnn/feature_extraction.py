import os
import tqdm
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications.resnet import ResNet152
from tensorflow.keras.callbacks import ReduceLROnPlateau,ModelCheckpoint
from tensorflow.keras.models import Model,load_model,save_model,Sequential
from tensorflow.keras.layers import Dense,BatchNormalization,Bidirectional,LSTM,Dropout,Conv2D,AveragePooling2D,Flatten


os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

class Dataloader:
    def __init__(self, datapath, classes, max_frames, img_shape, channels, saving_dir):
        self.datapath = datapath
        self.classes = classes
        self.seq_length = max_frames
        self.height = img_shape[0]
        self.width = img_shape[1]
        self.channels = channels
        self.base_model = ResNet152(include_top = False,input_shape=(224,224,3),weights = 'imagenet')
        #self.base_model.load_weights(r'D:\\Downloads\\resnet152_weights_tf_dim_ordering_tf_kernels_notop.h5')
        self.saving_dir = saving_dir
        
        for layer in self.base_model.layers:
            layer.trainable = False
        self.op = self.base_model.output
        self.x_model = AveragePooling2D((7,7),name='avg_pool')(self.op)
        self.x_model = Flatten()(self.x_model)
        
        self.model = Model(self.base_model.input,self.x_model)
        print(self.model.summary())
        
            
    def get_frame_sequence(self,path):
        flag = 1
        total_frames = os.listdir(path)
        arr = np.zeros((224,224,3,40)) # for 40 frames
        if len(total_frames) >= 160:
            counter = 0
            for i in range(1,160,4):
                x = Image.open(os.path.join(path,str(i) + '.JPG'))
                x = np.array(x)
                arr[:,:,:,counter] = x
                counter+=1
                if(counter >= self.seq_length):
                    break
                    
        elif((len(total_frames) >= 120) and (len(total_frames) <160)):
            counter = 0
            for i in range(1,120,3):
                x = Image.open(os.path.join(path,str(i) + '.JPG'))
                x = np.array(x)
                arr[:,:,:,counter] = x
                counter+=1
                if(counter >= self.seq_length):
                    break
        
        elif((len(total_frames) >= 99) and (len(total_frames) < 120)):
            counter = 0
            for i in range(0,40,2):
                x = Image.open(os.path.join(path,str(i) + '.JPG'))
                x = np.array(x)
                arr[:,:,:,counter] = x
                counter+=1
            for i in range(41,99,3):
                x = Image.open(os.path.join(path,str(i) + '.JPG'))
                x = np.array(x)
                arr[:,:,:,counter] = x
                counter += 1
                if(counter >= self.seq_length):
                    break
                
        elif((len(total_frames) >= 80) and (len(total_frames) < 98)):
            counter = 0
            for i in range(0,80,2):
                x = Image.open(os.path.join(path,str(i) + '.JPG'))
                x = np.array(x)
                arr[:,:,:,counter] = x
                counter += 1
                if(counter == self.seq_length):
                    break
        elif((len(total_frames)) >= 39):
            counter = 0
            for i in range(40):
                x = Image.open(os.path.join(path,str(i) + '.JPG'))
                x = np.array(x)
                arr[:,:,:,counter] = x
                counter += 1
                if(counter >= self.seq_length):
                    break
        else:
             flag = 0
                    
        #print(arr.shape)
        return flag,arr
            
    def extract_feature(self,x_train):
        x_op = np.zeros((2048,40))
        for i in range(x_train.shape[3]):
            x_t = x_train[:,:,:,i]
            x_t = np.expand_dims(x_t,axis = 0)
            x = self.model.predict(x_t)
            x_op[:,i] = x
        
        return x_op
    
    def get_all_sequence_in_memory(self):
        counter = 0
        y_train = []
        x_train = []
        for i in self.classes:
            directory_path = os.path.join(self.datapath,i)
            if i == 'violence':
                y = 1
            else:
                y = 0
                
            list_dir = os.listdir(directory_path)
            for folder in tqdm.tqdm(list_dir):
                path = os.path.join(directory_path,folder)
                flag,arr = self.get_frame_sequence(path)
                if(flag == 1):
                    x_ext = self.extract_feature(arr)
                    x_train.append(x_ext)
                    counter+=1
                    y_train.append(y)
        save_file_x = os.path.join(self.saving_dir,'data_x_ext.npy')
        save_file_y = os.path.join(self.saving_dir,'data_y.npy')
        np.save(save_file_x,np.array(x_train))
        np.save(save_file_y,np.array(y_train))
        return x_train,y_train
    
    def load_npy_file(self):
        x_train = np.load(self.datapath + 'data_x.npy')
        y_train = np.load(self.datapath + 'data_y.npy')
        
        return x_train,y_train

def load_data():
    data_loader = Dataloader(r'D:\temp_fight',['violence','non_violence'],40,(224,224),3,r'D:\temp_fight\saving_things')
    x_val,y_val = data_loader.get_all_sequence_in_memory()
    return x_val,y_val

if __name__ == '__main__':
    x_train,y_train = load_data()