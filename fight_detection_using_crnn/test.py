import sys
import os
import numpy as np 
import cv2
import tensorflow as tf
from tensorflow import keras
from keras import backend as K
from tensorflow.keras.models import load_model,Model
from tensorflow.keras.applications.resnet import ResNet152
from tensorflow.keras.layers import Dense,AveragePooling2D,Flatten
import argparse
import imutils
import threading
import time
import requests
import json
from phone_message import sendPostRequest
from datetime import datetime,date
import pytz

now = datetime.now()

# assuming now contains a timezone aware datetime
tz = pytz.timezone('Asia/Kolkata')
your_now = now.astimezone(tz)
print(your_now)

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

if sys.version_info >= (3, 0):
    from queue import Queue

else:
    from Queue import Queue


parser = argparse.ArgumentParser()
parser.add_argument('-m','--modelPath',required=True,
                    help="path to model")
'''parser.add_argument('-w','--weights',required=True,
                    help="weights file for encoder network")'''

args = vars(parser.parse_args())

''' Encoder Network architecture'''
encoder_network = ResNet152(include_top = False,input_shape=(224,224,3),weights = 'imagenet')
#encoder_network.load_weights(args['weights'])
for layer in encoder_network.layers:
            layer.trainable = False
op = encoder_network.output
x_model = AveragePooling2D((7,7),name='avg_pool')(op)
x_model = Flatten()(x_model)
        
encoder_model = Model(encoder_network.input,x_model)
print("Encoder Model prepared....")

#Decoder Network(laoding trained model)
decoder_network = load_model(args['modelPath'],compile=False)
print("Decoder network prepared.....")

def main(encoder_model,decoder_network):
    vs = cv2.VideoCapture(0)
    grabbed,frame = vs.read()
    time.sleep(0.5)
    counter = 0
    x_rnn = np.zeros((40,2048))
    color = (0,255,0)
    classes = ['no fight','fight']
    default = 'no fight'
    flag = 1
    d_flag = 1
    app_url = 'https://www.sms4india.com/api/v1/sendCampaign'
    curr_date = datetime.now().astimezone(tz).date()
    curr_time = datetime.now().astimezone(tz).time()

    while True:
        grabbed,frame = vs.read()
        frame_1 = cv2.resize(frame,(224,224),interpolation = cv2.INTER_NEAREST)
        f_x = np.array(frame_1)
        f_x = np.expand_dims(f_x,axis = 0)
        feature_map = encoder_model.predict(f_x)
        x_rnn[counter,:] = np.array(feature_map)
        counter+=1
        if((counter%40) == 0 and flag == 1):
            counter = 0
            flag = 0
            print("prediction time")
            x_test = np.expand_dims(x_rnn,axis=0)
            y_val = decoder_network.predict(x_test)
            print(y_val)
            if(y_val >= 0.02):
                class_val = 1
            else:
                class_val = 0
            prediction = classes[class_val]
            default = prediction
            if default=='fight':
                color = (0,0,255)
            else:
                color = (0,255,0)
        elif((counter%10) == 0 and flag == 0):
            print("prediction time")
            x_test = np.expand_dims(x_rnn,axis=0)
            y_val = decoder_network.predict(x_test)
            print(y_val)
            if(y_val >= 0.02):
                class_val = 1
            else:
                class_val = 0
            prediction = classes[class_val]
            default = prediction
            if default=='fight':  #send a message to authority if time difference from last time is > 60 minutes
                if(d_flag == 1):
                    d_flag = 0
                    last_date = datetime.now().astimezone(tz).date()
                    last_time = datetime.now().astimezone(tz).time()
                    response = sendPostRequest(app_url, 'Y0CRJP21JW1CEOE9MNKJ2PD1U2RQKMDC', 'XD326W5W51BT16D4', 'stage', '9925335903', 'Meet', 'Fight detected at date: ' + str(last_date) + 'at time: ' + str(last_time) )
                else:
                    curr_date = datetime.now().astimezone(tz).date()
                    curr_time = datetime.now().astimezone(tz).time()
                    if(curr_date != last_date):
                        response = sendPostRequest(app_url, 'Y0CRJP21JW1CEOE9MNKJ2PD1U2RQKMDC', 'XD326W5W51BT16D4', 'stage', '9925335903', 'Meet', 'Fight detected at date: ' + str(curr_date) + 'at time: ' + str(curr_time) )
                        last_date = curr_date
                        last_time = curr_time
                    else:
                        if((((datetime.combine(date.today(), curr_time) - datetime.combine(date.today(), last_time)).total_seconds())/60) >= 60.00):
                            response = sendPostRequest(app_url, 'Y0CRJP21JW1CEOE9MNKJ2PD1U2RQKMDC', 'XD326W5W51BT16D4', 'stage', '9925335903', 'Meet', 'Fight detected at date: ' + str(curr_date) + 'at time: ' + str(curr_time) )
                            last_date = curr_date
                            last_time = curr_time

                color = (0,0,255)
            else:
                color = (0,255,0)
            
            if((counter%40) == 0):
                counter = 0
            
        frame = cv2.putText(frame,str(default), (90,40), cv2.FONT_HERSHEY_SIMPLEX, 2, color, 2, cv2.LINE_AA)
        frame = cv2.putText(frame,'collecting frames',(70,90),cv2.FONT_HERSHEY_SIMPLEX, 2, color, 1, cv2.LINE_AA)
        cv2.imshow('live_fight_detection',frame)
        key = cv2.waitKey(1) & 0xFF
        if(key == 27):
            break

if __name__ == '__main__':
    print("starting stream...")
    main(encoder_model,decoder_network)