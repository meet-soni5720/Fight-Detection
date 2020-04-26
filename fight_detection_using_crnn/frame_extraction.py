import sys
import os
import cv2
import imutils
from imutils.video import FPS
import numpy as np
from threading import Thread
import time
from tqdm import tqdm
import argparse 

if sys.version_info >= (3, 0):
    from queue import Queue

else:
    from Queue import Queue

parser = argparse.ArgumentParser()
parser.add_argument('-d','--dataPath',required=True, 
                   help="path to dataset")
parser.add_argument('-s','--savingPath',required=True, 
                    help="path to save frames")
args = vars(parser.parse_args())

class FileVideoStream:
    def __init__(self,path,queueSize=128):
        self.stream = cv2.VideoCapture(path)
        self.stopped = False
        self.Q = Queue(maxsize = queueSize)
        
    def start(self):
        t = Thread(target=self.update,args=())
        t.daemon = True
        t.start()
        return self

    def update(self):
        while True:
            if self.stopped:
                return
            if not self.Q.full():
                (grabbed, frame) = self.stream.read()
            
                if not grabbed:
                    self.stop()
                    return
            
                self.Q.put(frame)

    def read(self):
        return self.Q.get()

    def more(self):
        return self.Q.qsize() > 0

    def stop(self):
        self.stopped = True


data_path = args['dataPath']
save_path = args['savingPath']

video_list = os.listdir(data_path)
print(video_list)

for video in tqdm(video_list):
    video_path = os.path.join(data_path,video)
    name = video.split('.')[0]
    saving_at = os.path.join(save_path,name)
    if not os.path.exists(saving_at):
        os.makedirs(saving_at)
    
    fvs = FileVideoStream(video_path).start()
    time.sleep(1)  #allowing buffer to fill
    
    counter = 0
    while fvs.more():
        frame = fvs.read()
        frame = cv2.resize(frame,(224,224),interpolation = cv2.INTER_NEAREST)
        frame_name = os.path.join(saving_at,str(counter)+'.JPG')
        cv2.imwrite(frame_name,frame)
        counter+=1