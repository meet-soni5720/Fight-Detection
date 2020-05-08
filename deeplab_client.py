import tensorflow as tf
assert tf.__version__ == '1.14.0'


# In[116]:


import io
import requests
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from grpc.beta import implementations
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2


# In[120]:


server = '127.0.0.1:8500'
host, port = server.split(':')

# create the RPC stub
channel = implementations.insecure_channel(host, int(port))
stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)

# create the request object and set the name and signature_name params
request = predict_pb2.PredictRequest()
request.model_spec.name = 'deepnet'
request.model_spec.signature_name = 'predict_images'


# In[154]:


def get_four_channel_image(PATH):
    print(PATH)
    ext = PATH[-3:]
    image = np.array(Image.open(PATH))
    height,width,_ = image.shape
    
    # fill in the request object with the necessary data
    request.inputs['images'].CopyFrom(tf.contrib.util.make_tensor_proto(image.astype(dtype=np.float32), shape=[1, height, width, 3]))
    request.inputs['height'].CopyFrom(tf.contrib.util.make_tensor_proto(image.shape[0], shape=[1]))
    request.inputs['width'].CopyFrom(tf.contrib.util.make_tensor_proto(image.shape[1], shape=[1]))

    # For async requests
    result_future = stub.Predict.future(request, 10.)
    result_future = result_future.result()

    # get the results
    output = np.array(result_future.outputs['segmentation_map'].int64_val)
    height = result_future.outputs['segmentation_map'].tensor_shape.dim[1].size
    width = result_future.outputs['segmentation_map'].tensor_shape.dim[2].size
    image_mask = np.reshape(output, (height, width))
    
    shape = image.shape
    temp_image = np.zeros((shape[0],shape[1],shape[2]+1))
    temp_image[:,:,:3] = image
    temp_image[:,:,3] = image_mask

    return temp_image,ext


# In[134]:


cv2.imwrite('fourchannelimage.jpg',temp_image)


# In[135]:


import cv2


# In[136]:


os.getcwd()


# In[137]:


PATH_TO_VIOLENCE = "images/violence"
PATH_TO_NON_VIOLENCE = "images/non_voilence"

SAVE_PATH_TO_VIOLENCE = "images/violence_4channeled"
SAVE_PATH_TO_NON_VIOLENCE = "images/non_voilence_4channeled"

if not os.path.exists(SAVE_PATH_TO_VIOLENCE):
    os.mkdir(SAVE_PATH_TO_VIOLENCE)
    
if not os.path.exists(SAVE_PATH_TO_NON_VIOLENCE):
    os.mkdir(SAVE_PATH_TO_NON_VIOLENCE)


# In[156]:


def make_four_channel_images(PATH_TO_FOLDER,save_path):
    index = 0
    for folder in os.listdir(PATH_TO_FOLDER):
        folder_path = os.path.join(PATH_TO_FOLDER,folder)
        for image in os.listdir(folder_path):
#             print(image)
            image_path = os.path.join(folder_path,image)
            four_channel_image,ext = get_four_channel_image(image_path)
            cv2.imwrite(save_path +'/'+ str(index) +'.'+ext,four_channel_image)        
            index+=1


# In[157]:


make_four_channel_images(PATH_TO_VIOLENCE,SAVE_PATH_TO_VIOLENCE)


# In[ ]:





# In[ ]:

