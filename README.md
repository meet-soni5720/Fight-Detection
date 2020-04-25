# Fight-Detection
This repository is made as a part of ACM month of code. This contains fight detection algorithm which is integrated in webapp.

In this project we have used two approaches to solve the problem
1) Using pretrained CNN to extract the features from video frames and then passing the extracted frames to rnn to get the prediction.
2) using POSENET

USING CRNN:-
      We have used pretrained Resnet152 model trained on imagenet dataset to extract the features from frames of videos and converting each frames to 2048 dimensional vector which is then passed as (2048,num_frames) dimenisonal tensor in a Recurrent network.
