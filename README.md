# Fight-Detection
This repository is made as a part of ACM month of code. This contains fight detection algorithms.

Fight detection is a part of action recognition tasks which is one of the hardest problems in the computer vision as it includes recognizing the activities from videos making a prediction from a series of frames, not just a single frame, so training an action recognition model is itself a challange.

In this project we have used two approaches to solve the problem
1) Using pretrained CNN to extract the features from video frames and then passing the extracted frames to rnn to get the prediction.
2) using POSENET

  <h1> USING CRNN:- </h1>
      CNNs are better at recognizing the basic and high level features of an image and Recurrent networks works well with time dependent data or sequential data, so we tried to leverage the power of both the networks to predict the fight in the video.
      <h3>The basic workflow is as follows:</h3><br>
      <p>
            1) We have used pretrained cnn model to encode the predefined number of frames of a video and encoded it in a feature map, so basically CNN model is acting as an encoder network in the architecture. We have used Resnet152 architecture pretrained on imagenet dataset as an encoder network to generate the feature vectors.<br><br>
            2) Training a RNN model on a feature vectors to get the prediction, the best choices for this are deep unidirectional/bidirectional LSTM or GRU layers, as bidirectional LSTM/GRU are computationally expensive as compared to unidirectional LSTM/GRU and also we are not getting significant accuracy boost, so we have used unidirectional LSTM architecture with 2 hidden layer and two dense layer.here RNN model is acting as decoder which decodes the featuremap of different map to binary classes fight and non-fight.</p>
                                <p>   <h1>General Architecture</h1> </p>  
                                 <p align="center" margin-top="20">
                                          <img src="images_for_readme/crnn.png">
                                  </p>
     <p> Training this model on 300 fight and 300 non-fight video we have achieved 95% accuracy on test dataset </p><br>
    <h1> Steps taken to train the model </h1>
    <ul>
  <li> First we have extracted frames from videos using frame_extraction.py </li>
  <li> Then we have selected 40 frames from the total frames of video and then passed it through pretrained resnet152 model to extract feature vectors. The whole procedure is in feature_extraction.py </li>
  <li> Then we have trained RNN network, in which we have used unidirectional LSTM layers, to get the prediction of fight or no-fight from this feature maps. This can be obtained from rnn_training.py  </li>
    </ul>
   <h1> To test Fight detection model using crnn follow underlined steps.</h1>
<ul>
  <li> First clone the repository on your local machine. (make sure you have all the requirements given in requirement.txt)</li>
  <li> Then run test.py in command as " python test.py -m 'path_to_model' ", (model is given in model folder of fight_detection_using_crnn </li>
</ul>
  <h2> Project demo video </h2>
  <p>https://drive.google.com/file/d/1EovOeSgtOsyhsiSE1K91q0ddIgYLmdbx/view?usp=sharing</p> <br><br>
                                  
<h1> USING POSENET</h1>

<p align="center" margin-top="20">
    <img src="images_for_readme/pose_estimation.gif">
</p>

<p>
PoseNet can be used to estimate either a single pose or multiple poses. The model overlays keypoints over the input image.
</p>
  

  <p>
  PoseNet can be used to estimate either a single pose or multiple poses. The model overlays keypoints over the input image.
  </p>
    <p align="center" margin-top="20">
      <img src="images_for_readme/pose.png">
  </p>
  <p>
  Removing the background from this results in a much more simplified output that can be given to a CNN to get prediction:
   </p>
      <p align="center" margin-top="20">
      <img src="images_for_readme/pose.jpg">
  </p>

 <h1>Set-Up </h1>
 <ul>
  <li>sign up on https://www.sms4india.com/.</li>
   <li>get API&Secret keys </li>
</ul>
