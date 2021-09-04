
# Sign language detection

## 1. Introduction
This project explores the possibility of computer learning how to recognize sign language patterns. 
To achieve that, deep neural network and MediaPipe framework were used through implementation in Python.
Project is assignment done for course Machine Learning which is held on The Faculty of Electrical Engineering, Computer Science and Information Technology Osijek

## 2. Making dataset
There was no available dataset on internet so making one was necessary. Dataset contains six words in Croatian language (Pozdrav, Hvala, Oprosti, Ja, Kuća, Naočale), each containg 50 sequences. One sequence contains 60 frames which captured movement of hands while doing sign language patter and then those values were saved into numpy arrays. Every numpy array was filled with hand landmarks provided by detection system MediaPipe framework has. Because using both hands in detection, 126 values were extracted in every numpy array ( 21 keypoint per hand and one keypoint contains values for x,y,z coordinates).

## 3. Creating model
Dataset was splitted in training and validation set using 25% data for validation and rest for training. Sequential model was best suitable for this problem so to create one, two LSTM layers and one Dense layers were used. Best settings showed to be while using 24 neurons in LSTM layers with tahn activation function. Desired output was in probabilities so Dense layer used softmax function. 

## 4. Evaluation
To determine if model is good enough, validation accuracy and loss were main subject of observing. With values of 0.96 for val. accuracy and 0.1809 for val. loss this model is well equiped for prediction of sign language or atleast for six words included in dataset. One thing to look at is confusion matrix which shows that model has problems with words which are simmilar in movement. 

## 5. Testing in real world application
Script detection_test.py contains python program for testing model prediction. Using camera feed you can make a correct sign language pattern for words in dataset. Feedback on screen will tell you which word and in what probability, is predicted by model.



More detailed documentation is written in folder Documentation(Croatian language). 