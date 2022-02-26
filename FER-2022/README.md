# **Facial Expression Recognition in Image Sequences**

## **Overview**
The goal of this challenge is to develop an algorithm for recognizing facial expressions from video sequences. 
Each sequence corresponds to an expression, among the following Ekman's six universal expressions: anger, surprise, disgust, happiness, fear and sadness. 
For practical reasons, each video sequence consists of exactly 10 frames, from the neutral state, to the final state of the facial expression (named apex). 
Each image is in grayscale (1 single channel, with pixels between [0,255]).

In addition to the images, you can use distributions of characteristic points (called facial landmarks) to analyze facial expressions. 
Each landmark corresponds to a specific region of the face (e.g. corner of the mouth).

Full discription [here](https://www.kaggle.com/c/fer22/data).

## **Data**
For this challenge, we provide two datasets, the TRAIN dataset allowing to train your algorithms (contains 540 sequences of 15 subjects), 
and the TEST dataset allowing to test the efficiency of your algorithms (includes 131 sequences from 22 subjects).

- The *IMAGE data* (`data_train_images.h5`, `data_test_images.h5`) are represented by an array of dimension (540, 10, 300, 200) for the TRAIN 
and of dimension (131, 10, 300, 200) for the TEST. With in order: 1) the number of sequences, 2) the number of images per sequence, 3) 
the number of lines in the image (height), 4) the number of columns in the image (width).
- *The LANDMARKS data* (`data_train_landmarks.h5`, `data_test_landmarks.h5`) are represented by an array of dimension (540, 10, 68, 2) 
- for the TRAIN and of dimension (131, 10, 68, 2) for the TEST. With in order: 1) the number of sequences, 2) the number of images per sequence, 3) 
- the number of points in an image, 4) the X and Y position of a point in the image.
- *The LABELS data* (`data_train_labels.h5`,) are represented by an array of dimension (540) for the TRAIN data.
The encoding dictionary used for labels is: { 'joy': 0, 'fear': 1, 'surprise': 2, 'anger': 3, 'disgust': 4, 'sadness': 5, }
