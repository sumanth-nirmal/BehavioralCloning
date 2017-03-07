#**Behavioral Cloning** 

This project **Behavioral Cloning** is done as a part of [selfdriving car Engineer NanoDegree](https://github.com/udacity/CarND-Term1-Starter-Kit) from udacity.

---

**Behavioral Cloning Project**

The approach follwed in this project is as follows:
* Use the [Udacity driving simulator](https://github.com/udacity/self-driving-car-sim) to collect data of good driving behavior, I have use the driving data given by udacity in the class.
* Desgined a neural network in Keras that predicts steering angles from images
* Trained and validated the model with a training and validation set (this is the split from the original dataset)
* Tested the model in the simulator for driving a complete lap.


## processData.py
This imports the raw images then resizes and normalises the images.
Resizing the images will have less features for network to train on. The resized images and the steering angles are saved as features and labels respondingly. The data is splitted into training and validation data and saved it as a .pickle file *(camera.pickle)*

**note: we can also use a python generator for this approach instead of a pickled dataset**                  
As the data set is smaller I have choosen pickle.

## model.py
This builds the neural network model details given below. Then compiles the model and saves the architecture as a .json file *(model.json)* Loads the data from pickle file and then trains the model over training data and evaluate over the test data and save the model with weights as .hd file *(model.h5)*

below shows the summary of model used for training              
        ___________________________________________________________________________________________________
        Layer (type)                     Output Shape          Param #     Connected to                     
        ====================================================================================================
        convolution2d_1 (Convolution2D)  (None, 16, 78, 16)    160         convolution2d_input_1[0][0]      
        ____________________________________________________________________________________________________
        activation_1 (Activation)        (None, 16, 78, 16)    0           convolution2d_1[0][0]            
        ____________________________________________________________________________________________________
        convolution2d_2 (Convolution2D)  (None, 14, 76, 8)     1160        activation_1[0][0]               
        ____________________________________________________________________________________________________
        activation_2 (Activation)        (None, 14, 76, 8)     0           convolution2d_2[0][0]            
        ____________________________________________________________________________________________________
        convolution2d_3 (Convolution2D)  (None, 12, 74, 4)     292         activation_2[0][0]               
        ____________________________________________________________________________________________________
        activation_3 (Activation)        (None, 12, 74, 4)     0           convolution2d_3[0][0]            
        ____________________________________________________________________________________________________
        convolution2d_4 (Convolution2D)  (None, 10, 72, 2)     74          activation_3[0][0]               
        ____________________________________________________________________________________________________
        activation_4 (Activation)        (None, 10, 72, 2)     0           convolution2d_4[0][0]            
        ____________________________________________________________________________________________________
        maxpooling2d_1 (MaxPooling2D)    (None, 5, 36, 2)      0           activation_4[0][0]               
        ____________________________________________________________________________________________________
        dropout_1 (Dropout)              (None, 5, 36, 2)      0           maxpooling2d_1[0][0]             
        ____________________________________________________________________________________________________
        flatten_1 (Flatten)              (None, 360)           0           dropout_1[0][0]                  
        ____________________________________________________________________________________________________
        dense_1 (Dense)                  (None, 16)            5776        flatten_1[0][0]                  
        ____________________________________________________________________________________________________
        activation_5 (Activation)        (None, 16)            0           dense_1[0][0]                    
        ____________________________________________________________________________________________________
        dense_2 (Dense)                  (None, 16)            272         activation_5[0][0]               
        ____________________________________________________________________________________________________
        activation_6 (Activation)        (None, 16)            0           dense_2[0][0]                    
        ____________________________________________________________________________________________________
        dense_3 (Dense)                  (None, 16)            272         activation_6[0][0]               
        ____________________________________________________________________________________________________
        activation_7 (Activation)        (None, 16)            0           dense_3[0][0]                    
        ____________________________________________________________________________________________________
        dropout_2 (Dropout)              (None, 16)            0           activation_7[0][0]               
        ____________________________________________________________________________________________________
        dense_4 (Dense)                  (None, 1)             17          dropout_2[0][0]                  
        ====================================================================================================
        Total params: 8023
               
 The model is with 8023 params, used *Adam optimizer*, *mean squared error* as loss metric. The model is trained for 30 epochs.
 
 **note**: getting more data and using pre processed data (i.e augmenting the images) and training it for more epochs would increase the accuracy.

## drive.py
This script is given by udacity in class.
Its a kind of ineference, predicts the steering angle using the model with trained weights *(model.hd)*, and these predicted steering commands are given to simulator with constant throttle (with simple PID controller) to drive the car in autonomous mode in simulator.

Since the images were reshaped and normalized during training, the image from the simulator is also reshaped and normalized just as in *processData.py* and *model.py*

## steps to run the code

**Just using the weights and running inference mode**

* `python drive.py model.h5 run1`         
this should use the pre trained weights and the model and predict the steering angle           
* then run the simulator in autonomous mode and the car should be sucessfully driven in the simulator and images hspould be saved in the folder run1

**Training and predicting**

* collect the data and modify the path accordingly in *processData.py*
* `python processData.py` this should iterate through all the images in the folder and generate pickle file 
* `python model.py` loads the pickled data and trains over it and should generate the model *model.json* and the model with weights *model.h5*
change the epochs if required
* `python drive.py model.h5 run1`          
this should use the pre trained weights and the model and predict the steering angle             
* then run the simulator in autonomous mode and the car should be sucessfully driven in the simulator and images hspould be saved in the folder run1

