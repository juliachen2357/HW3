# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
# import csv
# import cv2
# import matplotlib.pyplot as plt
# data_index=[]
# SteerAngle=[]
# Throttle=[]
# Brake=[]
# Speed=[]
# X_Position=[]
# Y_Position=[]
# Pitch=[]
# Yaw=[]
# Roll=[]

# with open('C:/Users/czyji/Desktop/train/robot_log.csv', newline='') as csvfile:
#      spamreader = csv.reader(csvfile, delimiter=';', quotechar='|')
#      for row in spamreader:
#          if row[0]!='Path':
#              data_index.append(int(row[0][-7:-4]))
#              SteerAngle.append(float(row[1]))
#              Throttle.append(float(row[2]))
#              Brake.append(float(row[3]))
#              Speed.append(float(row[4]))
#              X_Position.append(float(row[5]))
#              Y_Position.append(float(row[6]))
#              Pitch.append(float(row[7]))
#              Yaw.append(float(row[8]))
#              Roll.append(float(row[9]))
# # import required module
# import os
# # assign directory
# directory = 'C:/Users/czyji/Desktop/train/IMG'

# # iterate over files in
# # that directory
# for filename in os.listdir(directory):
#     f = os.path.join(directory, filename)
# 	# checking if it is a file
#     if os.path.isfile(f):
#         im=cv2.imread(f)
#         resized = cv2.resize(im, (180,180), interpolation = cv2.INTER_AREA)
#         cv2.imwrite('C:/Users/czyji/Desktop/train/IMG2/'+filename, resized)
# directory = 'C:/Users/czyji/Desktop/test/IMG'

# # iterate over files in
# # that directory
# for filename in os.listdir(directory):
#     f = os.path.join(directory, filename)
# 	# checking if it is a file
#     if os.path.isfile(f):
#         im=cv2.imread(f)
#         resized = cv2.resize(im, (180,180), interpolation = cv2.INTER_AREA)
#         cv2.imwrite('C:/Users/czyji/Desktop/test/IMG2/'+filename, resized)
        
        
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import tensorflow as tf
tf.compat.v1.set_random_seed(2019)
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(16,(3,3),activation = "relu" , input_shape = (180,180,3)) ,
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(32,(3,3),activation = "relu") ,  
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(64,(3,3),activation = "relu") ,  
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(128,(3,3),activation = "relu"),  
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(), 
    tf.keras.layers.Dense(550,activation="relu"),      #Adding the Hidden layer
    tf.keras.layers.Dropout(0.1,seed = 2019),
    tf.keras.layers.Dense(400,activation ="relu"),
    tf.keras.layers.Dropout(0.3,seed = 2019),
    tf.keras.layers.Dense(300,activation="relu"),
    tf.keras.layers.Dropout(0.4,seed = 2019),
    tf.keras.layers.Dense(200,activation ="relu"),
    tf.keras.layers.Dropout(0.2,seed = 2019),
    tf.keras.layers.Dense(5,activation = "softmax")   #Adding the Output Layer
])
model.summary()
from tensorflow.keras.optimizers import RMSprop,SGD,Adam
adam=Adam(lr=0.001)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics = ['acc'])
bs=30         #Setting batch size
train_dir = "C:/Users/czyji/Desktop/train/"   #Setting training directory
validation_dir = "C:/Users/czyji/Desktop/train/test/"   #Setting testing directory
from tensorflow.keras.preprocessing.image import ImageDataGenerator 
# All images will be rescaled by 1./255.
train_datagen = ImageDataGenerator( rescale = 1.0/255. )
test_datagen  = ImageDataGenerator( rescale = 1.0/255. )
# Flow training images in batches of 20 using train_datagen generator
#Flow_from_directory function lets the classifier directly identify the labels from the name of the directories the image lies in
train_generator=train_datagen.flow_from_directory(train_dir,batch_size=bs,class_mode='categorical',target_size=(180,180))
# Flow validation images in batches of 20 using test_datagen generator
validation_generator =  test_datagen.flow_from_directory(validation_dir,
                                                         batch_size=bs,
                                                         class_mode  = 'categorical',
                                                         target_size=(180,180))
