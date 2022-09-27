import numpy as np
#import matplotlib.pyplot as plt
from PIL import Image
import os
import cv2
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation,Flatten,Conv2D, MaxPooling2D
import tensorflow.keras as keras
from tensorflow.keras import layers
from keras.preprocessing.image import ImageDataGenerator

def create_model():
    model = Sequential([


    Conv2D(32, (3,3), activation='relu', input_shape=(110, 220, 3)
    ),
    MaxPooling2D(2, 2),

    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),

    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(2,2),

    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(2,2),

    Flatten(),
    Dropout(0.5),

    Dense(512, activation='relu'),
    Dense(17, activation='softmax')
    ])
    model.compile(loss = 'categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

batch_size = 100

train_datagen = ImageDataGenerator(
        rescale=1./255,
       )


test_datagen = ImageDataGenerator(rescale=1./255)


train_generator = train_datagen.flow_from_directory(
        'C:/Users/SAHIN/Desktop/New training/Dataset1',  
        target_size=(110, 220),  
        batch_size=batch_size,
        class_mode='categorical')  

validation_generator = test_datagen.flow_from_directory(
        'C:/Users/SAHIN/Desktop/New training/test',
        target_size=(110, 220),
        batch_size=batch_size,
        class_mode='categorical', shuffle=False)

callbacks = tf.keras.callbacks.ModelCheckpoint(filepath='C:/Users/SAHIN/Desktop/New training/best_model.h5',monitor='val_loss', mode
='min', save_best_only=True, verbose=1)

myModel = create_model()
myModel.fit_generator(
        train_generator,
        steps_per_epoch=  train_generator.samples   // batch_size,
        epochs=100,
        callbacks=callbacks,
        
        validation_data=validation_generator,
        validation_steps= validation_generator.samples // batch_size, verbose = 1)
myModel.save('/myModel.h5')