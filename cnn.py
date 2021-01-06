# -*- coding: utf-8 -*-
"""
Created on Wed Dec 30 13:21:23 2020

@author: Leo
"""

from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

#Inicializando 

classifier = Sequential()
#Layer 1
classifier.add(Convolution2D(32, 3, 3, input_shape= (64, 64, 3), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))

#Layer 2
classifier.add(Convolution2D(32, 3, 3, activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))

classifier.add(Flatten())

classifier.add(Dense(128, activation = 'relu'))
classifier.add(Dense(1, activation = 'sigmoid'))

classifier.compile(optimizer='adam', loss='binary_crossentropy',metrics=['accuracy'])

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(
        'dataset/training_set',
        target_size=(64, 64), #same size as input_shape
        batch_size=32,
        class_mode='binary')

test_set = test_datagen.flow_from_directory(
        'dataset/test_set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')

classifier.fit(
        training_set,
        steps_per_epoch=250, #len(training_set)
        epochs=25,
        validation_data=test_set,
        validation_steps=62) #len(test_set)