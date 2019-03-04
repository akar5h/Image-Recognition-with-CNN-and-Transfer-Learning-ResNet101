import tqdm
import pandas as pd
import matplotlib.pyplot as plt
import random

import glob
import shutil
import zipfile

import pickle
import h5py

from sklearn.metrics import confusion_matrix

import keras
from keras import backend as K
from keras.callbacks import  EarlyStopping, Callback
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from keras import optimizers
from keras.models import Sequential, model_from_json
from keras.layers import  Conv2D, MaxPool2D,Activation,Dropout,Flatten,Dense,BatchNormalization
from keras.optimizers import Adam

import tensorflow
import numpy


img_width = 224
img_height = 224
split_size = 0.2
batch_size = 128

channnel = 3

train_folder = '/home/ravi/Desktop/akarsh/dataset/'
valid_folder = '/home/ravi/Desktop/akarsh/valid_dataset/'
test_folder = '/home/ravi/Desktop/akarsh/test_dataset/'

#create image generator and dataset
datagen = ImageDataGenerator(rescale=1./255,rotation_range=5,zoom_range=0.2,horizontal_flip=True)
train_generator = datagen.flow_from_directory(train_folder, color_mode='rgb',target_size=(img_width,img_height),batch_size=5,classes=['bed','chair','table','wardrobe'])


vgg16_model = keras.applications.vgg16.VGG16()
print('VGG16 Summary:')
vgg16_model.summary()

model = Sequential()

for layer in vgg16_model.layers:
    model.add(layer)

model.layers.pop()

for layer in model.layers:
    layer.trainable = False

model.add(Dense(4, activation='softmax'))
model.summary()

#Training
valid_generator = datagen.flow_from_directory(valid_folder,color_mode='rgb',target_size=(img_width,img_height),batch_size=2,classes=['bed','chair','table','wardrobe'])
test_generator = datagen.flow_from_directory(test_folder,color_mode='rgb',target_size=(img_width,img_height),batch_size=1,classes=['bed','chair','table','wardrobe'])

model.compile(Adam(lr=0.0001),loss='categorical_crossentropy',metrics=['accuracy'])
model.fit_generator(train_generator, validation_data=valid_generator, epochs=5, steps_per_epoch = 5, validation_steps= 4  )
