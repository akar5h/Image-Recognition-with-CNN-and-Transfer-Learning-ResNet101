import h5py
import keras
from keras import backend as K
from keras.callbacks import  EarlyStopping, Callback
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from keras import optimizers
from keras.models import Sequential, model_from_json
from keras.layers import  Conv2D, MaxPool2D,Activation,Dropout,Flatten,Dense,BatchNormalization
from keras.optimizers import Adam
from keras.applications import ResNet50
import tensorflow
import numpy

img_width = 224
img_height = 224

channnel = 3

train_folder = '/../dataset/'
valid_folder = '/../valid_dataset/'

#create image generator and dataset
datagen = ImageDataGenerator(rescale=1./255,rotation_range=5,zoom_range=0.2,horizontal_flip=True)
train_generator = datagen.flow_from_directory(train_folder, color_mode='rgb',target_size=(img_width,img_height),batch_size=5,classes=['bed','chair','table','wardrobe'])


model = Sequential()

model.add(ResNet50(include_top=False,pooling = 'avg', weights='imagenet' ))
model.layers[0].trainable = False
model.add(Dense(4, activation='softmax'))
model.summary()

#Training
valid_generator = datagen.flow_from_directory(valid_folder,color_mode='rgb',target_size=(img_width,img_height),batch_size=2,classes=['bed','chair','table','wardrobe'])

model.compile(Adam(lr=0.0001),loss='categorical_crossentropy',metrics=['accuracy'])
model.fit_generator(train_generator, validation_data=valid_generator, epochs=6, steps_per_epoch = 3, validation_steps= 2  )
