import numpy as np
import keras
from keras import backend as K
from keras.callbacks import  EarlyStopping, Callback
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from keras import optimizers
from keras.models import Sequential, Model
from keras.layers import  Input, Conv2D, add , MaxPooling2D,Activation,Dropout,Flatten,Dense,BatchNormalization,AveragePooling2D,ZeroPadding2D,merge,Reshape
from keras.optimizers import Adam, SGD

import sys
sys.setrecursionlimit(3000)

from scale_layer import Scale

### Image Flow Generator

class resnet101:
    def __init__ (self, width,height,channel,num_classes):
        self.width = width
        self.height = height
        self.channel = channel
        self.eps = 1.1e-5
        self.num_classes = num_classes
        self.bn_axis = 3

    def identity_block(self, input_tensor, kernel_size, filters, stage, block,strides=(2,2)):
        '''
        Identity Block : This block create convolutional layers that has no convolutional layer at shortcut, where shortcut is used if __name__ == '__main__':
            refrence to the paper describing Resnet Models : https://arxiv.org/pdf/1512.03385.pdf .

        #Args:
            input_tensor: input tensor
            kernel_size : (default value is 3 ) , the kernel size of middle convolutional layer in the Block
            filters : lists of integer , Number of filters
            stage : integer , current state label , used for generating layer names
            block: current block label , used for generating layer names
       #Output : a tensor flow
         '''

        nb_filter1, nb_filter2, nb_filter3 = filters

        conv_name_base = 'res101' + str(stage) + block + '_branch'
        bn_name_base = 'bn' + str(stage) + block + '_branch'
        scale_name_base = 'scale' + str(stage) + block + '_branch'

        t = Conv2D(nb_filter1,(1,1), name=conv_name_base + '2a', use_bias = False)(input_tensor)
        t = BatchNormalization(epsilon=self.eps, axis = self.bn_axis, name= bn_name_base + '2a' )(t)
        t = Scale(axis = self.bn_axis, name=scale_name_base + '2a')(t)
        t = Activation('relu', name=conv_name_base + '2a_relu' )(t)

        t = ZeroPadding2D((1,1),name=conv_name_base + '2b_zeropadding')(t)
        t = Conv2D(nb_filter2,(kernel_size,kernel_size), use_bias=False,name=conv_name_base + '2b' )(t)
        t = BatchNormalization(epsilon=self.eps, axis= self.bn_axis, name=bn_name_base + '2b')(t)
        t = Scale(axis = self.bn_axis, name=scale_name_base + '2b')(t)
        t = Activation('relu', name=conv_name_base + '2b_relu')(t)

        t = Conv2D(nb_filter3,(1,1), use_bias=False,name=conv_name_base + '2c' )(t)
        t = BatchNormalization(epsilon=self.eps, axis= self.bn_axis, name=bn_name_base + '2c')(t)
        t = Scale(axis = self.bn_axis, name=scale_name_base + '2c')(t)

        t = add([t,input_tensor])
        t = Activation('relu', name='res' + str(stage) + block +'_relu')(t)
        return t

    def conv_block(self, input_tensor, kernel_size, filters, stage, block,strides=(2,2)):
        '''
        Conv Block : This block create convolutional layers that has a convolutional layer at shortcut, where shortcut is used if __name__ == '__main__':
            refrence to the paper describing Resnet Models : https://arxiv.org/pdf/1512.03385.pdf .

        #Args:
            input_tensor: input tensor
            kernel_size : (default value is 3 ) , the kernel size of middle convolutional layer in the Block
            filters : lists of integer , Number of filters
            stage : integer , current state label , used for generating layer names
            block: current block label , used for generating layer names
       #Output : a tensor flow
         '''

        nb_filter1, nb_filter2, nb_filter3 = filters

        conv_name_base = 'res101' + str(stage) + block + '_branch'
        bn_name_base = 'bn' + str(stage) + block + '_branch'
        scale_name_base = 'scale' + str(stage) + block + '_branch'

        t = Conv2D(nb_filter1,(1,1), strides= strides , use_bias=False, name=conv_name_base + '2a')(input_tensor)
        t = BatchNormalization(epsilon=self.eps, axis = self.bn_axis, name= bn_name_base + '2a' )(t)
        t = Scale(axis = self.bn_axis, name=scale_name_base + '2a')(t)
        t = Activation('relu', name=conv_name_base + '2a_relu' )(t)

        t = ZeroPadding2D((1,1),name=conv_name_base + '2b_zeropadding')(t)
        t = Conv2D(nb_filter2,(kernel_size,kernel_size), use_bias=False,name=conv_name_base + '2b' )(t)
        t = BatchNormalization(epsilon=self.eps, axis= self.bn_axis, name=bn_name_base + '2b')(t)
        t = Scale(axis = self.bn_axis, name=scale_name_base + '2b')(t)
        t = Activation('relu', name=conv_name_base + '2b_relu')(t)

        t = Conv2D(nb_filter3,(1,1), use_bias=False,name=conv_name_base + '2c' )(t)
        t = BatchNormalization(epsilon=self.eps, axis= self.bn_axis, name=bn_name_base + '2c')(t)
        t = Scale(axis = self.bn_axis, name=scale_name_base + '2c')(t)

        shortcut = Conv2D(nb_filter3, (1,1), strides=strides,use_bias=False,  name=conv_name_base + '1' )(input_tensor)
        shortcut = BatchNormalization(epsilon=self.eps, axis= self.bn_axis, name=bn_name_base + '1')(shortcut)
        shortcut = Scale(axis=self.bn_axis,  name=scale_name_base + '1')(shortcut)

        t = add([t,shortcut])
        t = Activation('relu', name='res' + str(stage) + block +'_relu')(t)
        return t


    def ResNet101(self,color_type= 1, num_classes = None):
        '''
            ResNet101 Model for Keras

        '''
        img_input = Input(shape=(self.width,self.height, 3 ), name='data')

        t = ZeroPadding2D((3,3), name='conv1_zeropadding')(img_input)
        t = Conv2D(64, (7,7), strides = (2,2) , use_bias = False , name='conv1' )(t)
        t = BatchNormalization(epsilon = self.eps, axis= self.bn_axis, name='bn_conv1')(t)
        t = Scale(axis = self.bn_axis, name='scale_conv1')(t)
        t = Activation('relu', name= 'conv1_relu')(t)
        t = MaxPooling2D((3,3),strides = (2,2) , name= 'pool1')(t)

        t = self.conv_block(t, 3, [64,64,256], stage=2 , block= 'a' , strides=(1,1))
        t = self.identity_block(t,3,[64,64,256], stage=2, block='b')
        t = self.identity_block(t,3,[64,64,256], stage=2, block='c')

        t = self.conv_block(t, 3, [128,128,512], stage=3 , block= 'a' )
        for i in range(1,4):
            t = self.identity_block(t,3,[128,128,512], stage=3 , block = 'b'+str(i))

        t = self.conv_block(t,3,[256,256,1024], stage=4, block='a')
        for i in range(1,23):
            t = self.identity_block(t,3,[256,256,1024], stage = 4 , block='b' + str(i))

        t = self.conv_block(t,3,[512,512,2048], stage= 5, block= 'a')
        t = self.identity_block(t,3, [512,512,2048], stage=5, block= 'b')
        t = self.identity_block(t,3, [512,512,2048], stage=5, block= 'c')

        x_fc = AveragePooling2D((7,7), name= 'avg_pool')(t)
        x_fc = Flatten()(x_fc)
        x_fc = Dense(1000, activation='softmax', name='fc1000')(x_fc)


        #Pretrained weights
        weights_path = '../resnet101_weights_tf.h5'


        # Truncate and replace softmax layer for transfer learning
        # The method below works since pre-trained weights are stored in layers but not in the model

        x_newfc = AveragePooling2D((7,7), name='avg_pool')(t)
        x_newfc = Flatten()(x_newfc)
        x_newfc = Dense(self.num_classes, activation='relu', name= 'fc8')(x_newfc)

        model = Model(img_input, x_newfc)

        model.load_weights(weights_path, by_name= True)

        sgd = SGD(lr= 1e-3, decay=1e-6, momentum=0.9, nesterov= True)
        model.compile(optimizer=sgd , loss='categorical_crossentropy', metrics=['accuracy'])

        return model


res101 = resnet101(224,224,3,4)

train_folder = '../dataset/'
valid_folder = '../valid_dataset/'

img_width = 224
img_height = 224

datagen = ImageDataGenerator(rescale=1./255,rotation_range=5,zoom_range=0.2,horizontal_flip=True)
train_generator = datagen.flow_from_directory(train_folder, color_mode='rgb',target_size=(img_width,img_height),batch_size=5,classes=['bed','chair','table','wardrobe'])
valid_generator = datagen.flow_from_directory(valid_folder,color_mode='rgb',target_size=(img_width,img_height),batch_size=2,classes=['bed','chair','table','wardrobe'])

model = res101.ResNet101(4)
model.summary()
model.fit_generator(train_generator, validation_data=valid_generator, epochs=10, steps_per_epoch = 3,  validation_steps= 3  )
