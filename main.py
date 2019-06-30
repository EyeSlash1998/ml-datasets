# -*- coding: utf-8 -*-

# Building CNN
 #Import libaries
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
 
 #from keras.layers import Convolution2D
model = Sequential()

#Convolution Layer 1
model.add(Conv2D(32, (3, 3), input_shape=(64, 64, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())

#Complete Connected model
model.add(Dense(128, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

#Fitting the CNN to the image
from keras.preprocessing.image import ImageDataGenerator

train_dataset = ImageDataGenerator(
    rescale = 1./255,
    shear_range = 0.2,
    zoom_range = 0.3,
    horizontal_flip = True
    )

test_dataset = ImageDataGenerator(rescale = 1./255)

train_gen = train_dataset.flow_from_directory(
    'dataset/training_set',
    target_size = (64,64),
    shuffle = True,
    batch_size = 32,
    class_mode = 'binary'
    )

test_gen = test_dataset.flow_from_directory(
    'dataset/test_set',
    target_size = (64, 64),
    batch_size = 32,
    class_mode = 'binary'
    )

model.fit_generator(
    train_gen,
    steps_per_epoch = 8000,
    epochs = 25,
    validation_data = test_gen,
    validation_steps = 2000
    )




















