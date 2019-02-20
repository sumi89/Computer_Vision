#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 30 17:42:40 2018

@author: sumi
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 22 14:22:46 2018

@author: sumi
"""



from keras import layers
from keras import models
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.utils import np_utils
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dense, Dropout, Flatten
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image

import matplotlib.pyplot as plt

import os, shutil

#Path to the directory where the original dataset was uncompressed
original_dataset_dir = '/Users/sumi/python/research/deep_learning_with_python/chap5/cats_n_dogs_original'

#Directory where you’ll store your smaller dataset
base_dir = '/Users/sumi/python/research/deep_learning_with_python/chap5/cats_n_dogs'
os.mkdir(base_dir)

#Directories for the training, validation, and test splits
train_dir = os.path.join(base_dir, 'train')
os.mkdir(train_dir)
validation_dir = os.path.join(base_dir, 'validation')
os.mkdir(validation_dir)
test_dir = os.path.join(base_dir, 'test')
os.mkdir(test_dir)

#Directory with training cat pictures
train_cats_dir = os.path.join(train_dir, 'cats')
os.mkdir(train_cats_dir)

#Directory with training dog pictures
train_dogs_dir = os.path.join(train_dir, 'dogs')
os.mkdir(train_dogs_dir)

#Directory with validation cat pictures
validation_cats_dir = os.path.join(validation_dir, 'cats')
os.mkdir(validation_cats_dir)

#Directory with validation dog pictures
validation_dogs_dir = os.path.join(validation_dir, 'dogs')
os.mkdir(validation_dogs_dir)

#Directory with test cat pictures
test_cats_dir = os.path.join(test_dir, 'cats')
os.mkdir(test_cats_dir)

#Directory with test dog pictures
test_dogs_dir = os.path.join(test_dir, 'dogs')
os.mkdir(test_dogs_dir)

#Copies the first 1,000 cat images to train_cats_dir
fnames = ['cat.{}.jpg'.format(i) for i in range(1000)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(train_cats_dir, fname)
    shutil.copyfile(src, dst)

#Copies 1,000 to 1500 cat images to validation_cats_dir
fnames = ['cat.{}.jpg'.format(i) for i in range(1000,1500)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(validation_cats_dir, fname)
    shutil.copyfile(src, dst)
    
#Copies 1500 to 2000 cat images to test_cats_dir    
fnames = ['cat.{}.jpg'.format(i) for i in range(1500,2000)]
for fnamem in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(test_cats_dir, fname)
    shutil.copyfile(src, dst)
    
#Copies the first 1,000 dog images to train_dogs_dir
fnames = ['dog.{}.jpg'.format(i) for i in range(1000)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(train_dogs_dir, fname)
    shutil.copyfile(src, dst)

#Copies 1,000 to 1500 dog images to validation_dogs_dir
fnames = ['dog.{}.jpg'.format(i) for i in range(1000,1500)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(validation_dogs_dir, fname)
    shutil.copyfile(src, dst)
    
#Copies 1500 to 2000 dog images to test_dogs_dir    
fnames = ['dog.{}.jpg'.format(i) for i in range(1500,2000)]
for fnamem in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(test_dogs_dir, fname)
    shutil.copyfile(src, dst)

#print('total training cat images:', len(os.listdir(train_cats_dir)))


## Instantiating a small convnet for dogs vs. cats classification
model = models.Sequential()
model.add(Conv2D(32, (3,3), activation = 'relu', input_shape = (150, 150, 3)))
model.add(MaxPooling2D((2,2)))
model.add(Conv2D(64, (3,3), activation = 'relu'))
model.add(MaxPooling2D((2,2)))
model.add(Conv2D(128, (3,3), activation = 'relu'))
model.add(MaxPooling2D((2,2)))
model.add(Conv2D(128, (3,3), activation = 'relu'))
model.add(MaxPooling2D((2,2)))
model.add(Flatten())
model.add(Dense(512, activation = 'relu'))
model.add(Dense(1, activation = 'sigmoid'))

model.summary()

## Configuring the model for training
model.compile(loss='binary_crossentropy', optimizer = optimizers.RMSprop(lr = 1e-4),
              metrics = ['acc'])

## Using ImageDataGenerator to read images from directories
### Rescales all images by 1/255
train_datagen = ImageDataGenerator(rescale = 1./255)
test_datagen = ImageDataGenerator(rescale = 1./255)

train_generator = train_datagen.flow_from_directory(train_dir, # Target directory
                                                    target_size = (150, 150), # Resizes all images to 150 × 150 
                                                    batch_size = 20, 
                                                    class_mode = 'binary') # Because you use binary_crossentropy loss, you need binary labels.

validation_generator = test_datagen.flow_from_directory(validation_dir, 
                                                        target_size = (150, 150),
                                                        batch_size = 20,
                                                        class_mode = 'binary')
## Fitting the model using a batch generator
history = model.fit_generator(train_generator, steps_per_epoch = 100, epochs = 20,
                              validation_data = validation_generator, validation_steps = 50)
## Saving the model
model.save('cats_and_dogs_small_1.h5')

## Displaying curves of loss and accuracy during training
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, 'bo', label = 'Training acc')
plt.plot(epochs, val_acc, 'b', label = 'Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()

plt.plot(epochs, loss, 'bo', label = 'Training loss')
plt.plot(epochs, val_loss, 'b', label = 'Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()




















 











