#!/usr/bin/env python
# coding: utf-8

from PIL import Image
import tensorflow as tf
import pandas as pd
import numpy as np
from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.optimizers import Adam, SGD
from keras.utils import np_utils
from sklearn import metrics
from keras.layers import Activation, Dense, Dropout, Conv2D, Flatten, MaxPooling2D, GlobalMaxPooling2D, GlobalAveragePooling1D, AveragePooling2D, Input, Add
import os, sys
from keras.preprocessing.image import ImageDataGenerator


if (len(sys.argv) > 1):
    PREPROCESS_MODE = int(sys.argv[1])
else:
    PREPROCESS_MODE = 0

if PREPROCESS_MODE == 0:
    print("FFT MODE")
    #FOR FFT:
    width, height = 432, 288
    train_path = "spec_imgs/train"
    test_path = "spec_imgs/test"
    num_train_samples = len(os.listdir("spec_imgs/train/cats")) + len(os.listdir("spec_imgs/train/dogs"))
    num_test_samples = len(os.listdir("spec_imgs/test/cats")) + len(os.listdir("spec_imgs/test/dogs"))

    n_epochs = 100
    learning_rate = 1e-3 #1e-4
    n_filters = [16, 32, 64]

else:
    # FOR GAF
    width, height = 50, 50
    train_path = "output/train"
    test_path = "output/test"
    arr_train = os.listdir(train_path)
    num_train_samples = len(os.listdir("output/train/cat")) + len(os.listdir("output/train/dog"))
    num_test_samples = len(os.listdir("output/test/cat")) + len(os.listdir("output/test/dog"))

    n_epochs = 500
    learning_rate = 5e-6
    n_filters = [8, 16, 32]

# Generate batches of tensor image data with real-time data augmentation.
train_datagen = ImageDataGenerator(
        rescale=1./255, 
        shear_range=0.2, 
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)


# Set batch size for test generator to a number that divides into your 
# num_test_samples exactly.
# In this case num_test_samples = 67, so we will use 1 
batch_size_test = 1

training_set = train_datagen.flow_from_directory(
        train_path,
        target_size=(width, height),
        batch_size=15,
        class_mode='categorical',
        shuffle = False)

testing_set = test_datagen.flow_from_directory(
        test_path,
        target_size=(width, height),
        batch_size=batch_size_test,
        class_mode='categorical',
        shuffle = False)



# define your custom callback for prediction
# from: https://stackoverflow.com/questions/36864774/python-keras-how-to-access-each-epoch-prediction
class PredictionCallback(tf.keras.callbacks.Callback):    
  def on_epoch_end(self, epoch, logs={}):
      if epoch % 10 == 0:
        testing_set.reset()
        pred = self.model.predict_generator(testing_set, steps=num_test_samples) #.flatten()
        classes = np.argmax(pred, axis=1)#[(1 if (x > 0.5) else 0) for x in pred]
        for i in range(len(classes)):
            print('value: {:.4f}\t{:.4f}\tclass: {}'.format(pred[i][0], pred[i][1], classes[i]))
    



    
# Convolutional network
if K.image_data_format() == 'channels_first':
    input_shape = (3, width, height)
else:
    input_shape = (width, height, 3)

model = Sequential()

# 1st layer
model.add(Conv2D(n_filters[0], (2, 2), strides=(2, 2), input_shape=input_shape))
model.add(AveragePooling2D((2, 2), strides=(2,2)))
model.add(Dropout(rate=0.2))
model.add(Activation('relu'))

# 2nd layer
model.add(Conv2D(n_filters[1], (2, 2), padding="same"))
model.add(AveragePooling2D((2, 2), strides=(2,2)))
model.add(Dropout(rate=0.2))
model.add(Activation('relu'))

# 3rd hidden layer
model.add(Conv2D(n_filters[2], (2, 2), padding="same"))
model.add(AveragePooling2D((2, 2), strides=(2,2)))
model.add(Dropout(rate=0.2))
model.add(Activation('relu'))

# Flatten
model.add(Flatten())
model.add(Dropout(rate=0.5))

# Fully connected layer
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(rate=0.5))

# Output layer
model.add(Dense(2))
model.add(Activation('softmax'))


model.compile(loss="binary_crossentropy", optimizer=SGD(learning_rate=learning_rate), metrics=['accuracy'])
print(model.summary())



model.fit(
        training_set,
        epochs=n_epochs,
        validation_data=testing_set,
        validation_steps=67,
        callbacks=[PredictionCallback()])



# Now that the model is trained, evaluate it
# OUTPUT: [LOSS, ACCURACY]

# Testing accuracy
print(model.evaluate(x=testing_set))


# Training accuracy
print(model.evaluate(x=training_set))


testing_set.reset()
pred = model.predict(testing_set, steps=num_test_samples, verbose=1)
predicted_class_indices = np.argmax(pred, axis=1)#[(1 if (x > 0.5) else 0) for x in pred.flatten()]
labels = (training_set.class_indices)
labels = dict((v,k) for k,v in labels.items())
predictions = [labels[k] for k in predicted_class_indices]
print(predictions)
filenames=testing_set.filenames
results=pd.DataFrame({"Filename":filenames,
                      "Predictions":predictions})
results.to_csv("prediction_results.csv",index=False)




