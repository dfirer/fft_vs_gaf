#!/usr/bin/env python
# coding: utf-8

# In[41]:


from PIL import Image
import tensorflow as tf
import pandas as pd
import numpy as np
from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.optimizers import Adam
from keras.utils import np_utils
from sklearn import metrics
from keras.layers import Activation, Dense, Dropout, Conv2D, Flatten, MaxPooling2D, GlobalMaxPooling2D, GlobalAveragePooling1D, AveragePooling2D, Input, Add
import os
from keras.preprocessing.image import ImageDataGenerator


# In[42]:


#FOR FFT:
width, height = 432, 288
train_path = "spec_imgs2/train"
test_path = "spec_imgs2/test"
num_train_samples = len(os.listdir("spec_imgs2/train/cat")) + len(os.listdir("spec_imgs2/train/dog"))
num_test_samples = len(os.listdir("spec_imgs2/test/cat")) + len(os.listdir("spec_imgs2/test/dog"))

# FOR GAF
#width, height = 50, 50
#train_path = "output/cat_dog/train"
#test_path = "output/cat_dog/test"
#arr_train = os.listdir(train_path)
#num_train_samples = len(os.listdir("output/cat_dog/train/cat")) + len(os.listdir("output/cat_dog/train/dog"))
#num_test_samples = len(os.listdir("output/cat_dog/test/cat")) + len(os.listdir("output/cat_dog/test/dog"))


# In[43]:


# Generate batches of tensor image data with real-time data augmentation.
train_datagen = ImageDataGenerator(
        rescale=1./255, 
        shear_range=0.2, 
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)


# In[44]:


# Set batch size for test generator to a number that divides into your 
# num_test_samples exactly.
# In this case num_test_samples = 67, so we will use 1 
batch_size_test = 1

training_set = train_datagen.flow_from_directory(
        train_path,
        target_size=(width, height),
        batch_size=15,
        class_mode='binary',
        shuffle = False)

testing_set = test_datagen.flow_from_directory(
        test_path,
        target_size=(width, height),
        batch_size=batch_size_test,
        class_mode='binary',
        shuffle = False)


# In[45]:


# Convolutional network
if K.image_data_format() == 'channels_first':
    input_shape = (3, width, height)
else:
    input_shape = (width, height, 3)

model = Sequential()

# 1st layer
model.add(Conv2D(32, (2, 2), strides=(2, 2), input_shape=input_shape))
model.add(AveragePooling2D((2, 2), strides=(2,2)))
model.add(Activation('relu'))

# 2nd layer
model.add(Conv2D(32, (2, 2), padding="same"))
model.add(AveragePooling2D((2, 2), strides=(2,2)))
model.add(Activation('relu'))

# 3rd hidden layer
model.add(Conv2D(32, (2, 2), padding="same"))
model.add(AveragePooling2D((2, 2), strides=(2,2)))
model.add(Activation('relu'))

# Flatten
model.add(Flatten())
model.add(Dropout(rate=0.5))

# Fully connected layer
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(rate=0.5))

# Output layer
model.add(Dense(1))
model.add(Activation('sigmoid'))


# In[46]:


model.compile(loss="binary_crossentropy", optimizer="sgd", metrics=['accuracy'])
print(model.summary())


# In[ ]:


model.fit_generator(
        training_set,
        epochs=10,
        validation_data=testing_set,
        validation_steps=67)


# In[ ]:


# Now that the model is trained, evaluate it
# OUTPUT: [LOSS, ACCURACY]

# Testing accuracy
print(model.evaluate_generator(generator=testing_set))


# In[ ]:


# Training accuracy
print(model.evaluate_generator(generator=training_set))


# In[ ]:


testing_set.reset()
pred = model.predict_generator(testing_set, steps=num_test_samples, verbose=1)
predicted_class_indices=np.argmax(pred,axis=1)
labels = (training_set.class_indices)
labels = dict((v,k) for k,v in labels.items())
predictions = [labels[k] for k in predicted_class_indices]
predictions = predictions[:70]
filenames=testing_set.filenames
results=pd.DataFrame({"Filename":filenames,
                      "Predictions":predictions})
results.to_csv("prediction_results.csv",index=False)


# In[ ]:




