
# coding: utf-8

# In[ ]:

#Behavioral cloning project


# In[1]:

import csv
import cv2
import numpy as np

import sklearn
from sklearn.model_selection import train_test_split

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Dropout, Lambda, Cropping2D
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.optimizers import Adam

#Resize function to reshape cropped images to INVIDIA architecture input
def resize_func(input):        
    from keras.backend import tf as ktf
    return ktf.image.resize_images(input, (66, 200))


# In[2]:

lines = []
with open('Trainning_data_IK/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)
        
train_data, validation_data = train_test_split(lines, test_size = 0.2)  


# In[13]:

def generator(lines, batch_size=32):
    num_data = len(lines)
    while 1:
        for offset in range(0,num_data, batch_size):
            batch_data = lines[offset:offset+batch_size]
            
            images = []
            measurements = []
            for line in batch_data:
                for i in range(3):
                    source_path = line[i]
                    image = cv2.imread(source_path)
                    b,g,r = cv2.split(image)       # get b,g,r channels
                    image = cv2.merge([r,g,b])     # switch it to rgb 
                    images.append(image)
                    measurement = float(line[3])
                    if i==0:
                        measurements.append(measurement)
                    else:
                        if i ==1:
                            measurements.append(measurement+0.23)
                        else:
                            measurements.append(measurement-0.23)
        
            aug_images, aug_measurements = [], []
            for image,measurement in zip(images, measurements):
                img = image
                aug_images.append(img)
                aug_measurements.append(measurement)
                aug_images.append(cv2.flip(img,1))
                aug_measurements.append(-measurement)
            
            X_train = np.array(aug_images)
            y_train = np.array(aug_measurements)
            yield sklearn.utils.shuffle(X_train, y_train)
            


# In[14]:

train_gen = generator(train_data,batch_size=32)
validation_gen = generator(validation_data,batch_size=32)


# In[15]:

model = Sequential()
# Image processing, normalization, cropping and resizeing
model.add(Lambda( lambda x: x/255.0 - 0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((60,20),(0,0))))
model.add(Lambda(resize_func))

# NVIDIA Architecture implementation with RELU activation an dropout layer to prevent overfitting

# 5x5 convolution layers, 2x2 stride and Valid padding
model.add(Convolution2D(24, 5, 5, subsample=(2, 2), border_mode='valid'))
model.add(Activation('relu'))
model.add(Convolution2D(36, 5, 5, subsample=(2, 2), border_mode='valid'))
model.add(Activation('relu'))
model.add(Convolution2D(48, 5, 5, subsample=(2, 2), border_mode='valid'))
model.add(Activation('relu'))

# 3x3 convolution layers, Valid Padding
model.add(Convolution2D(64, 3, 3, border_mode='valid'))
model.add(Activation('relu'))
model.add(Convolution2D(64, 3, 3, border_mode='valid'))
model.add(Activation('relu'))

# Flatten layer
model.add(Flatten())

# Fully connected layers, depths: 100, 50, 10, RELU activation 
model.add(Dense(100))
model.add(Activation('relu'))
model.add(Dropout(0.50))     #Dropout layer with 0.5 keep prob to prevent overfitting
model.add(Dense(50))
model.add(Activation('relu'))
model.add(Dense(10))
model.add(Activation('relu'))

# Output layer
model.add(Dense(1))

# Compile with adam optimizer and mean square error loss function, 
model.compile(optimizer='adam', loss='mse')

# Train with generators
model.fit_generator(train_gen, samples_per_epoch=len(train_data)*6, validation_data=validation_gen, nb_val_samples=len(validation_data)*6, nb_epoch=3)

#Save model
model.save('model.h5')

