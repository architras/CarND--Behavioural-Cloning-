
# coding: utf-8

# In[1]:


# Importing all the packages
import csv
import cv2
import numpy as np
import matplotlib.pyplot as plt


from keras.models import Sequential
from keras.layers import Lambda, Cropping2D
from keras.layers.core import Dense, Activation, Flatten, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D

# Reading data from the csv data file- Udacity data

lines=[]
with open('data/driving_log.csv') as csvfile:
    reader= csv.reader(csvfile)
    next(reader, None)
    for line in reader:
        lines.append(line)
 
images =[]
measurements=[]



# In[2]:


# These are from Vivek Yadav's post : https://chatbotslife.com/using-augmentation-to-mimic-human-driving-496b569760a9



# Function to take in an image and apply translations in X and Y for specified range. Also change the steer accordingly
# Input: Image, current steer angle, translation range for X
# Output: Modified image and steer angle

def trans_image(image,steer,trans_range):
    # Translation
    rows,cols,channels= image.shape
    tr_x = trans_range*np.random.uniform()-trans_range/2
    steer_ang = steer + tr_x/trans_range*2*.2
    tr_y = 10*np.random.uniform()-10/2
    Trans_M = np.float32([[1,0,tr_x],[0,1,tr_y]])
    image_tr = cv2.warpAffine(image,Trans_M,(cols,rows))
    
    return image_tr,steer_ang

# Function to randomly change the brightness of the image
# Input: Image
# Output: Modified image 

def augment_brightness_camera_images(image):
    image1 = cv2.cvtColor(image,cv2.COLOR_RGB2HSV)
    random_bright = .25+np.random.uniform()
    image1[:,:,2] = image1[:,:,2]*random_bright
    image1 = cv2.cvtColor(image1,cv2.COLOR_HSV2RGB)
    return image1


# In[3]:


# Generating the training dataset
# This reads in image one by one, randomly takes either one of center, left or right camera image
# Apply a random translation through tans_image() and change the brightness through augment_brightness_camera_images() 
for row in lines:
    # create adjusted steering measurements for the side camera images
    steering_center = float(row[3])
    correction = 0.25 # this is a parameter to tune
    steering_left = steering_center + correction
    steering_right = steering_center - correction
    current_path='data/'
    
    i_lrc = np.random.randint(3)
    if (i_lrc == 0):
        image = cv2.imread(current_path + row[1].lstrip(' '))[...,::-1]
        steer= steering_left
    if (i_lrc == 1):
        image = cv2.imread(current_path + row[0].lstrip(' '))[...,::-1]
        steer= steering_center
    if (i_lrc == 2):
        image = cv2.imread(current_path + row[2].lstrip(' '))[...,::-1]
        steer= steering_right

    image,steer= trans_image(image,steer,150)
    image=augment_brightness_camera_images(image) 
    images.append(image)
    measurements.append(steer*1.5)

    
# Adding the flipped images to make the data more uniform 
augmented_images, augmented_measurements = [], []

for image, measurement in zip(images,measurements):
    augmented_images.append(image)
    augmented_measurements.append(measurement)
    augmented_images.append(cv2.flip(image,1))
    augmented_measurements.append(measurement*-1)    

X_train= np.array(augmented_images)
y_train= np.array(augmented_measurements)






# In[4]:


#Lenet with Dropout
def Lenet_mod():
    model= Sequential()
    model.add(Lambda(lambda x: (x/255.0)-0.5, input_shape=(160,320,3)))
    model.add(Cropping2D(cropping=((50,20), (0,0))))

    model.add(Convolution2D(6, 5, 5))
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2, 2)))

    model.add(Convolution2D(16, 5, 5))
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2, 2)))

    model.add(Flatten())
    model.add(Dense(120))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))


    model.add(Dense(84))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1,activation='linear'))
    return model

# Nvidia network


def nvidia_mod():
    model= Sequential()

    model.add(Lambda(lambda x: (x/255.0)-0.5, input_shape=(160,320,3)))
    model.add(Cropping2D(cropping=((50,20), (0,0))))

    model.add(Convolution2D(24, 5, 5))
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2, 2)))

    model.add(Convolution2D(36, 5, 5))
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2, 2)))

    model.add(Convolution2D(48, 5, 5))
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2, 2)))

    model.add(Dropout(0.5))

    model.add(Convolution2D(64, 3, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2, 2)))

    model.add(Flatten())
    model.add(Dense(1164, activation='linear'))
    model.add(Dropout(.5))
    model.add(Dense(100, activation='linear'))
    model.add(Dropout(.2))
    model.add(Dense(50, activation='linear'))
    model.add(Dropout(.1))
    model.add(Dense(10, activation='linear'))
    model.add(Dense(1, activation='linear'))
    return model



# In[5]:



# Using Nvidia model
model = nvidia_mod()
# providing the summary
model.summary()

#using mse loss and adam optimizer
model.compile(loss='mse', optimizer='adam')

# Train -Validation split of 80-20
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=5)

# saving the model          
model.save('model.h5')
    


