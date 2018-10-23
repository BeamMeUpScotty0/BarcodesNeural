#!/usr/bin/env python
# -*- coding: utf-8 -*-
from PIL import Image
import matplotlib as plt
import numpy as np
import os
import timeit
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense


#paths to find train and test data
train_1_dir = '/home/kris/Рабочий стол/white/1/'
train_5_dir = '/home/kris/Рабочий стол/white/5/'
train_7_dir = '/home/kris/Рабочий стол/white/7/'


test_black_dir = '/home/kris/Рабочий стол/white/are/'
test_no_dir = '/home/kris/Рабочий стол/white/not/'

#get data set count
train_1_sample = len(os.listdir(train_1_dir))
train_5_sample = len(os.listdir(train_5_dir))
train_7_sample = len(os.listdir(train_7_dir))

test_black_sample = len(os.listdir(test_black_dir))
test_no_sample = len(os.listdir(test_no_dir))


print("Train Set samples- 1-"+str(train_1_sample)+" 2-"+str(train_5_sample)+" 3-"+str(train_7_sample))



print("Test Set samples- HAND-"+str(test_black_sample)+" NON_HAND-"+str(test_no_sample))

train_x_data_set=np.zeros([train_1_sample+train_5_sample+train_7_sample,100,100,3])
print("shape of training data set: "+ str(train_x_data_set.shape))




#load images containing hand in train_x_data_set matrix
for index,filename in enumerate(os.listdir(train_1_dir)):
    img = Image.open(train_1_dir+filename)
    img = img.resize((100,100),Image.ANTIALIAS)
    im = np.array(img)
    train_x_data_set[index, :, :, :] = im

for index,filename in enumerate(os.listdir(train_5_dir)):
    img = Image.open(train_5_dir+filename)
    img = img.resize((100,100),Image.ANTIALIAS)
    im = np.array(img)
    train_x_data_set[index, :, :, :] = im

for index,filename in enumerate(os.listdir(train_7_dir)):
    img = Image.open(train_7_dir+filename)
    img = img.resize((100,100),Image.ANTIALIAS)
    im = np.array(img)
    train_x_data_set[index, :, :, :] = im



train_x_data_set = train_x_data_set/255
print(train_x_data_set)


train_y_data_set=np.array([])

train_y_data_set=np.append(np.append(np.append(train_y_data_set,[0.9]*train_1_sample),[0.6]*train_5_sample),[0.1]*train_7_sample)
print("shape of train label:"+str(train_y_data_set.shape))
print(train_y_data_set)



model = Sequential()
model.add(Conv2D(8, (3, 3), input_shape=(100,100,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2),strides=2))
model.add(Conv2D(12, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2),strides=2))
model.add(Conv2D(16, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dense(units=128, activation='relu'))
#model.add(Dense(units=1, activation='sigmoid'))
model.add(Dense(units=1, activation='sigmoid'))


model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

model.fit(train_x_data_set, train_y_data_set,epochs=50)

test_x_data_set=np.zeros([test_black_sample+test_no_sample,100,100,3])
test_file_list = []


for index,filename in enumerate(os.listdir(test_black_dir)):
    img = Image.open(test_black_dir+filename)
    test_file_list.append(filename)
    img = img.resize((100,100),Image.ANTIALIAS)
    im = np.array(img)
    test_x_data_set[index,:,:,:]=im

test_x_data_set = test_x_data_set/255

model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)

model.save_weights("model.h5")
print("Saved model to disk")

time_predictions=model.predict(test_x_data_set)
print(time_predictions)
