from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import os
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

#paths to find train and test data
train_B_dir = '/home/kris/Рабочий стол/B/'
train_non_B_dir = '/home/kris/Рабочий стол/C/'
test_B_dir = '/home/kris/Рабочий стол/B (copy 1)/'
test_non_B_dir = '/home/kris/Рабочий стол/C (copy 1)/'

#get data set count
train_B_sample = len(os.listdir(train_B_dir))
train_non_B_sample =len(os.listdir(train_non_B_dir))
test_B_sample = len(os.listdir(test_B_dir))
test_non_B_sample = len(os.listdir(test_non_B_dir))

print("Train Set samples- B-"+str(train_B_sample)+" NON_B-"+str(train_non_B_sample))
print("Test Set samples- B-"+str(test_B_sample)+" NON_B-"+str(test_non_B_sample))

train_x_data_set=np.zeros([train_B_sample+train_non_B_sample,100,100,3])
print("shape of training data set: "+ str(train_x_data_set.shape))

#load images containing B in train_x_data_set matrix
for index,filename in enumerate(os.listdir(train_B_dir)):
    img = Image.open(train_B_dir+filename)
    img = img.resize((100,100),Image.ANTIALIAS)
    im = np.array(img)
    train_x_data_set[index,:,:,:]=im


#load images that does not contain B in train_x_data_set matrix
for index,filename in enumerate(os.listdir(train_non_B_dir)):
    img = Image.open(train_non_B_dir+filename)
    img = img.resize((100,100),Image.ANTIALIAS)
    im=np.array(img)
    train_x_data_set[train_B_sample+index,:,:,:]=im



train_x_data_set = train_x_data_set/255
print('hey')
#print(train_x_data_set)

train_y_data_set=np.array([])


train_y_data_set=np.append(np.append(train_y_data_set,[1]*train_B_sample),[0]*train_non_B_sample)
print("shape of train label:"+str(train_y_data_set.shape))
#print(train_y_data_set)

model = Sequential()
model.add(Conv2D(8,(3,3),input_shape=(100,100,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2),strides=2))
model.add(Conv2D(12,(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2),strides=2))
model.add(Conv2D(16,(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dense(units=128,activation='relu'))
model.add(Dense(units=1,activation='sigmoid'))



model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])



model.fit(train_x_data_set,train_y_data_set,epochs=10)

test_x_data_set=np.zeros([test_B_sample+test_non_B_sample,100,100,3])
test_file_list = []


for index,filename in enumerate(os.listdir(test_B_dir)):
    img = Image.open(test_B_dir+filename)
    test_file_list.append(filename)
    img = img.resize((100,100),Image.ANTIALIAS)
    im = np.array(img)
    test_x_data_set[index,:,:,:]=im



for index,filename in enumerate(os.listdir(test_non_B_dir)):
    img = Image.open(test_non_B_dir+filename)
    test_file_list.append(filename)
    img = img.resize((100,100),Image.ANTIALIAS)
    im = np.array(img)
    test_x_data_set[test_B_sample+index,:,:,:]=im



test_x_data_set = test_x_data_set/255


test_y_data_set=np.array([])
test_y_data_set=np.append(np.append(test_y_data_set,[1]*test_B_sample),[0]*test_non_B_sample)


time_predictions=model.predict(test_x_data_set)



model.save('/home/kris/BarcodesNeural/B_detect_24_feb.model')



model.evaluate(test_x_data_set,test_y_data_set)
import timeit


time_predictions=model.predict(test_x_data_set)
print(time_predictions)
print(model.evaluate(test_x_data_set,test_y_data_set))
for filename,predict in zip(test_file_list,time_predictions):
    print(filename+"-->"+str(predict))
    if(predict>=0.5000):
        print('It is a B')
    else:
        print("It's not a B")