from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense, Dropout
from keras.preprocessing.image import ImageDataGenerator


img_width, img_height = 100, 100
batch_size=100
train_dir = '/home/kris/Рабочий стол/Dataset/Train/train/'
val_dir = '/home/kris/Рабочий стол/Dataset/Train/val/'
test_dir = '/home/kris/Рабочий стол/Dataset/Train/test/'
nb_train_samples = 1000
nb_val_samples = 700
nb_test_samples=100
epochs = 7



model = Sequential()
model.add(Conv2D(32,(3,3),input_shape=(100,100,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(32,(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(64,(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dense(units=128,activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(units=1,activation='sigmoid'))




model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])


datagen=ImageDataGenerator(rescale=1./255, shear_range=0.2,rotation_range=90,zoom_range=0.2,horizontal_flip=True, vertical_flip=True)

train_generator=datagen.flow_from_directory(train_dir,
                                            target_size=(img_width, img_height),
                                            batch_size=batch_size, class_mode='binary')
val_generator=datagen.flow_from_directory(val_dir,
                                          target_size=(img_width,img_height),
                                          batch_size=batch_size, class_mode='binary')
test_generator=datagen.flow_from_directory(test_dir, target_size=(img_width, img_height), batch_size=batch_size, class_mode='binary')


testgen=ImageDataGenerator(rescale=1./255,  shear_range=0.2, zoom_range=0.2, horizontal_flip=True)

model.fit_generator(train_generator,steps_per_epoch=nb_train_samples//batch_size,epochs=epochs,
                    validation_data=val_generator,
                    validation_steps=nb_val_samples//batch_size)

scores=model.evaluate_generator(test_generator, nb_test_samples//batch_size)

print("Точность на тестовых данных: %.2f%%"%(scores[1]*100))
print(scores)
model.save('/home/kris/BarcodesNeural/detect.model')
print('Model saved')

model.save_weights("/home/kris/BarcodesNeural/weights.h5")
print("Weights saved")
