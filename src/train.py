from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense, Dropout
from keras.preprocessing.image import ImageDataGenerator
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

img_width, img_height = 100, 100
batch_size = 10

train_dir = '/home/kris/Рабочий стол/Dataset/Train/train/'
val_dir = '/home/kris/Рабочий стол/Dataset/Train/val/'

nb_train_samples = 1000
nb_val_samples = 300

epochs = 5



model = Sequential()
model.add(Conv2D(8, (3, 3), input_shape=(100, 100, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(12, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(16, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(units=512, activation='relu'))
model.add(Dropout(0.7))
model.add(Dense(units=1, activation='sigmoid'))

# model.summary()
# input('Wait...')

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


datagen = ImageDataGenerator(rescale=1./255,
                             # rotation_range=20,
                             # width_shift_range=0.2,
                             # height_shift_range=0.2,
                             horizontal_flip=True)

train_generator = datagen.flow_from_directory(train_dir,
                                              target_size=(img_width, img_height),
                                              batch_size=batch_size,
                                              class_mode='binary')
val_generator = datagen.flow_from_directory(val_dir,
                                            target_size=(img_width,img_height),
                                            batch_size=batch_size,
                                            class_mode='binary')

model.fit_generator(train_generator,
                    steps_per_epoch=nb_train_samples//batch_size,
                    epochs=epochs,
                    validation_data=val_generator,
                    validation_steps=nb_val_samples//batch_size)


model.save('/home/kris/BarcodesNeural/detect.model')
print('Model saved')

model.save_weights("/home/kris/BarcodesNeural/weights.h5")
print("Weights saved")
