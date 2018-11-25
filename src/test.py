import os
import numpy as np
from PIL import Image
from keras.models import load_model
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

img_width, img_height = 100, 100
batch_size = 10
x_test_dir = '/home/kris/Рабочий стол/Dataset/Train/test/no/'
y_test_dir = '/home/kris/Рабочий стол/Dataset/Train/test/yes/'

x_test_sample = len(os.listdir(x_test_dir))
y_test_sample = len(os.listdir(y_test_dir))


x_test = np.zeros([x_test_sample + y_test_sample, 100, 100, 3])

test_file_list = []

for index,filename in enumerate(os.listdir(x_test_dir)):
    img = Image.open(x_test_dir + filename)
    test_file_list.append(filename)
    img = img.resize((100, 100), Image.ANTIALIAS)
    im = np.array(img)
    x_test[index, :, :, :] = im


for index,filename in enumerate(os.listdir(x_test_dir)):
    img = Image.open(x_test_dir + filename)
    test_file_list.append(filename)
    img = img.resize((100, 100), Image.ANTIALIAS)
    im = np.array(img)
    x_test[index, :, :, :] = im

x_test = x_test/255

y_test = np.array([])
y_test = np.append(np.append(y_test, [1]*x_test_sample), [0]*y_test_sample)


model = load_model('/home/kris/BarcodesNeural/detect.model')


print("Model evaluate...")
score = model.evaluate(x_test, y_test)
print(score)
print('Test loss:', score[0])
print("Test accuracy: %.2f%%"%(score[1]*100))
# a=model.predict(x_test)
# print("Prediction: ", a)


