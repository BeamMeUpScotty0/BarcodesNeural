
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model


img_width, img_height = 100, 100
batch_size=10
test_dir = '/home/kris/Рабочий стол/Dataset/Train/test/'
nb_test_samples = 50

model=load_model('/home/kris/BarcodesNeural/detect.model')
test_datagen =ImageDataGenerator(rescale=1./255)

test_generator=test_datagen.flow_from_directory(
        test_dir,
        target_size=(img_width, img_height),
        batch_size=1,
        class_mode='binary')


print("model evaluate")

scores=model.evaluate_generator(test_generator, nb_test_samples//10)
print(scores)
print('Test score:', scores[0])
print("test accuracy: %.2f%%"%(scores[1]*100))
#a=model.predict_generator(test_generator, nb_test_samples)
#print("Prediction: ", a)


