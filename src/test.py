from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model

import cv2
img_width, img_height = 100, 100
test_dir = '/home/kris/Рабочий стол/Dataset/Train/test/'
nb_test_samples = 50
model=load_model('/home/kris/BarcodesNeural/detect.model')
test_datagen=ImageDataGenerator(rescale=1./255)


test_generator=test_datagen.flow_from_directory(
        test_dir,
        target_size=(img_width, img_height),
        batch_size=10,
        class_mode='binary')


print("model evaluate")
scores=model.evaluate_generator(test_generator, nb_test_samples)
print("Точность на тестовых данных: %.2f%%"%(scores[1]*100))
a=model.predict_generator(test_generator, nb_test_samples)
print("Prediction: ", a)



    testgen = ImageDataGenerator(rescale=1. / 255)
    now_generator = testgen.flow_from_directory(sys.argv[3], target_size=(150, 150), batch_size=1,
                                                class_mode='binary')
    a=int(1-(model.predict_generator(now_generator, 1//1)))
    print(a)
    if cv2.waitKey(1) & 0xFF == ord('q'):
	        break
cap.release()
cv2.destroyAllWindows()