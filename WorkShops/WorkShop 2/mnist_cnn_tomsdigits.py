'''Trains a simple convnet on the MNIST dataset.
Gets to 99.25% test accuracy after 12 epochs
(there is still a lot of margin for parameter tuning).
16 seconds per epoch on a GRID K520 GPU.
'''

from __future__ import print_function
import keras
from keras.datasets import mnist # type: ignore
from keras.models import Sequential # type: ignore
from keras.layers import Dense, Dropout, Flatten # type: ignore
from keras.layers import Conv2D, MaxPooling2D # type: ignore
from keras import backend as K
import sys
import cv2 # type: ignore
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

batch_size = 128
num_classes = 10
epochs = 12
img_rows, img_cols = 28, 28
TomsDigits = {}

def loadTomsDigits():
   for i in range(0, 10):
      img_ori = cv2.imread("TomsDigits/img-t"+str(i)+".png")
      # img_ori = cv2.imread("TomsDigits\img-t7.png")
      img_gray = cv2.cvtColor(img_ori, cv2.COLOR_BGR2GRAY)
      (thresh, bw) = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY)
      img_bw = cv2.bitwise_not(bw)
      resized = cv2.resize(img_bw, (img_rows, img_cols), interpolation = cv2.INTER_AREA)
      reshaped = np.reshape(resized, (-1, img_rows, img_cols, 1)) 
      # cv2.imshow('Black white image', reshaped)
      # cv2.waitKey(0)
      # cv2.destroyAllWindows()
      reshaped = reshaped.astype('float32')
      normalised = reshaped/255
      TomsDigits[str(i)] = normalised

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

if len(sys.argv)>1 and sys.argv[1] == "train":

   model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test))
   # model.save_weights("mnist-cnn-model.weights.h5")
   score = model.evaluate(x_test, y_test, verbose=0)
   print('Test loss:', score[0])
   print('Test accuracy:', score[1])   

elif len(sys.argv)>1 and sys.argv[1] == "test":

   model.load_weights("mnist-cnn-model.weights.h5")
   print("MODEL LOADED!")

   loadTomsDigits()
   print("DATA LOADED!")

   for i in range(0, 10):
      test_example = TomsDigits[str(i)]
      prediction = model.predict(test_example)
      best_class = np.argmax(prediction)
      print("i="+str(i)+" prediction="+str(prediction[0])+" best_class="+str(best_class))

else:
   print("DON'T KNOW WHAT TO DO ---> mnist_cnn.py [train|test]")


