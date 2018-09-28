
# coding: utf-8

import numpy as np

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, model_from_json
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D
from keras.layers import ZeroPadding2D
from keras.layers import Input
from keras.layers.pooling import MaxPooling2D
from keras.layers import GlobalAveragePooling2D

from keras.models import Model

from keras.optimizers import SGD
from keras.utils import np_utils
from keras.callbacks import LearningRateScheduler, ModelCheckpoint
from keras import backend as K
from keras.models import load_model
K.set_image_data_format('channels_first')

from input_dataset import read_gtsrb_dataset

NUM_CLASSES = 43

def cnn_model():

    inputs = Input(shape=(3,None,None))

    x = (Conv2D(32, (3, 3), padding='same',
            #input_shape=(3, IMG_SIZE, IMG_SIZE),
             activation='relu'))(inputs)
    x = (Conv2D(32, (3, 3), activation='relu'))(x)
    x = (MaxPooling2D(pool_size=(2, 2)))(x)
    x = (Dropout(0.2))(x)

    x = (Conv2D(64, (3, 3), padding='same',
             activation='relu'))(x)
    x = (Conv2D(64, (3, 3), activation='relu'))(x)
    x = (MaxPooling2D(pool_size=(2, 2)))(x)
    x = (Dropout(0.2))(x)

    x = (Conv2D(128, (3, 3), padding='same',
             activation='relu'))(x)
    x = (Conv2D(128, (3, 3), activation='relu'))(x)
    x = (MaxPooling2D(pool_size=(2, 2)))(x)
    x = (Dropout(0.2))(x)

    #x = (Flatten())(x)
    x = (GlobalAveragePooling2D())(x)
    x = (Dense(512, activation='relu'))(x)
    x = (Dropout(0.5))(x)
    x = (Dense(NUM_CLASSES, activation='softmax'))(x)

    model = Model(inputs=inputs, outputs=x)
    return model


def lr_schedule(epoch):
    return lr*(0.1**int(epoch/10))

batch_size = 32
nb_epoch = 5
model = cnn_model()
imag_sched = [24, 36, 48];

for IMG_SIZE in imag_sched:
    X, Y, X_val, Y_val, X_test, Y_test = read_gtsrb_dataset(IMG_SIZE=IMG_SIZE)

    # let's train the model using SGD + momentum (how original).
    lr = 0.01
    sgd = SGD(lr=lr, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    model.fit(X, Y,
              batch_size=batch_size,
              epochs=nb_epoch,
              validation_split=0.0,
              validation_data=(X_val, Y_val),
              shuffle=True,
              callbacks=[LearningRateScheduler(lr_schedule),
                        ModelCheckpoint('model.h5',save_best_only=True)]
                )

Y_pred = np.argmax(model.predict(X_test), axis=1)
acc = np.sum(y_pred==np.argmax(Y_test,axis=1))/np.size(Y_pred)
print("Test accuracy = {}".format(acc))


