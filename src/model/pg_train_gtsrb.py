
# coding: utf-8

# In[1]:


import numpy as np
from skimage import io, color, exposure, transform
from sklearn.cross_validation import train_test_split
import os
import glob
import h5py

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

from matplotlib import pyplot as plt
# get_ipython().run_line_magic('matplotlib', 'inline')

NUM_CLASSES = 43


# ## Function to preprocess the image:

# In[2]:


def preprocess_img(img, img_size):
    # Histogram normalization in y
    hsv = color.rgb2hsv(img)
    hsv[:,:,2] = exposure.equalize_hist(hsv[:,:,2])
    img = color.hsv2rgb(hsv)

    # central scrop
    min_side = min(img.shape[:-1])
    centre = img.shape[0]//2, img.shape[1]//2
    img = img[centre[0]-min_side//2:centre[0]+min_side//2,
              centre[1]-min_side//2:centre[1]+min_side//2,
              :]

    # rescale to standard size
    img = transform.resize(img, (img_size, img_size))

    # roll color axis to axis 0
    img = np.rollaxis(img,-1)

    return img


def get_class(img_path):
    return int(img_path.split('/')[-2])

def create_h5(root_dir, h5_fname='X.h5'):
    all_img_paths = glob.glob(os.path.join(root_dir, '*/*.ppm'))
    np.random.shuffle(all_img_paths)
    imgs = []
    labels = []
    for img_path in all_img_paths:
        try:
            img = preprocess_img(io.imread(img_path), IMG_SIZE)
            label = get_class(img_path)
            imgs.append(img)
            labels.append(label)

            if len(imgs)%1000 == 0: print("Processed {}/{}".format(len(imgs), len(all_img_paths)))
        except (IOError, OSError):
            print('missed', img_path)
            pass

    X = np.array(imgs, dtype='float32')
    Y = np.eye(NUM_CLASSES, dtype='uint8')[labels]

    with h5py.File(h5_fname,'w') as hf:
        hf.create_dataset('imgs', data=X)
        hf.create_dataset('labels', data=Y)
    return (X,Y)

# # Define Keras model

# In[5]:

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

# ## Preprocess all training images into a numpy array

# In[3]:

batch_size = 32
nb_epoch = 15

model = cnn_model()

imag_sched = [24, 36, 48];
for IMG_SIZE in imag_sched:
    try:
        with  h5py.File('X.h5.'+str(IMG_SIZE)) as hf:
            X, Y = hf['imgs'][:], hf['labels'][:]
        print("Loaded images from X.h5."+str(IMG_SIZE))

    except (IOError,OSError, KeyError):
        print("Error in reading X.h5.IMG_SIZE Processing all images...")
        root_dir = '/home/ubuntu/datasets/GTSRB/train/'
        X, Y = create_h5(root_dir, h5_fname = 'X.h5.'+str(IMG_SIZE))

    try:
        with  h5py.File('X_val.h5.'+str(IMG_SIZE)) as hf:
            X_val, Y_val = hf['imgs'][:], hf['labels'][:]
        print("Loaded images from X_val.h5."+str(IMG_SIZE))

    except (IOError,OSError, KeyError):
        print("Error in reading X_val.h5.IMG_SIZE Processing all images...")
        root_dir = '/home/ubuntu/datasets/GTSRB/valid/'
        create_h5(root_dir, h5_fname = 'X_val.h5.'+str(IMG_SIZE))

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

# # Load Test data

# In[7]:


import pandas as pd
test = pd.read_csv('/home/ubuntu/datasets/GTSRB/GT-final_test.csv',sep=';')

X_test = []
y_test = []
i = 0
for file_name, class_id  in zip(list(test['Filename']), list(test['ClassId'])):
    img_path = os.path.join('/home/ubuntu/datasets/GTSRB/test/',file_name)
    X_test.append(preprocess_img(io.imread(img_path), 48))
    y_test.append(class_id)

X_test = np.array(X_test)
y_test = np.array(y_test)


# In[8]:


#y_pred = model.predict_classes(X_test)
y_pred = np.argmax(model.predict(X_test), axis=1)
acc = np.sum(y_pred==y_test)/np.size(y_pred)
print("Test accuracy = {}".format(acc))


