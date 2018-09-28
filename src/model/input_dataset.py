import csv
import numpy as np
import random
random.seed(1234)

import glob
import h5py
import os
import numpy as np
from skimage import io, color, exposure, transform

NUM_CLASSES = 43

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

def create_h5(root_dir, h5_fname='X.h5', IMG_SIZE=48):
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

def read_gtsrb_dataset(test_prop = 0.2, IMG_SIZE=48):
    X, Y, X_val, Y_val = [], [], [], []
    try:
        with  h5py.File('X.h5.'+str(IMG_SIZE)) as hf:
            X, Y = hf['imgs'][:], hf['labels'][:]
        print("Loaded images from X.h5."+str(IMG_SIZE))

    except (IOError,OSError, KeyError):
        print("Error in reading X.h5.IMG_SIZE Processing all images...")
        root_dir = '/home/ubuntu/datasets/GTSRB/train/'
        X, Y = create_h5(root_dir, h5_fname = 'X.h5.'+str(IMG_SIZE), IMG_SIZE=IMG_SIZE)

    try:
        with  h5py.File('X_val.h5.'+str(IMG_SIZE)) as hf:
            X_val, Y_val = hf['imgs'][:], hf['labels'][:]
        print("Loaded images from X_val.h5."+str(IMG_SIZE))

    except (IOError,OSError, KeyError):
        print("Error in reading X_val.h5.IMG_SIZE Processing all images...")
        root_dir = '/home/ubuntu/datasets/GTSRB/valid/'
        X_val, Y_val = create_h5(root_dir, h5_fname = 'X_val.h5.'+str(IMG_SIZE), IMG_SIZE=IMG_SIZE)

    import pandas as pd
    test = pd.read_csv('/home/ubuntu/datasets/GTSRB/GT-final_test.csv',sep=';')

    X_test = []
    Y_test = []
    i = 0
    for file_name, class_id  in zip(list(test['Filename']), list(test['ClassId'])):
        img_path = os.path.join('/home/ubuntu/datasets/GTSRB/test/',file_name)
        X_test.append(preprocess_img(io.imread(img_path), 48))
        Y_test.append(class_id)
        i+=1
        if i >800:
            break # temporary to reduce run time

    X_test = np.array(X_test)
    Y_test = np.eye(NUM_CLASSES, dtype='uint8')[Y_test] # one hot code Y_test

    print ("X_TRAIN : ", X.shape)
    print ("Y_TRAIN : ", Y.shape)
    print ("X_val : ", X_val.shape)
    print ("Y_val : ", Y_val.shape)
    print ("X_TEST : ", X_test.shape)
    print ("Y_TEST : ", Y_test.shape)
    return X, Y, X_val, Y_val, X_test, Y_test


