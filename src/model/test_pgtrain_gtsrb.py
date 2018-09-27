#!/usr/bin/env python
# coding: utf-8

from fastai.imports import *
from fastai.transforms import *
from fastai.conv_learner import *
from fastai.model import *
from fastai.dataset import *
from fastai.sgdr import *
from fastai.plots import *
from fastai.metrics import *

import matplotlib.pyplot as plt
import csv
from collections import defaultdict, namedtuple
import os
import shutil
import pandas as pd
from sklearn.metrics import confusion_matrix
from collections import Counter

# Download and unpack the training set and the test set

# get_ipython().system(' wget http://benchmark.ini.rub.de/Dataset/GTSRB_Final_Training_Images.zip -P data')
# get_ipython().system(' wget http://benchmark.ini.rub.de/Dataset/GTSRB_Final_Test_Images.zip -P data')
# get_ipython().system(' wget http://benchmark.ini.rub.de/Dataset/GTSRB_Final_Test_GT.zip -P data')
# get_ipython().system(' unzip data/GTSRB_Final_Training_Images.zip -d data')
# get_ipython().system(' unzip data/GTSRB_Final_Test_Images.zip -d data')
# get_ipython().system(' unzip data/GTSRB_Final_Test_GT.zip -d data')
#
# # Move the test set to data/test
#
# get_ipython().system(' mkdir data/test')
# get_ipython().system(' mv data/GTSRB/Final_Test/Images/*.ppm data/test')
#
# # Download class names
# get_ipython().system(' wget https://raw.githubusercontent.com/georgesung/traffic_sign_classification_german/master/signnames.csv -P data')

def read_annotations(filename):
    annotations = []

    with open(filename) as f:
        reader = csv.reader(f, delimiter=';')
        next(reader) # skip header

        # loop over all images in current annotations file
        for row in reader:
            filename = row[0] # filename is in the 0th column
            label = int(row[7]) # label is in the 7th column
            annotations.append(Annotation(filename, label))

    return annotations

def load_training_annotations(source_path):
    annotations = []
    for c in range(0,43):
        filename = os.path.join(source_path, format(c, '05d'), 'GT-' + format(c, '05d') + '.csv')
        annotations.extend(read_annotations(filename))
    return annotations

def copy_files(label, filenames, source, destination, move=False):
    func = os.rename if move else shutil.copyfile

    label_path = os.path.join(destination, str(label))
    if not os.path.exists(label_path):
        os.makedirs(label_path)

    for filename in filenames:
        destination_path = os.path.join(label_path, filename)
        if not os.path.exists(destination_path):
            func(os.path.join(source, format(label, '05d'), filename), destination_path)

def split_train_validation_sets(source_path, train_path, validation_path, all_path, validation_fraction=0.2):
    """
    Splits the GTSRB training set into training and validation sets.
    """

    if not os.path.exists(train_path):
        os.makedirs(train_path)

    if not os.path.exists(validation_path):
        os.makedirs(validation_path)

    if not os.path.exists(all_path):
        os.makedirs(all_path)

    annotations = load_training_annotations(source_path)
    filenames = defaultdict(list)
    for annotation in annotations:
        filenames[annotation.label].append(annotation.filename)

    for label, filenames in filenames.items():
        filenames = sorted(filenames)

        validation_size = int(len(filenames) // 30 * validation_fraction) * 30
        train_filenames = filenames[validation_size:]
        validation_filenames = filenames[:validation_size]

        copy_files(label, filenames, source_path, all_path, move=False)
        copy_files(label, train_filenames, source_path, train_path, move=True)
        copy_files(label, validation_filenames, source_path, validation_path, move=True)

# Normal version

def open_image_normal(fn):
    """ Opens an image using OpenCV given the file path.

    Arguments:
        fn: the file path of the image

    Returns:
        The numpy array representation of the image in the RGB format
    """
    flags = cv2.IMREAD_UNCHANGED+cv2.IMREAD_ANYDEPTH+cv2.IMREAD_ANYCOLOR
    if not os.path.exists(fn):
        raise OSError('No such file or directory: {}'.format(fn))
    elif os.path.isdir(fn):
        raise OSError('Is a directory: {}'.format(fn))
    else:
        try:
            return cv2.cvtColor(cv2.imread(fn, flags), cv2.COLOR_BGR2RGB).astype(np.float32)/255
        except Exception as e:
            raise OSError('Error handling image at: {}'.format(fn)) from e

# Histogram equalization

def open_image_hist_eq(fn):
    """ Opens an image using OpenCV given the file path.

    Arguments:
        fn: the file path of the image

    Returns:
        The numpy array representation of the image in the RGB format
    """
    flags = cv2.IMREAD_UNCHANGED+cv2.IMREAD_ANYDEPTH+cv2.IMREAD_ANYCOLOR
    if not os.path.exists(fn):
        raise OSError('No such file or directory: {}'.format(fn))
    elif os.path.isdir(fn):
        raise OSError('Is a directory: {}'.format(fn))
    else:
        try:
            img = cv2.cvtColor(cv2.imread(fn, flags), cv2.COLOR_BGR2RGB)
            img = np.concatenate([np.expand_dims(cv2.equalizeHist(img[:,:,i]), axis=2) for i in range(3)], axis=2)
            return img.astype(np.float32)/255
        except Exception as e:
            raise OSError('Error handling image at: {}'.format(fn)) from e

# Look at examples of image augmentation
def get_augs():
    x,_ = next(iter(data.aug_dl))
    return data.trn_ds.denorm(x)[1]

def plot_loss_change(sched, sma=1, n_skip=20, y_lim=(-0.01,0.01)):
    """
    Plots rate of change of the loss function.
    Parameters:
        sched - learning rate scheduler, an instance of LR_Finder class.
        sma - number of batches for simple moving average to smooth out the curve.
        n_skip - number of batches to skip on the left.
        y_lim - limits for the y axis.
    """
    derivatives = [0] * (sma + 1)
    for i in range(1 + sma, len(learn.sched.lrs)):
        derivative = (learn.sched.losses[i] - learn.sched.losses[i - sma]) / sma
        derivatives.append(derivative)

    plt.ylabel("d/loss")
    plt.xlabel("learning rate (log scale)")
    plt.plot(learn.sched.lrs[n_skip:], derivatives[n_skip:])
    plt.xscale('log')
    plt.ylim(y_lim)


Annotation = namedtuple('Annotation', ['filename', 'label'])

path = '/home/ubuntu/datasets/GTSRB/'
source_path = os.path.join(path, 'GTSRB/Final_Training/Images')
train_path = os.path.join(path, 'train')
validation_path = os.path.join(path, 'valid')
all_path = os.path.join(path, 'all')
validation_fraction = 0.2
split_train_validation_sets(source_path, train_path, validation_path, all_path, validation_fraction)

test_annotations = read_annotations('/home/ubuntu/datasets/GTSRB/GT-final_test.csv')

classes = pd.read_csv('/home/ubuntu/datasets/GTSRB/signnames.csv')
class_names = {}
for i, row in classes.iterrows():
    class_names[str(row[0])] = row[1]

arch = resnet34
bs = 256
wd = 5e-4
lr = 0.01

# imgsz_sched = [12, 24, 48, 72, 96]
imgsz_sched = [96, 96, 96, 96, 96]

def pgfit(arch, data_path, bs, wd, imgsz_sched=[] ):

    pretrained = False

    def get_data(sz):
        aug_tfms = [RandomRotate(20), RandomLighting(0.8, 0.8)]
        tfms = tfms_from_model(arch, sz, aug_tfms=aug_tfms, max_zoom=1.2)
        data = ImageClassifierData.from_paths(data_path, tfms=tfms, trn_name='all', val_name='valid', test_name='test', bs=bs)
        return data

    for idx, sz in enumerate(imgsz_sched):
        data = get_data(sz)

        if idx == 0 and pretrained:
            learn = ConvLearner.pretrained(arch, data, precompute=False)

        if idx == 0 and (not pretrained):
            #from fastai.models.cifar10.resnext import resnext29_8_64
            from fastai.models.resnet import vgg_resnet34
            m = vgg_resnet34()
            bm = BasicModel(m.cuda(), name='vgg_resnet34')
            learn = ConvLearner(data, bm)

        learn.set_data(get_data(sz))
        learn.freeze()
        learn.fit(lr,1, cycle_len=1, cycle_mult=2)

        learn.unfreeze()
        learn.fit(lr, 3, cycle_len=1, cycle_mult = 2)

        learn.save('learn_sz_'+str(sz)+'.pkl')

    learn.sched.plot_loss()
    return learn


import datetime
fh =  open('log.txt', 'w')
print(str(datetime.datetime.now()))
fh.write(str(datetime.datetime.now()))

learn = pgfit(arch, path, bs, wd, imgsz_sched)

print(str(datetime.datetime.now()))
fh.write(str(datetime.datetime.now()))
fh.close()

''' For testing
'''
print("Below is for testing only")
sz = 96
aug_tfms = [RandomRotate(20), RandomLighting(0.8, 0.8)]
tfms = tfms_from_model(arch, sz, aug_tfms=aug_tfms, max_zoom=1.2)
data = ImageClassifierData.from_paths(path, tfms=tfms, trn_name='all', val_name='valid', test_name='test', bs=bs)

print("I am here 1")
true_test_labels = {a.filename: a.label for a in test_annotations}
class_indexes = {c: i for i, c in enumerate(data.classes)}
filenames = [filepath[filepath.find('/') + 1:] for filepath in data.test_ds.fnames]
labels = [str(true_test_labels[filename]) for filename in filenames]
y_true = np.array([class_indexes[label] for label in labels])

print("I am here 2")

log_preds = learn.predict(is_test=True)
preds = np.exp(log_preds)
accuracy_np(preds, y_true)

print("I am here 3")
log_preds,_ = learn.TTA(n_aug=20, is_test=True)
print("I am here 3.1")
preds = np.mean(np.exp(log_preds),0)
print("I am here 3.2")
accuracy_np(preds, y_true)
print("I am here 3.3")
pred_labels = np.argmax(preds, axis=1)
incorrect = [i for i in range(len(pred_labels)) if pred_labels[i] != y_true[i]]
print("I am here 4")

for i in range(0,10):
    print(class_names[data.classes[y_true[incorrect[i]]]], class_names[data.classes[pred_labels[incorrect[i]]]],
          preds[incorrect[i], y_true[incorrect[i]]], preds[incorrect[i], pred_labels[incorrect[i]]])
    plt.imshow(load_img_id(data.test_ds, incorrect[i], path))
    plt.savefig('showoff1'+str(i)+'.png')

print("I am here 5")
cm = confusion_matrix(y_true, pred_labels)
np.savetxt(os.path.join(path, 'confusion_matrix.tsv'), cm, delimiter='\t')
c = Counter([class_names[data.classes[y_true[i]]] for i in incorrect])
c.most_common(20)
c = Counter([class_names[data.classes[pred_labels[i]]] for i in incorrect])
c.most_common(20)

pred_labels = np.argmax(preds, axis=1)

print("I am here 6")
for i in range(10):
    class_id = data.classes[pred_labels[i]]
    filename = data.test_ds.fnames[i].split('/')[1]
    print(filename, class_id, class_names[class_id])
    plt.imshow(load_img_id(data.test_ds, i, path))
    plt.savefig('showoff'+str(i)+'.png')

