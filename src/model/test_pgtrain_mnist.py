import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

from fastai.imports import *
from fastai.transforms import *
from fastai.conv_learner import *
from fastai.model import *
from fastai.dataset import *
from fastai.sgdr import *
from fastai.plots import *

def reshape_img(matrix):
    """
    Reshape an existing 2D pandas.dataframe into 3D-numpy.ndarray
    """
    try:
        return matrix.values.reshape(-1, 28, 28)
    except AttributeError as e:
        print(e)


def add_color_channel(matrix):
    """
    Add missing color channels to previously reshaped image
    """
    matrix = np.stack((matrix, ) *3, axis = -1)
    return matrix


def convert_ndarry(matrix):
    """
    Convert pandas.series into numpy.ndarray
    """
    try:
        return matrix.values.flatten()
    except AttributeError as e:
        print(e)

def get_data(arch, sz, bs):
    tfms = tfms_from_model(arch, sz, aug_tfms=transforms_side_on, max_zoom=1.1)
    data = ImageClassifierData.from_csv(PATH, 'train', f'{PATH}labels.csv', test_name='test', val_idxs=val_idxs, suffix='.jpg', tfms=tfms, bs=bs)


def pgtrain_classification_fromScratch(learn, data, convg_tol, imgsz_sched=[]):

    # define image size variation during training
    if len(imgsz_sched) == 0:
        imgsz_sched = [0.25, 0.5, 0.75, 1] * data.shape[0]
    else:
        imgsz_sched = [ i*data.shape[0] for i in imgsz_sched]

    # define differential learning rates
    lr = np.array([0.001, 0.0075, 0.01])

    for sz in imgsz_sched:
        # simply change data size for each training epoch
        learn.set_data(get_data(sz, bs))

        if index > 0:
            learn.fit(1e-2, 1,wd=wd)

        # by default, [:-2] layers are all freezed initially
        learn.unfreeze()

        # find optimal learning rate
        learn.lr_find()
        learn.fit(lr, max_epochs, cycle_len=3, cycle_mult = 2)

    # plot loss vs. learning rate
    learn.sched.plot()

def predict_test_classification(learn):
    log_preds, y_test = learn.TTA(is_test=True)
    probs = np.mean(np.exp(log_preds), 0)
    accuracy_np(probs, y)


print(torch.cuda.is_available(), torch.backends.cudnn.enabled)

wd = "../../../MNIST/"

# load data
train_df = pd.read_csv(f"{wd}train.csv")
test_df = pd.read_csv(f"{wd}test.csv")

print(train_df.shape, test_df.shape)

# create validation dataset
val_df = train_df.sample(frac=0.2, random_state=1337)
val_df.shape

# remove validation data from train dataset
train_df = train_df.drop(val_df.index)
train_df.shape

# separate labels from data
Y_train = train_df["label"]
Y_valid = val_df["label"]
X_train = train_df.drop("label", axis=1)
X_valid = val_df.drop("label", axis=1)

print(X_train.shape, X_valid.shape)
print(Y_train.shape, Y_valid.shape)

# display an actual image/digit
img = X_train.iloc[0,:].values.reshape(28,28)
plt.imshow(img, cmap="gray")

# reshape data and add color channels
X_train = reshape_img(X_train)
X_train = add_color_channel(X_train)
X_valid = reshape_img(X_valid)
X_valid = add_color_channel(X_valid)
test_df = reshape_img(test_df)
test_df = add_color_channel(test_df)


# convert y_train and y_valid into proper numpy.ndarray
Y_train = convert_ndarry(Y_train)
Y_valid = convert_ndarry(Y_valid)

preprocessed_data = [X_train, Y_train, X_valid, Y_valid, test_df]
print([e.shape for e in preprocessed_data])
print([type(e) for e in preprocessed_data])

data = ImageClassifierData.from_arrays(path=wd,
                                       trn=(X_train, Y_train),
                                       val=(X_valid, Y_valid),
                                       classes=Y_train,
                                       test=test_df,
                                       tfms=tfms_from_model(arch, sz))


learn = ConvLearner.pretrained(arch, data, precompute=True)

pgtrain_classification_from_scratch(learn, data, imgsz_sched=[0.25, 0.5, 0.75, 1])

predict_test_classification(learn)


