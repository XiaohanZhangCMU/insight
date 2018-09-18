#!/usr/bin/env python
# coding: utf-8

# This file contains all the main external libs we'll use
from fastai.imports import *
from fastai.transforms import *
from fastai.conv_learner import *
from fastai.model import *
from fastai.dataset import *
from fastai.sgdr import *
from fastai.plots import *
from sklearn.metrics import confusion_matrix

def rand_by_mask(mask): return np.random.choice(np.where(mask)[0], min(len(preds), 4), replace=False)
def rand_by_correct(is_correct): return rand_by_mask((preds == data.val_y)==is_correct)

def plots(ims, figsize=(12,6), rows=1, titles=None):
    f = plt.figure(figsize=figsize)
    for i in range(len(ims)):
        sp = f.add_subplot(rows, len(ims)//rows, i+1)
        sp.axis('Off')
        if titles is not None: sp.set_title(titles[i], fontsize=16)
        plt.imshow(ims[i])

def load_img_id(ds, idx): return np.array(PIL.Image.open(PATH+ds.fnames[idx]))

def plot_val_with_title(idxs, title):
    imgs = [load_img_id(data.val_ds,x) for x in idxs]
    title_probs = [probs[x] for x in idxs]
    print(title)
    return plots(imgs, rows=1, titles=title_probs, figsize=(16,8)) if len(imgs)>0 else print('Not Found.')

def most_by_mask(mask, mult):
    idxs = np.where(mask)[0]
    return idxs[np.argsort(mult * probs[idxs])[:4]]

def most_by_correct(y, is_correct):
    mult = -1 if (y==1)==is_correct else 1
    return most_by_mask(((preds == data.val_y)==is_correct) & (data.val_y == y), mult)


def get_augs():
    data = ImageClassifierData.from_paths(PATH, bs=2, tfms=tfms, num_workers=1)
    x,_ = next(iter(data.aug_dl))
    return data.trn_ds.denorm(x)[1]

def binary_loss(y, p):
    return np.mean(-(y * np.log(p) + (1-y)*np.log(1-p)))

PATH = "/Users/x/Downloads/GraphicsScratch/dogscats/"
sz=224

torch.cuda.is_available()
torch.backends.cudnn.enabled

files = os.listdir(f'{PATH}valid/cats')[:5]
img = plt.imread(f'{PATH}valid/cats/{files[0]}')

''' Model and learning rate schedule
'''
arch=resnet34

''' Data augmentation
'''
tfms = tfms_from_model(resnet34, sz, aug_tfms=transforms_side_on, max_zoom=1.1)
ims = np.stack([get_augs() for i in range(6)])
plots(ims, rows=2)
data = ImageClassifierData.from_paths(PATH, tfms=tfms)

# By default when we create a learner, it sets all but the last layer to *frozen*. That means that it's still only updating the weights in the last layer when we call `fit`.

learn = ConvLearner.pretrained(arch, data, precompute=True)
lrf = learn.lr_find()
learn.sched.plot()

learn.fit(1e-2, 3, cycle_len=1)
learn.sched.plot_lr()
learn.save('/home/ubuntu/224_lastlayer')
#learn.load('224_lastlayer')

exit(0)

''' Now that final layer is trained, fine-tuning the other layers to unfreeze the remaining layers
'''
learn.unfreeze()

lr=np.array([1e-4,1e-3,1e-2])
learn.fit(lr, 3, cycle_len=1, cycle_mult=2)
learn.sched.plot_lr()

learn.save('224_all')
#learn.load('224_all')

''' Make prediction with Test Time Augmentation
'''
log_preds,y = learn.TTA()
probs = np.mean(np.exp(log_preds),0)
preds = np.argmax(probs, axis=1)
accuracy_np(probs, y)

''' Results visualiztion. confusion matrix etc
'''
cm = confusion_matrix(y, preds)
plot_confusion_matrix(cm, data.classes)

''' Randomly check correct/incorrect labels
'''
# 1. A few correct labels at random
plot_val_with_title(rand_by_correct(True), "Correctly classified")

# 2. A few incorrect labels at random
plot_val_with_title(rand_by_correct(False), "Incorrectly classified")

plot_val_with_title(most_by_correct(0, True), "Most correct cats")

plot_val_with_title(most_by_correct(1, True), "Most correct dogs")

plot_val_with_title(most_by_correct(0, False), "Most incorrect cats")

plot_val_with_title(most_by_correct(1, False), "Most incorrect dogs")

most_uncertain = np.argsort(np.abs(probs -0.5))[:4]
plot_val_with_title(most_uncertain, "Most uncertain predictions")




