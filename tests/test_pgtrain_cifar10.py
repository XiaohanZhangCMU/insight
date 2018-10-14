from fastai.conv_learner import *

PATH = "/home/ubuntu/datasets/cifar10/"
os.makedirs(PATH,exist_ok=True)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
stats = (np.array([ 0.4914 ,  0.48216,  0.44653]), np.array([ 0.24703,  0.24349,  0.26159]))

def get_data(sz,bs):
    tfms = tfms_from_stats(stats, sz, aug_tfms=[RandomFlip()], pad=sz//8)
    return ImageClassifierData.from_csv(PATH, 'train', f'{PATH}train.csv', suffix='.png',  tfms=tfms, bs=bs)

bs=128

data = get_data(32,4)

from fastai.models.cifar10.resnext import resnext29_8_64

m = resnext29_8_64()
bm = BasicModel(m.cuda(), name='cifar10_rn29_8_64')

data = get_data(8,bs*4)

learn = ConvLearner(data, bm)
learn.unfreeze()

lr=1e-2; wd=5e-4

learn.lr_find()

learn.fit(lr, 1)

learn.fit(lr, 2, cycle_len=1)


