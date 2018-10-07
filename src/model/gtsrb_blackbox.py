"""
This tutorial shows how to generate adversarial examples
using FGSM in black-box setting.
The original paper can be found at:
https://arxiv.org/abs/1602.02697
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import time
import functools

import numpy as np
from six.moves import xrange

import logging
import pickle
import tensorflow as tf
from tensorflow.python.platform import flags

from cleverhans.loss import CrossEntropy
from cleverhans.model import Model
from cleverhans.utils_mnist import data_mnist
from cleverhans.utils import other_classes, set_log_level
from cleverhans.utils import to_categorical
from cleverhans.utils import set_log_level
from cleverhans.utils import pair_visual, grid_visual, AccuracyReport
from cleverhans.utils_tf import model_eval, train # batch_eval
from cleverhans.utils_tf import model_argmax
#from cleverhans.train import train
from cleverhans.attacks import FastGradientMethod
from cleverhans.attacks_tf import jacobian_graph, jacobian_augmentation
from cleverhans.evaluation import batch_eval

from cleverhans_tutorials.tutorial_models import ModelBasicCNN, \
    HeReLuNormalInitializer
from cleverhans.utils import TemporaryLogLevel
from cleverhans.utils_keras import KerasModelWrapper

import keras
#from keras.models import Model as KerasModel
from keras.models import load_model
from input_dataset import read_gtsrb_dataset

FLAGS = flags.FLAGS

NB_CLASSES = 43
BATCH_SIZE = 128
LEARNING_RATE = .1
NB_EPOCHS = 30
HOLDOUT = 1<<10 # This number needs to be smaller than len(x_test)
DATA_AUG = 1 # <<5
NB_EPOCHS_S = 1 # 15 #
LMBDA = .1
AUG_BATCH_SIZE = 512
IMG_SIZE = 48
MODEL_PATH = './pg_train_results/model.pg.h5.48'


def setup_tutorial():
    """
    Helper function to check correct configuration of tf for tutorial
    :return: True if setup checks completed
    """

    # Set TF random seed to improve reproducibility
    tf.set_random_seed(1234)

    return True

def add_gaussian_noise(img,mean=0.0, std=1e-3):
    noisy_img = img + np.random.normal(mean, std, img.shape)
    noisy_img_clipped = np.clip(noisy_img, 0, 1)  # might get out of bounds due to noise
    return noisy_img_clipped


def prep_bbox(sess, x, y, x_train, y_train, x_test, y_test,
              nb_epochs, batch_size, learning_rate,
              rng, nb_classes=10, img_rows=28, img_cols=28, nchannels=1):
    """
    Define and train a model that simulates the "remote"
    black-box oracle described in the original paper.
    :param sess: the TF session
    :param x: the input placeholder for MNIST
    :param y: the ouput placeholder for MNIST
    :param x_train: the training data for the oracle
    :param y_train: the training labels for the oracle
    :param x_test: the testing data for the oracle
    :param y_test: the testing labels for the oracle
    :param nb_epochs: number of epochs to train model
    :param batch_size: size of training batches
    :param learning_rate: learning rate for training
    :param rng: numpy.random.RandomState
    :return:
    """
    keras.layers.core.K.set_learning_phase(1)
    config = tf.ConfigProto(device_count = {'GPU' : 1})
    # sess = tf.InteractiveSession(config=config)
    keras.backend.set_session(sess)

    model_path = MODEL_PATH
    try:
        oracle = KerasModelWrapper(load_model(model_path))
    except:
        import errno, os
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), model_path)

    loss = CrossEntropy(oracle, smoothing=0.1)
    predictions = oracle.get_logits(x)
    print("Loaded well-trained Keras oracle.")

    # Print out the accuracy on legitimate data
    eval_params = {'batch_size': batch_size}
    accuracy = model_eval(sess, x, y, predictions, x_test, y_test,
                          args=eval_params)
    print('Test accuracy of black-box on legitimate test '
          'examples: ' + str(accuracy))

    return oracle, predictions, accuracy


class ModelSubstitute(Model):
    def __init__(self, scope, nb_classes, session= tf.Session(), istrain=False, nb_filters=200, **kwargs):
        del kwargs

        self.session = session
        with session.as_default():
            Model.__init__(self, scope, nb_classes, locals())
            self.nb_filters = nb_filters
            #self.session.run(tf.global_variables_initializer())
            #self.session.run(tf.initialize_all_variables())

            self.c1 = tf.Variable(tf.truncated_normal(shape=[3,3,3,32], stddev=0.1))
            self.b1 = tf.Variable(tf.constant(1.0, shape=[32, 48,48]))
            self.c2 = tf.Variable(tf.truncated_normal(shape=[3,3,32,64], stddev=0.1))
            self.b2 = tf.Variable(tf.constant(1.0, shape=[64,24,24]))
            self.c3 = tf.Variable(tf.truncated_normal(shape=[2,2,64,128], stddev=0.1))
            self.b3 = tf.Variable(tf.constant(1.0, shape=[128,12,12]))

            self.w1 = tf.Variable(tf.truncated_normal(shape=[6*6*128, 2048], stddev=0.1))
            self.b4 = tf.Variable(tf.constant(0.0, shape=[2048]))
            self.w2 = tf.Variable(tf.truncated_normal(shape=[2048, 1024], stddev=0.1))
            self.b5 = tf.Variable(tf.constant(0.0, shape=[1024]))

            self.w3 = tf.Variable(tf.truncated_normal(shape=[1024,43], stddev=0.1))
            self.b6 = tf.Variable(tf.constant(0.0, shape=[43]))

            self.istrain = istrain

    def fprop(self, x, **kwargs):
        del kwargs
        #with self.session.as_default():
        with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE):
            conv = tf.nn.conv2d(input=x, filter=self.c1, strides=[1,1,1,1], padding='SAME', data_format='NCHW')
            conv = tf.nn.relu(conv + self.b1)
            maxp = tf.nn.max_pool(conv, ksize=[1,1,2,2], strides=[1,1,2,2], padding='SAME', data_format='NCHW')

            # Conv+Max Pool (26,26,32) -> (13, 13, 64)
            conv2 = tf.nn.conv2d(input=maxp, filter=self.c2, strides=[1,1,1,1], padding='SAME', data_format='NCHW')
            conv2 = tf.nn.relu(conv2 + self.b2)
            maxp2 = tf.nn.max_pool(conv2, ksize=[1,1,2,2], strides=[1,1,2,2], padding='SAME', data_format='NCHW')

            # Conv+Max Pool (13, 13, 64) -> (7,7,128)
            conv3 = tf.nn.conv2d(input=maxp2, filter=self.c3, strides=[1,1,1,1], padding='SAME', data_format='NCHW')
            conv3 = tf.nn.relu(conv3 + self.b3)
            maxp3 = tf.nn.max_pool(conv3, ksize=[1,1,2,2], strides=[1,1,2,2], padding='SAME', data_format='NCHW')
            maxp3 = tf.reshape(maxp3, shape=[-1, 6*6*128])

            # Dense Layers
            dl = tf.nn.relu(tf.matmul(maxp3, self.w1)+self.b4)
            if self.istrain:
                dl = tf.nn.dropout(dl, 0.5)
            dl2 = tf.nn.relu(tf.matmul(dl, self.w2)+self.b5)
            if self.istrain:
                dl2 = tf.nn.dropout(dl2,0.5)
            logits = tf.matmul(dl2, self.w3) + self.b6

            return {self.O_LOGITS: logits,
                    self.O_PROBS: tf.nn.softmax(logits=logits)}

def train_sub1(sess, x, y, bbox_preds, x_sub, y_sub, nb_classes,
              nb_epochs_s, batch_size, learning_rate, data_aug, lmbda,
              aug_batch_size, rng, img_rows=48, img_cols=48,
              nchannels=3):
    """
    This function creates the substitute by alternatively
    augmenting the training data and training the substitute.
    :param sess: TF session
    :param x: input TF placeholder
    :param y: output TF placeholder
    :param bbox_preds: output of black-box model predictions
    :param x_sub: initial substitute training data
    :param y_sub: initial substitute training labels
    :param nb_classes: number of output classes
    :param nb_epochs_s: number of epochs to train substitute model
    :param batch_size: size of training batches
    :param learning_rate: learning rate for training
    :param data_aug: number of times substitute training data is augmented
    :param lmbda: lambda from arxiv.org/abs/1602.02697
    :param rng: numpy.random.RandomState instance
    :return:
    """
    # Define TF model graph (for the black-box model)
    model_sub = ModelSubstitute('model_s', nb_classes)
    preds_sub = model_sub.get_logits(x)
    loss_sub = CrossEntropy(model_sub, smoothing=0)

    print("Defined TensorFlow model graph for the substitute.")

    # Define the Jacobian symbolically using TensorFlow
    grads = jacobian_graph(preds_sub, x, nb_classes)

    # Train the substitute and augment dataset alternatively
    for rho in xrange(data_aug):
        print("Substitute training epoch #" + str(rho))
        train_params = {
            'nb_epochs': nb_epochs_s,
            'batch_size': batch_size,
            'learning_rate': learning_rate
        }
        #with TemporaryLogLevel(logging.WARNING, "cleverhans.utils.tf"):
        train(sess, loss_sub, x, y, x_sub,
              to_categorical(y_sub, nb_classes),
              init_all=False, args=train_params, rng=rng)
              #var_list=model_sub.get_params())


        # If we are not at last substitute training iteration, augment dataset
        if rho < data_aug - 1:
            print("Augmenting substitute training data.")
            # Perform the Jacobian augmentation
            lmbda_coef = 2 * int(int(rho / 3) != 0) - 1
            # print(x.shape)
            # print(x_sub.shape)
            # print(y_sub.shape)
            #print(grads.shape)
            x_sub = jacobian_augmentation(sess, x, x_sub, y_sub, grads,
                                          lmbda_coef * lmbda, aug_batch_size)

            print("Labeling substitute training data.")
            # Label the newly generated synthetic points using the black-box
            y_sub = np.hstack([y_sub, y_sub])
            x_sub_prev = x_sub[int(len(x_sub)/2):]
            eval_params = {'batch_size': batch_size}
            #tmp = batch_eval(sess, [x], [bbox_preds], [x_sub_prev],args=eval_params)
            tmp = batch_eval(sess, [x], [bbox_preds], [x_sub_prev],batch_size=batch_size)
            print(tmp)
            bbox_val = tmp[0]

            # Note here that we take the argmax because the adversary
            # only has access to the label (not the probabilities) output
            # by the black-box model
            y_sub[int(len(x_sub)/2):] = np.argmax(bbox_val, axis=1)

    return model_sub, preds_sub


def train_sub(sess, x, y, bbox_preds, x_sub, y_sub, nb_classes,
              nb_epochs_s, batch_size, learning_rate, data_aug, lmbda,
              aug_batch_size, rng, img_rows=48, img_cols=48,
              nchannels=3):
    """
    This function creates the substitute by alternatively
    augmenting the training data and training the substitute.
    :param sess: TF session
    :param x: input TF placeholder
    :param y: output TF placeholder
    :param bbox_preds: output of black-box model predictions
    :param x_sub: initial substitute training data
    :param y_sub: initial substitute training labels
    :param nb_classes: number of output classes
    :param nb_epochs_s: number of epochs to train substitute model
    :param batch_size: size of training batches
    :param learning_rate: learning rate for training
    :param data_aug: number of times substitute training data is augmented
    :param lmbda: lambda from arxiv.org/abs/1602.02697
    :param rng: numpy.random.RandomState instance
    :return:
    """
    assert(y_sub.shape[1]>1)

    try:
        saver.restore(sess, "./model.ckpt")
        model_sub = tf.get_variable("logits", shape=[1])
        preds_sub = tf.get_variable("probs", shape=[1])
        return model_sub, preds_sub
    except:
        print("Model ckpt is not found. Retrain substitute starts.")

    # Define TF model graph (for the black-box model)
    model_sub = ModelSubstitute('model_s',nb_classes, session=sess, istrain=True)
    logits = model_sub.get_logits(x)
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=y))
    optimiser = tf.train.AdamOptimizer().minimize(loss)
    preds_sub = tf.nn.softmax(logits=logits)

    saver = tf.train.Saver()

    print("Defined TensorFlow model graph for the substitute.")

    # Define the Jacobian symbolically using TensorFlow
    grads = jacobian_graph(preds_sub, x, nb_classes)
    sess.run(tf.global_variables_initializer())

    def evaluate():
        acc = model_eval(sess, x, y, preds_sub, x_sub, y_sub, args=eval_params)
        print('Test accuracy on test examples: %0.4f' % (acc))

    # Train the substitute and augment dataset alternatively
    for rho in xrange(data_aug):
        print("Substitute training epoch #" + str(rho))

        for s in range(batch_size):
            batch_xs = x_sub[s*batch_size: (s+1)*batch_size]
            batch_ys = y_sub[s*batch_size: (s+1)*batch_size]
            feed_dict = {x:batch_xs, y:batch_ys}
            op, lval,pre = sess.run([optimiser, loss, preds_sub], feed_dict=feed_dict)
        print("rho = {0}. loss : {1}".format(rho, sess.run(loss, feed_dict={x:batch_xs, y:batch_ys})))

        # If we are not at last substitute training iteration, augment dataset
        if 0: # rho < data_aug - 1:
            print("Augmenting substitute training data.")
            # Perform the Jacobian augmentation
            lmbda_coef = 2 * int(int(rho / 3) != 0) - 1
            y_sub_labels = np.argmax(y_sub, axis=1).reshape(-1,1)
            x_sub = jacobian_augmentation(sess, x, x_sub, y_sub_labels, grads,
                                          lmbda_coef * lmbda, aug_batch_size)

            # Label the newly generated synthetic points using the black-box
            new_y_sub_labels = np.vstack((y_sub_labels, y_sub_labels))
            x_sub_prev = x_sub[int(len(x_sub)/2):]
            eval_params = {'batch_size': batch_size}
            tmp = batch_eval(sess,[x],[bbox_preds],[x_sub_prev],batch_size=batch_size)
            bbox_val = tmp[0]

            # Note here that we take the argmax because the adversary
            # only has access to the label (not the probabilities) output
            # by the black-box model
            tmp1 = np.argmax(bbox_val, axis=1)
            tmp2 = y_sub_labels[int(len(x_sub)/2):]
            new_y_sub_labels[int(len(x_sub)/2):] = np.argmax(bbox_val, axis=1).reshape(-1,1)
            y_sub = to_categorical(new_y_sub_labels, nb_classes)

    save_path = saver.save(sess, "./model.ckpt")
    print("Model saved in path: %s" % save_path)

    print(preds_sub.shape)
    print(model_sub.shape)

    return model_sub, preds_sub

def savefigfromarray(sample, filename = 'my2.png'):
    from PIL import Image
    data = sample
    print(data.shape)
    d = np.transpose(data, (1,2,0))
    img = Image.fromarray(data*255,'RGB')
    print(img)
    img.save(filename)

def gtsrb_blackbox(train_start=0, train_end=60000, test_start=0,
                   test_end=10000, nb_classes=NB_CLASSES,
                   batch_size=BATCH_SIZE, learning_rate=LEARNING_RATE,
                   nb_epochs=NB_EPOCHS, holdout=HOLDOUT, data_aug=DATA_AUG,
                   nb_epochs_s=NB_EPOCHS_S, lmbda=LMBDA,
                   aug_batch_size=AUG_BATCH_SIZE):
    """
    MNIST tutorial for the black-box attack from arxiv.org/abs/1602.02697
    :param train_start: index of first training set example
    :param train_end: index of last training set example
    :param test_start: index of first test set example
    :param test_end: index of last test set example
    :return: a dictionary with:
             * black-box model accuracy on test set
             * substitute model accuracy on test set
             * black-box model accuracy on adversarial examples transferred
               from the substitute model
    """

    # Set logging level to see debug information
    set_log_level(logging.DEBUG)

    # Dictionary used to keep track and return key accuracies
    accuracies = {}

    # Perform tutorial setup
    assert setup_tutorial()

    # Create TF session
    sess = tf.Session()

    t1 = time.time()
    x_train, y_train, x_VAL, y_VAL, x_test, y_test = read_gtsrb_dataset()
    print('Data reading time :', time.time()-t1, 'seconds')

    # Initialize substitute training set reserved for adversary
    x_sub = x_test[:holdout]

    savefigfromarray(x_sub[0],filename = 'my2.ppm')
    #y_sub = np.argmax(y_test[:holdout], axis=1)
    y_sub = y_test[:holdout]

    print(x_sub.shape)
    print(y_sub.shape)
    print(x_train.shape)
    print(y_train.shape)
    print(x_test.shape)
    print(y_test.shape)

    # Redefine test set as remaining samples unavailable to adversaries
    x_test = x_test[holdout:]
    y_test = y_test[holdout:]

    # Obtain Image parameters
    nchannels, img_rows, img_cols = x_train.shape[1:4]
    nb_classes = y_train.shape[1]

    # Define input TF placeholder
    x = tf.placeholder(tf.float32, shape=(None, nchannels, img_rows, img_cols))
    y = tf.placeholder(tf.float32, shape=(None, nb_classes))

    # Seed random number generator so tutorial is reproducible
    rng = np.random.RandomState([2017, 8, 30])

    # Simulate the black-box model locally
    print("Loading the black-box model.")
    t1 = time.time()
    prep_bbox_out = prep_bbox(sess, x, y, x_train, y_train, x_test, y_test,
                              nb_epochs, batch_size, learning_rate,
                              rng, nb_classes, img_rows, img_cols, nchannels)
    model, bbox_preds, accuracies['bbox'] = prep_bbox_out
    print(bbox_preds.shape)
    print('Oracle loading time :', time.time()-t1, 'seconds')

    # Evaluate oracle on random noised test samples
    rand_x_test, rand_y_test = [], y_test
    try:
        rand_x_test = np.load('rand_x_test.npy')
    except:
        for itest in range(len(x_test)):
            rand_x_test.append(add_gaussian_noise(x_test[itest], std=0.1))
        rand_x_test = np.array(rand_x_test)
    eval_params = {'batch_size': batch_size}
    acc = model_eval(sess, x, y, bbox_preds, rand_x_test, rand_y_test, args=eval_params)
    accuracies['oracle on noise'] = acc

    # Train substitute using method from https://arxiv.org/abs/1602.02697
    print("Training the substitute model.")
    t1 = time.time()

    train_sub_out = train_sub(sess, x, y, bbox_preds, x_train, y_train,
                              nb_classes, nb_epochs_s, batch_size,
                              learning_rate, data_aug, lmbda, aug_batch_size,
                              rng, img_rows, img_cols, nchannels)

    print('Substitute training time :', time.time()-t1, 'seconds')

    model_sub, preds_sub = train_sub_out
    print(preds_sub.shape)
    # Evaluate the substitute model on clean test examples
    eval_params = {'batch_size': batch_size}
    acc = model_eval(sess, x, y, preds_sub, x_train, y_train, args=eval_params)
    accuracies['sub'] = acc
    print('sub on clean test {0}'.format(acc))

    # Initialize the Fast Gradient Sign Method (FGSM) attack object.
    fgsm_par = {'eps': 0.3, 'ord': np.inf, 'clip_min': 0., 'clip_max': 1.}
    fgsm = FastGradientMethod(model_sub, sess=sess)

    # Craft adversarial examples using the substitute
    t1 = time.time()
    eval_params = {'batch_size': batch_size}
    x_adv_sub = fgsm.generate(x, **fgsm_par)
    print('Adversarial example crafting time :', time.time()-t1, 'seconds')

    # Evaluate the accuracy of the "black-box" model on adversarial examples
    accuracy = model_eval(sess, x, y, model.get_logits(x_adv_sub),
                          x_test, y_test, args=eval_params)
    print('Test accuracy of oracle on adversarial examples generated '
          'using the substitute: ' + str(accuracy))
    accuracies['bbox_on_sub_adv_ex'] = accuracy

    # Visualize one example:
    x_adv_sub_0  = x_adv_sub.eval(session=sess, feed_dict = {x:x_test[0].reshape(1,3,48,48)})
    print('ONE EXMAPLE: shape = {0}'.format(x_adv_sub_0.shape))
    print('symbolic x_adv_sub: shape = {0}'.format(x_adv_sub.shape))
    np.save('x_adv_sub_0', x_adv_sub_0)

    ###########################################################################
    # Visualize adversarial examples as a grid of pictures.
    ###########################################################################
    source_samples = 10
    img_rows = 48
    img_cols = 48
    nchannels = 3
    print('Crafting ' + str(source_samples) + ' * ' + str(nb_classes - 1) +
          ' adversarial examples')

    # Keep track of success (adversarial example classified in target)
    results = np.zeros((nb_classes, source_samples), dtype='i')

    # Rate of perturbed features for each test set example and target class
    perturbations = np.zeros((nb_classes, source_samples), dtype='f')

    # Initialize our array for grid visualization
    grid_shape = (nb_classes, nb_classes, img_rows, img_cols, nchannels)
    grid_viz_data = np.zeros(grid_shape, dtype='f')

    # Instantiate a SaliencyMapMethod attack object
    # jsma = SaliencyMapMethod(model, back='tf', sess=sess)
    # jsma_params = {'theta': 1., 'gamma': 0.1,
    #                'clip_min': 0., 'clip_max': 1.,
    #                'y_target': None}

    figure = None
    # Loop over the samples we want to perturb into adversarial examples
    for sample_ind in xrange(0, source_samples):
        print('--------------------------------------')
        print('Attacking input %i/%i' % (sample_ind + 1, source_samples))
        sample = x_test[sample_ind:(sample_ind + 1)]

        # We want to find an adversarial example for each possible target class
        # (i.e. all classes that differ from the label given in the dataset)
        current_class = int(np.argmax(y_test[sample_ind]))
        target_classes = other_classes(nb_classes, current_class)

        # For the grid visualization, keep original images along the diagonal
        grid_viz_data[current_class, current_class, :, :, :] = np.reshape(
            sample, (img_rows, img_cols, nchannels))

        # Loop over all target classes
        for target in target_classes[:3]:
            print('Generating adv. example for target class %i' % target)

            # This call runs the Jacobian-based saliency map approach
            one_hot_target = np.zeros((1, nb_classes), dtype=np.float32)
            one_hot_target[0, target] = 1
            # jsma_params['y_target'] = one_hot_target
            #adv_x = jsma.generate_np(sample, **jsma_params)
            adv_x = fgsm.generate_np(sample, **fgsm_par)

            # Check if success was achieved
            res = int(model_argmax(sess, x, preds_sub, adv_x) == target)

            # Computer number of modified features
            adv_x_reshape = adv_x.reshape(-1)
            test_in_reshape = x_test[sample_ind].reshape(-1)
            nb_changed = np.where(adv_x_reshape != test_in_reshape)[0].shape[0]
            percent_perturb = float(nb_changed) / adv_x.reshape(-1).shape[0]

            # Display the original and adversarial images side-by-side
            fig1 = pair_visual(
                np.reshape(sample, (img_rows, img_cols, nchannels)),
                np.reshape(adv_x, (img_rows, img_cols, nchannels)), figure)

            # Add our adversarial example to our grid data
            fig2 = grid_viz_data[target, current_class, :, :, :] = np.reshape(
                adv_x, (img_rows, img_cols, nchannels))

            # Update the arrays for later analysis
            results[target, sample_ind] = res
            perturbations[target, sample_ind] = percent_perturb
    fig1.savefig('fig1.png')
    np.save('fig2.png', fig2)

    print('--------------------------------------')

    return accuracies


def main(argv=None):
    metrics = gtsrb_blackbox(nb_classes=FLAGS.nb_classes, batch_size=FLAGS.batch_size,
                   learning_rate=FLAGS.learning_rate,
                   nb_epochs=FLAGS.nb_epochs, holdout=FLAGS.holdout,
                   data_aug=FLAGS.data_aug, nb_epochs_s=FLAGS.nb_epochs_s,
                   lmbda=FLAGS.lmbda, aug_batch_size=FLAGS.data_aug_batch_size)

    print('Statistics of attacking : {0}'.format(time.time()))

    for key, val in metrics.items():
        print('acc of {0} is : {1}\n'.format(key,val))


if __name__ == '__main__':
    # General flags
    flags.DEFINE_integer('nb_classes', NB_CLASSES,
                         'Number of classes in problem')
    flags.DEFINE_integer('batch_size', BATCH_SIZE,
                         'Size of training batches')
    flags.DEFINE_float('learning_rate', LEARNING_RATE,
                       'Learning rate for training')

    # Flags related to oracle
    flags.DEFINE_integer('nb_epochs', NB_EPOCHS,
                         'Number of epochs to train model')

    # Flags related to substitute
    flags.DEFINE_integer('holdout', HOLDOUT,
                         'Test set holdout for adversary')
    flags.DEFINE_integer('data_aug', DATA_AUG,
                         'Number of substitute data augmentations')
    flags.DEFINE_integer('nb_epochs_s', NB_EPOCHS_S,
                         'Training epochs for substitute')
    flags.DEFINE_float('lmbda', LMBDA, 'Lambda from arxiv.org/abs/1602.02697')
    flags.DEFINE_integer('data_aug_batch_size', AUG_BATCH_SIZE,
                         'Batch size for augmentation')
    flags.DEFINE_string('model_path', MODEL_PATH, 'model to run')

    tf.app.run()
