import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend
from tensorflow.keras import optimizers, regularizers
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.datasets import cifar10, mnist
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPooling2D, BatchNormalization, Activation, Dropout, Input, Layer, ReLU
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, LearningRateScheduler, Callback
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.constraints import Constraint
from scipy.spatial import distance
from utilities import weight_prosessing, quantized, weight_perturbation


# tf.config.experimental_run_functions_eagerly(True)
# tf.compat.v1.disable_eager_execution()
import os
import math
from tensorflow.keras.models import load_model
import numpy as np
import copy
from scipy.linalg import hadamard
import sys
from AAWO import ONI_Conv2d

#os.environ['CUDA_VISIBLE_DEVICES'] = "1" #2
# os.environ['HOMEPATH']
#os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
batch_size = 128
iterations = 391
num_classes = 10
num_classifiers = 63

# def rmse(y_true, y_pred):
# 	return backend.sqrt(backend.mean(backend.square(backend.round(y_pred) - y_true), axis=-1))
batch_size = 128
hidden_units1 = 125
hidden_units2 = 256
dropout = 0.2  # 0.45
weightclip = 0.125

lam1 = 0.2


def new_accuracy(y_true, y_pred):
    # correct_pred = backend.mean(backend.equal(backend.round(y_pred), y_true))
    # # tf.keras.backend.print_tensor(correct_pred)
    # accuracy = countZeroes(correct_pred) backend.round
    res = backend.mean(backend.cast(backend.equal(backend.round(y_pred), y_true), dtype='float32'), axis=1)
    accuracy = backend.mean(backend.cast(backend.equal(res, 1), dtype='float32'))
    return accuracy


def dist_calculation(y_pred):
    H = cm

    dist = distance.cdist(y_pred, np.array(H), 'euclidean')
    index = np.argmin(dist, axis=1)
    true = []
    for i in index:
        true.append(H[i])

    return true


def validate_accuracy(y_true, y_pred):
    y_predm = tf.convert_to_tensor(np.array(dist_calculation(y_pred)))
    res = tf.reduce_mean(tf.cast(tf.equal(y_predm, y_true), dtype='float32'), axis=1)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(res, 1.), dtype='float32'))

    return accuracy


def Alexnet_ecnn_cifar10(input_img, n):
#    x = Conv2D(64, (11, 11), strides=(4, 4), padding='same', activation='relu', kernel_initializer='uniform')(input_img)
    x = ONI_Conv2d(64, (11, 11), strides=(4, 4), padding='SAME', activation='relu', T=20)(input_img)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='VALID')(x)
#    x = Conv2D(192, (5, 5), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform')(x)
    x = ONI_Conv2d(192, (5, 5), strides=(1, 1), padding='SAME', activation='relu', T=10)(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='VALID')(x)
    model_out = []
    for i in range(n):
        out = Conv2D(20, (3, 3), strides=(1, 1), padding='SAME', activation='relu')(x)
        out = Conv2D(10, (3, 3), strides=(1, 1), padding='SAME', activation='relu')(out)
        out = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(out)
        out = Flatten()(out)
        out = Dense(1)(out)
        model_out.append(out)
    model_output = tf.keras.layers.concatenate(model_out)
    model = Model(input_img, model_output)
    return model

def lr_schedule(epoch):
    """Learning Rate Schedule

    Learning rate is scheduled to be reduced after 80, 120, 160, 180 epochs.
    Called automatically every epoch as part of callbacks during training.

    # Arguments
        epoch (int): The number of epochs

    # Returns
        lr (float32): learning rate
    """
    lr = 1e-3
    if epoch > 180:
        lr *= 0.5e-3
    elif epoch > 160:
        lr *= 1e-3
    elif epoch > 120:
        lr *= 1e-2
    elif epoch > 80:
        lr *= 1e-1
    print('Learning rate: ', lr)
    return lr

def corr_bce(lam1):
    
    def loss(labels, logits):
        
        y_true=labels        
        y_pred=logits
        
       
        bce_loss=tf.keras.losses.BinaryCrossentropy(from_logits=True)(y_true, y_pred)
        
        
        y_pred_bits=tf.math.sigmoid(y_pred)
        
        y_pred_bits=tf.clip_by_value(y_pred_bits,0.001,0.999)
        
        dec_error=y_pred_bits-tf.cast(y_true,tf.float32)
        dec_error_mu=tf.math.reduce_mean(dec_error,0)        
        dec_error=dec_error-dec_error_mu+0.001
#        
        dec_error=dec_error+tf.math.sign(dec_error)*0.0001

        
        cov=tf.transpose(dec_error)@dec_error/batch_size+0.0001*np.eye(num_classifiers)

        t_corr=tf.linalg.trace(tf.math.log(cov))-tf.linalg.logdet(cov)
        

        return bce_loss+lam1*t_corr/num_classifiers
        

    return loss


if __name__ == '__main__':
    # load data
    # (x_train, y_train), (x_test, y_test) = mnist.load_data()
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    y_test_predict = y_test

    image_size = x_train.shape[1]
    input_shape = x_train.shape[1:]
    # print(image_size)
    input_size = image_size * image_size
    # x_train = np.reshape(x_train, [-1, input_size, 1])
    # x_train = np.reshape(x_train, [-1, image_size, image_size, 1])
    x_train = np.reshape(x_train, [-1, image_size, image_size, 3])
    # x_train = x_train.astype('float32') / 255
    # x_test = np.reshape(x_test, [-1, input_size, 1])
    # x_test = np.reshape(x_test, [-1, image_size, image_size, 1])
    x_test = np.reshape(x_test, [-1, image_size, image_size, 3])
    # x_test = x_test.astype('float32') / 255

    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.

    cm = hadamard(num_classifiers+1)[1:11][:,1:]
    cm[np.where(cm == -1)] = 0
    
#    cm = np.identity(num_classes)


    # Augment labels
    y_train_code = np.zeros((y_train.shape[0], num_classifiers))
    y_test_code = np.zeros((y_test.shape[0], num_classifiers))
    for i in range(10):
        idx_train = list(np.where(y_train == i)[0])
        idx_test = list(np.where(y_test == i)[0])
        y_train_code[idx_train, :] = cm[i]
        y_test_code[idx_test, :] = cm[i]


    idx = 1
    if idx == 0:
        dir_pwd = os.getcwd()
        dir_name = 'cifar10_onehot{}_alexnet_sep_orth_lam1={}_4'.format(num_classifiers, lam1)
#            dir_name = 'cifar10_hadamard_alexnet_sep_orth_lam1=0_lam2=0.8'
        save_dir = os.path.join(dir_pwd, dir_name)
#        model_name = 'model.085.hdf5'
        model_name = 'model.050.hdf5'
        model_name = 'model.hdf5'
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        filepath = os.path.join(save_dir, model_name)
        
        model_input = Input(shape=input_shape)
        model = Alexnet_ecnn_cifar10(model_input, num_classifiers)
        model.load_weights(filepath)

        sgd = optimizers.SGD(lr=0.1, momentum=0.9, nesterov=True)
        model.compile(loss=tf.keras.losses.BinaryCrossentropy(), optimizer='adam',
                      metrics=[new_accuracy])
        # model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
        model.summary()
        weight_org = model.get_weights()
        s = []
        for sigma in [0, 0.1, 0.3, 0.5]:
            acc_total = []
            for i in range(100):
                w_org_copy = copy.deepcopy(weight_org)
                weight_perturbation(w_org_copy, sigma, 'lognormal')
                y_pred = model.predict(x_test, batch_size=100)
                accuracy = validate_accuracy(y_test_code, y_pred)
                # score, acc = model.evaluate(x_test, y_test, batch_size=batch_size)
                acc_total.append(accuracy)
#                    print(sess.run(accuracy))
            s.append(np.average(acc_total))
            print(sigma, np.average(acc_total))
        print(s)
        
        for rate in [0, 0.01, 0.05, 0.1]:
            acc_total = []
            for i in range(100):
                w_org_copy = copy.deepcopy(weight_org)
                model.set_weights(quantized(w_org_copy, 8, rate, 99))
                y_pred = model.predict(x_test, batch_size=100)
                accuracy = validate_accuracy(y_test_code, y_pred)
                # score, acc = model.evaluate(x_test, y_test, batch_size=batch_size)
                acc_total.append(accuracy)
#                print(accuracy)
#                    print(sess.run(accuracy))
            s.append(np.average(acc_total))
            print(rate, np.average(acc_total))
        print(s)
#            

    else:
        dir_pwd = os.getcwd()
        dir_name = 'cifar10_onehot{}_alexnet_sep_orth_lam1={}_4'.format(num_classifiers, lam1)
        save_dir = os.path.join(dir_pwd, dir_name)
        model_name = 'model.hdf5'
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        filepath = os.path.join(save_dir, model_name)
        

        model_input = Input(shape=input_shape)
        model = Alexnet_ecnn_cifar10(model_input, num_classifiers)
        sgd = optimizers.SGD(lr=0.01, momentum=0.9, nesterov=True)
        # adam = optimizers.adam(lr=0.01, momentum=0.9, nesterov=True)
        model.compile(loss=corr_bce(lam1), optimizer=Adam(lr=lr_schedule(0)),
                      metrics=[new_accuracy])  # binary_accuracy
        # model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
        model.summary()
        
        print(filepath)

        checkpoint = ModelCheckpoint(
            filepath=filepath, monitor='val_loss', mode='min', verbose=1, save_best_only=True, save_weights_only=True)
        lr_scheduler = LearningRateScheduler(lr_schedule)
        # orthcheck = OrthCallback()
        lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1),
                                       cooldown=0,
                                       patience=5,
                                       min_lr=0.5e-6)
        callbacks = [checkpoint, lr_scheduler]

        datagen = ImageDataGenerator(
            # set input mean to 0 over the dataset
            featurewise_center=False,
            # set each sample mean to 0
            samplewise_center=False,
            # divide inputs by std of dataset
            featurewise_std_normalization=False,
            # divide each input by its std
            samplewise_std_normalization=False,
            # apply ZCA whitening
            zca_whitening=False,
            # epsilon for ZCA whitening
            zca_epsilon=1e-06,
            # randomly rotate images in the range (deg 0 to 180)
            rotation_range=0,
            # randomly shift images horizontally
            width_shift_range=0.1,
            # randomly shift images vertically
            height_shift_range=0.1,
            # set range for random shear
            shear_range=0.,
            # set range for random zoom
            zoom_range=0.,
            # set range for random channel shifts
            channel_shift_range=0.,
            # set mode for filling points outside the input boundaries
            fill_mode='nearest',
            # value used for fill_mode = "constant"
            cval=0.,
            # randomly flip images
            horizontal_flip=True,
            # randomly flip images
            vertical_flip=False,
            # set rescaling factor (applied before any other transformation)
            rescale=None,
            # set function that will be applied on each input
            preprocessing_function=None,
            # image data format, either "channels_first" or "channels_last"
            data_format=None,
            # fraction of images reserved for validation (strictly between 0 and 1)
            validation_split=0.0)

        # Compute quantities required for featurewise normalization
        # (std, mean, and principal components if ZCA whitening is applied).
        datagen.fit(x_train)

        # Fit the model on the batches generated by datagen.flow().
        model.fit_generator(
            datagen.flow(x_train, y_train_code, batch_size=batch_size),
            validation_data=(x_test, y_test_code),
            epochs=200,
            verbose=1,
            callbacks=callbacks)



