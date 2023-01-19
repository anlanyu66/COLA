import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend
# import tensorflow.keras as keras
# from tensorflow.keras.optimizers import SGD, Adam, RMSprop
from tensorflow.keras import optimizers, regularizers
# from tensorflow.keras.optimizers import SGD
# from tensorflow.keras import layers
from tensorflow.keras.datasets import cifar10, mnist, cifar100
# from tensorflow.keras.models import Sequential,load_model
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPooling2D, BatchNormalization, Activation, Dropout, Input, Layer
# from tensorflow.keras.callbacks import LearningRateScheduler, TensorBoard
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, LearningRateScheduler
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.constraints import Constraint
from scipy.spatial import distance
# from utils_ecnn_qary import *
from utilities import quantized
import copy


# tf.config.experimental_run_functions_eagerly(True)
# tf.compat.v1.disable_eager_execution()
import os
import math
from tensorflow.keras.models import load_model
import numpy as np
from scipy.linalg import hadamard
#print(tf.config.get_visible_devices())
np.random.seed(10)

os.environ['CUDA_VISIBLE_DEVICES'] = "1"  # 2
# os.environ['HOMEPATH']
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
batch_size = 128
iterations = 391
num_classes = 100
num_classifiers = 127

maxepoches = 500
learning_rate = 0.05
lr_decay = 1e-6
lr_drop = 20

lam1 = 0.2


class ONINorm(Layer):
    def __init__(self, T=5, norm_groups=1, *args, **kwargs):
        super(ONINorm, self).__init__()
        self.T = T
        self.norm_groups = norm_groups
        self.eps = 1e-5

    def matrix_power3(self, Input):
        B = Input @ Input
        return B @ Input


    def call(self, inputs):
        Z = tf.reshape(inputs, (self.norm_groups, -1, inputs.shape[-1] // self.norm_groups))
        Zc = Z - tf.reduce_mean(Z, axis=1, keepdims=True)
        S = tf.transpose(Zc, perm=[0,2,1]) @ Zc
        eye = tf.expand_dims(tf.eye(S.shape[-1]), axis=0)
        S = S + self.eps*eye
        norm_S = tf.norm(S, ord='fro', axis=(1, 2), keepdims=True)
        S = S / norm_S
        B = [tf.convert_to_tensor([]) for _ in range(self.T + 1)]
        B[0] = tf.expand_dims(tf.eye(S.shape[-1]), axis=0)
        for t in range(self.T):
            #B[t + 1] = torch.baddbmm(1.5, B[t], -0.5, torch.matrix_power(B[t], 3), S)
            # B[t + 1] = torch.baddbmm(1.5, B[t], -0.5, self.matrix_power3(B[t]), S)
            B[t + 1] = 1.5 * B[t] - 0.5 * self.matrix_power3(B[t]) @ S
        W = B[self.T] @ tf.transpose(Zc, perm=[0,2,1]) / tf.math.sqrt(norm_S)
#        print(W @ tf.transpose(W, [0, 2, 1]))
        # W = oni_py.apply(weight, self.T, ctx.groups)
        return tf.reshape(tf.transpose(W, perm=[0,2,1]), shape=inputs.shape)

    def extra_repr(self):
        fmt_str = ['T={}'.format(self.T)]
        if self.norm_groups > 1:
            fmt_str.append('groups={}'.format(self.norm_groups))
        return ', '.join(fmt_str)

class ONI_Conv2d(Conv2D):
    
    
    def __init__(self,
               out_channels,
               kernel_size,
               strides=(1, 1),
               padding='VALID',
               data_format=None,
               dilation_rate=(1, 1),
               groups=1,
               activation=None,
               use_bias=True,
               kernel_initializer='uniform',
               bias_initializer='zeros',
               kernel_regularizer=None,
               T=6, norm_groups=1, NScale=1.414,**kwargs):
        super(ONI_Conv2d, self).__init__(
                filters=out_channels, 
                kernel_size=kernel_size, 
                strides=strides, 
                padding=padding, 
                data_format=data_format, 
                dilation_rate=dilation_rate, 
#                groups=groups, 
                activation=activation,
                use_bias=use_bias, 
                kernel_initializer=kernel_initializer, 
                bias_initializer=bias_initializer, 
                kernel_regularizer=kernel_regularizer,
                **kwargs
               )
         
        
        self.padding = padding
        #print('ONI_Conv_Row:--T=', T, '----norm_groups:', norm_groups, '---NScale:', NScale, '---adjust:', adjustScale)
        self.weight_normalization = ONINorm(T=T, norm_groups=norm_groups)
        self.scale_ = np.ones((1, 1, 1, out_channels))* NScale

        # self.WNScale = tf.constant(self.scale_, dtype=tf.float32)
        
        
        self.diag_w = tf.Variable(np.random.randn(out_channels), dtype=tf.float32)


    def call(self, inputs):
        weight_q = self.weight_normalization(self.kernel)
        
        weight_q=weight_q*tf.tile(tf.reshape(self.diag_w,[1,1,1,-1]),[*weight_q.shape[0:3],1])
        
        
        # weight_q = weight_q * self.WNScale
        
        out = tf.nn.conv2d(inputs, weight_q, self.strides, self.padding)
        outputs = tf.nn.bias_add(out, self.bias)
        if self.activation is not None:
            outputs = self.activation(outputs)
        return outputs
    
    def test_orth(self):
        
        weight_q = self.weight_normalization(self.kernel)
#        weight_q=weight_q*tf.tile(tf.reshape(self.diag_w,[1,1,1,-1]),[*weight_q.shape[0:3],1])
        
        return weight_q
        
    
#    def get_config(self):
#        config = super(ONI_Conv2d, self).get_config()
#        config.update({'out_channels': self.filters})
#        return config


def weight_perturbation(weights, stddev, type):
    for i, layer in enumerate(weights):
        if i % 2 == 0:
            noise = np.random.normal(loc=0.0, scale=stddev, size=layer.shape)
        else:
            continue
        if type == 'awgn':
            layer += noise
        elif type == 'lognormal':
            layer *= np.exp(noise)
    model.set_weights(weights)


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

def vgg_had(input_img, n):
    # Build the network of vgg for 10 classes with massive dropout and weight decay as described in the paper.
    weight_decay = 0.0005
    x = ONI_Conv2d(64, (5, 5), padding='SAME', activation='relu', kernel_regularizer=regularizers.l2(weight_decay))(input_img)

#    x = Conv2D(64, (3, 3), padding='SAME', activation='relu', kernel_regularizer=regularizers.l2(weight_decay))(input_img)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)

    x = Conv2D(64, (3, 3), padding='SAME', activation='relu', kernel_regularizer=regularizers.l2(weight_decay))(x)
    x = BatchNormalization()(x)

    x = MaxPooling2D(pool_size=(2, 2))(x)
    
    x = Conv2D(128, (3, 3), padding='SAME', activation='relu', kernel_regularizer=regularizers.l2(weight_decay))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)

    x = Conv2D(128, (3, 3), padding='SAME', activation='relu', kernel_regularizer=regularizers.l2(weight_decay))(x)
    x = BatchNormalization()(x)

    x = MaxPooling2D(pool_size=(2, 2))(x)
    
    x = Conv2D(256, (3, 3), padding='same', activation='relu', kernel_regularizer=regularizers.l2(weight_decay))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)

    x = Conv2D(256, (3, 3), padding='same', activation='relu', kernel_regularizer=regularizers.l2(weight_decay))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)

    x = Conv2D(256, (3, 3), padding='same', activation='relu', kernel_regularizer=regularizers.l2(weight_decay))(x)
    x = BatchNormalization()(x)

    x = MaxPooling2D(pool_size=(2, 2))(x)
    
    x = Conv2D(512, (3, 3), padding='same', activation='relu', kernel_regularizer=regularizers.l2(weight_decay))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)

    x = Conv2D(512, (3, 3), padding='same', activation='relu', kernel_regularizer=regularizers.l2(weight_decay))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)

    x = Conv2D(512, (3, 3), padding='same', activation='relu', kernel_regularizer=regularizers.l2(weight_decay))(x)
    x = BatchNormalization()(x)

    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = Conv2D(512, (3, 3), padding='same', activation='relu', kernel_regularizer=regularizers.l2(weight_decay))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)

    x = Conv2D(512, (3, 3), padding='same', activation='relu', kernel_regularizer=regularizers.l2(weight_decay))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)

    x = Conv2D(512, (3, 3), padding='SAME', activation='relu', kernel_regularizer=regularizers.l2(weight_decay))(x)
    x = BatchNormalization()(x)

    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.5)(x)

    x = Flatten()(x)
    model_out = []
    
    for i in range(n):
        out = Dense(200, activation='relu', kernel_regularizer=regularizers.l2(weight_decay))(x)
        out = Dense(200, activation='relu', kernel_regularizer=regularizers.l2(weight_decay))(out)
        out = BatchNormalization()(out)

        out = Dropout(0.5)(out)
        output = Dense(1)(out)
        model_out.append(output)
    model_output = tf.keras.layers.concatenate(model_out)
    model = Model(input_img, model_output)
    return model

def normalize(X_train, X_test):
    # this function normalize inputs for zero mean and unit variance
    # it is used when training a model.
    # Input: training set and test set
    # Output: normalized training set and test set according to the trianing set statistics.
    mean = np.mean(X_train, axis=(0, 1, 2, 3))
    std = np.std(X_train, axis=(0, 1, 2, 3))
    X_train = (X_train - mean) / (std + 1e-7)
    X_test = (X_test - mean) / (std + 1e-7)
    return X_train, X_test


def lr_scheduler(epoch):
    return learning_rate * (0.5 ** (epoch // lr_drop))


def corr_bce(lam1):
    def loss(labels, logits):

        y_true = labels
        y_pred = logits

        bce_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)(y_true, y_pred)
        y_pred_bits = tf.math.sigmoid(y_pred)
        y_pred_bits = tf.clip_by_value(y_pred_bits, 0.001, 0.999)


        dec_error = tf.math.abs(y_pred_bits - tf.cast(y_true, tf.float32)) + 0.0001

        cov = tf.transpose(dec_error) @ dec_error / batch_size + 0.001 * tf.eye(num_classifiers)

        t_corr = tf.linalg.trace(tf.math.log(cov)) - tf.linalg.logdet(cov)

        return bce_loss + lam1 * t_corr / num_classifiers

    return loss


if __name__ == '__main__':
    (x_train, y_train), (x_test, y_test) = cifar100.load_data()
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train, x_test = normalize(x_train, x_test)

    input_shape = x_train.shape[1:]

    idx = np.random.permutation(num_classes)
    cm = hadamard(num_classifiers + 1)[idx][:, 1:]
    cm = hadamard(num_classifiers + 1)[1: num_classes + 1][:, 1:]
    cm[np.where(cm == -1)] = 0
    
    np.random.seed(5)

    # Augment labels
    y_train_code = np.zeros((y_train.shape[0], num_classifiers))
    y_test_code = np.zeros((y_test.shape[0], num_classifiers))
    for i in range(num_classes):
        idx_train = list(np.where(y_train == i)[0])
        idx_test = list(np.where(y_test == i)[0])
        y_train_code[idx_train, :] = cm[i]
        y_test_code[idx_test, :] = cm[i]

    train = 1
    if not train:
        dir_pwd = os.getcwd()
        dir_name = 'cifar100/cifar100_vgg16_sep_orthnormal_lam1={}'.format(lam1)
        save_dir = os.path.join(dir_pwd, dir_name)
        model_name = 'model.{epoch:03d}.hdf5'
        model_name = 'model.hdf5'
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        filepath = os.path.join(save_dir, model_name)
        
        
        model_input = Input(shape=input_shape)
        model = vgg_had(model_input, num_classifiers)
        model.load_weights(filepath)
#        model = load_model(filepath, compile=False)
        sgd = optimizers.SGD(lr=0.1, momentum=0.9, nesterov=True)
        # model.compile(loss=tf.keras.losses.BinaryCrossentropy(), optimizer=sgd,
        #               metrics=[new_accuracy])
        model.compile(loss=corr_bce(lam1), optimizer=sgd, metrics=['accuracy'])
        

        weight_org = model.get_weights()
        w = weight_org[0].reshape(-1, weight_org[0].shape[-1])
        a = w.T @ w
        s = []
        for sigma in [0, 0.1, 0.2, 0.3]:
            acc_total = []
            for i in range(100):
                w_org_copy = copy.deepcopy(weight_org)
                weight_perturbation(w_org_copy, sigma, 'lognormal')
                y_pred = model.predict(x_test, batch_size=100)
                acc = validate_accuracy(y_test_code, y_pred)
                acc_total.append(acc)
                print(acc)
            s.append(np.average(acc_total))
            print(sigma, np.average(acc_total))
        print(s)
        
        for rate in [0, 0.001, 0.01, 0.05]:
            acc_total = []
            for i in range(100):
                w_org_copy = copy.deepcopy(weight_org)
                model.set_weights(quantized(w_org_copy, 8, rate, 99.0))
                y_pred = model.predict(x_test, batch_size=batch_size)
                acc = validate_accuracy(y_test_code, y_pred)
                acc_total.append(acc)
                print(acc)
            s.append(np.average(acc_total))
            print(rate, np.average(acc_total))
        print(s)

    else:
        dir_pwd = os.getcwd()
        dir_name = 'cifar100/cifar100_vgg16_sep_orthnormal_lam1={}'.format(lam1)
        save_dir = os.path.join(dir_pwd, dir_name)
        model_name = 'model.hdf5'
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        filepath = os.path.join(save_dir, model_name)

        model_input = Input(shape=input_shape)
        model = vgg_had(model_input, num_classifiers)
        sgd = optimizers.SGD(lr=learning_rate, decay=lr_decay, momentum=0.9, nesterov=True)
        model.compile(loss=corr_bce(lam1), optimizer=sgd, metrics=[new_accuracy])

        model.summary()

        checkpoint = ModelCheckpoint(
            filepath=filepath, monitor='val_loss', mode='min', verbose=1, save_best_only=True, save_weights_only=False)
        lr_scheduler = LearningRateScheduler(lr_scheduler)
        lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1),
                                       cooldown=0,
                                       patience=5,
                                       min_lr=0.5e-6)
        callbacks = [checkpoint, lr_scheduler, lr_reducer]

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
            rotation_range=15,
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
            epochs=maxepoches,
            verbose=1,
            callbacks=callbacks)
        





  
        
