import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend
from tensorflow.keras import optimizers, regularizers
from tensorflow.keras.datasets import cifar10, mnist
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPooling2D, BatchNormalization, Activation, Dropout, Input, Layer, ReLU
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, LearningRateScheduler
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.constraints import Constraint
from scipy.spatial import distance
from utilities import *

# tf.config.experimental_run_functions_eagerly(True)
# tf.compat.v1.disable_eager_execution()
import os
import math
from tensorflow.keras.models import load_model
import numpy as np
import copy
from scipy.linalg import hadamard
os.environ['CUDA_VISIBLE_DEVICES'] = "1" #2
# os.environ['HOMEPATH']
# os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
batch_size = 128
iterations = 391
num_classes = 10
num_classifiers = 15

# def rmse(y_true, y_pred):
# 	return backend.sqrt(backend.mean(backend.square(backend.round(y_pred) - y_true), axis=-1))
batch_size = 128
hidden_units1 = 125
hidden_units2 = 256
dropout = 0.2  # 0.45

lam1 = 0
lam2 = 0



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
        k = inputs.shape
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
        #print(W.matmul(W.transpose(1,2)))
        # W = oni_py.apply(weight, self.T, ctx.groups)
        return tf.reshape(tf.transpose(W, perm=[0,2,1]), shape=k)

    def extra_repr(self):
        fmt_str = ['T={}'.format(self.T)]
        if self.norm_groups > 1:
            fmt_str.append('groups={}'.format(self.norm_groups))
        return ', '.join(fmt_str)

class ONI_Conv2d(Conv2D):
    
    
    def __init__(self,
               filters,
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
               T=5, norm_groups=1, NScale=1.414,**kwargs):
        super(ONI_Conv2d, self).__init__(
                filters=filters, 
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
                **kwargs
               )
         
        
        self.padding = padding
        #print('ONI_Conv_Row:--T=', T, '----norm_groups:', norm_groups, '---NScale:', NScale, '---adjust:', adjustScale)
        self.weight_normalization = ONINorm(T=T, norm_groups=norm_groups)
        self.scale_ = np.ones((1, 1, 1, filters))* NScale

        self.WNScale = tf.constant(self.scale_, dtype=tf.float32)


    def call(self, inputs):
        weight_q = self.weight_normalization(self.kernel)
        weight_q = weight_q * self.WNScale
        
        out = tf.nn.conv2d(inputs, weight_q, self.strides, self.padding)
        outputs = tf.nn.bias_add(out, self.bias)
        if self.activation is not None:
            outputs = self.activation(outputs)
        return outputs
    
class ONI_linear(Dense):
    
    
    def __init__(self,
                units,
                activation=None,
                use_bias=True,
                kernel_initializer='glorot_uniform',
                bias_initializer='zeros',
                kernel_regularizer=None,
                bias_regularizer=None,
                activity_regularizer=None,
                kernel_constraint=None,
                bias_constraint=None,
                T=5, norm_groups=1, NScale=1.414, **kwargs):
        super(ONI_linear, self).__init__(units,
                activation=None,
                use_bias=True,
                kernel_initializer='glorot_uniform',
                bias_initializer='zeros',
                kernel_regularizer=None,
                bias_regularizer=None,
                activity_regularizer=None,
                kernel_constraint=None,
                bias_constraint=None, **kwargs)
        self.weight_normalization = ONINorm(T=T, norm_groups=norm_groups)
        self.scale_ = np.ones((1, units))* NScale
        
        self.WNScale = tf.constant(self.scale_, dtype=tf.float32)
        
    def call(self, inputs):
        weight_q = self.weight_normalization(self.kernel)
        weight_q = weight_q * self.WNScale

        out = inputs @ weight_q
#        print(out)
        outputs = tf.nn.bias_add(out, self.bias)
        if self.activation is not None:
            outputs = self.activation(outputs)
        return outputs
        

def shared():
    shared = Dense(2)
    return shared

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

def lenet5_mnist_sep(img_input, n, shared_dense=None):
    x = ONI_Conv2d(6, (5, 5), padding='VALID', activation='relu', kernel_initializer='he_normal',
               input_shape=(28, 28, 1))(img_input)
#    x = Conv2D(6, (5, 5), padding='valid', activation='relu', kernel_initializer='he_normal',
#               input_shape=(28, 28, 1))(img_input)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)
    x = ONI_Conv2d(16, (5, 5), padding='VALID', activation='relu', kernel_initializer='he_normal')(x)
#    x = Conv2D(16, (5, 5), padding='valid', activation='relu', kernel_initializer='he_normal')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)
    x = Flatten()(x)
    x = ONI_linear(120, activation='relu', kernel_initializer='he_normal')(x)
#    x = Dense(120, activation='relu', kernel_initializer='he_normal')(x)
    model_out = []
    for i in range(n):
        out = Dense(10, activation='relu', kernel_initializer='he_normal')(x)
        if shared_dense:
            out = shared_dense(out)
        else:
            out = Dense(1)(out)
        model_out.append(out)
    model_output = tf.keras.layers.concatenate(model_out)
    model = Model(img_input, model_output)
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
    # load data
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    # (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    y_test_predict = y_test

    image_size = x_train.shape[1]
    # print(image_size)
    input_size = image_size * image_size
    # x_train = np.reshape(x_train, [-1, input_size, 1])
    x_train = np.reshape(x_train, [-1, image_size, image_size, 1])
    x_test = np.reshape(x_test, [-1, image_size, image_size, 1])
    # x_test = x_test.astype('float32') / 255

    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.

    input_shape = x_train.shape[1:]

    cm = hadamard(num_classifiers+1)[1:11][:,1:]
    cm[np.where(cm == -1)] = 0


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
        dir_name = 'mnist/mnist_hadamard15_lenet5_sep_orth_lam1={}_2'.format(lam1)
        save_dir = os.path.join(dir_pwd, dir_name)
        model_name = 'model.051.hdf5'
        model_name = 'model.101.hdf5'
#        model_name = 'model.hdf5'
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        filepath = os.path.join(save_dir, model_name)
        
        model_input = Input(shape=input_shape)
        model = lenet5_mnist_sep(model_input, num_classifiers)
        model.load_weights(filepath)

        model.summary()
        weight_org = model.get_weights()
        s = []
        for sigma in [0, 0.1, 0.4, 0.7]:
            acc_total = []
            for i in range(100):
                w_org_copy = copy.deepcopy(weight_org)
                weight_perturbation(w_org_copy, sigma, 'lognormal')
                y_pred = model.predict(x_test, batch_size=100)
                accuracy = validate_accuracy(y_test_code, y_pred)
                # score, acc = model.evaluate(x_test, y_test, batch_size=batch_size)
                acc_total.append(accuracy)
                #print(accuracy)
            s.append(np.average(acc_total))
            print(sigma, np.average(acc_total))
        print(s)
        
        for rate in [0, 0.01, 0.05, 0.10]:
            acc_total = []
            for i in range(100):
                w_org_copy = copy.deepcopy(weight_org)
                model.set_weights(quantized(w_org_copy, 8, rate, 98))
                y_pred = model.predict(x_test, batch_size=100)
                acc = validate_accuracy(y_test_code, y_pred)
#                score, acc = model.evaluate(x_test, y_test, batch_size=batch_size)
                acc_total.append(acc)
                #print(acc)
            s.append(np.average(acc_total))
            print(rate, np.average(acc_total))
        print(s)



    else:
        dir_pwd = os.getcwd()
        dir_name = 'mnist/mnist_hadamard{}_lenet5_sep_orth_lam1={}_4'.format(num_classifiers, lam1)
        save_dir = os.path.join(dir_pwd, dir_name)
        model_name = 'model.hdf5'
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        filepath = os.path.join(save_dir, model_name)

        model_input = Input(shape=input_shape)
        model = lenet5_mnist_sep(model_input, num_classifiers)
        model.summary()
        print(filepath)

        sgd = optimizers.SGD(lr=0.01, momentum=0.9, nesterov=True)
        # adam = optimizers.adam(lr=0.01, momentum=0.9, nesterov=True)
        model.compile(loss=corr_bce(lam1), optimizer='adam',
                      metrics=[new_accuracy])  # binary_accuracy
        # model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

        checkpoint = ModelCheckpoint(
            filepath=filepath, monitor='val_loss', mode='min', verbose=1, save_best_only=True, save_weights_only=False)
        lr_scheduler = LearningRateScheduler(lr_schedule)
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
            epochs=300,
            verbose=1,
            callbacks=callbacks)


