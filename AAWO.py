import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPooling2D, BatchNormalization, Activation, Dropout, Input, Layer
# from tensorflow.keras.callbacks import LearningRateScheduler, TensorBoard
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, LearningRateScheduler
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.constraints import Constraint
from scipy.spatial import distance

import numpy as np

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
        S = tf.transpose(Zc, perm=[0 ,2 ,1]) @ Zc
        eye = tf.expand_dims(tf.eye(S.shape[-1]), axis=0)
        S = S + self.eps *eye
        norm_S = tf.norm(S, ord='fro', axis=(1, 2), keepdims=True)
        S = S / norm_S
        B = [tf.convert_to_tensor([]) for _ in range(self.T + 1)]
        B[0] = tf.expand_dims(tf.eye(S.shape[-1]), axis=0)
        for t in range(self.T):
            # B[t + 1] = torch.baddbmm(1.5, B[t], -0.5, torch.matrix_power(B[t], 3), S)
            # B[t + 1] = torch.baddbmm(1.5, B[t], -0.5, self.matrix_power3(B[t]), S)
            B[t + 1] = 1.5 * B[t] - 0.5 * self.matrix_power3(B[t]) @ S
        W = B[self.T] @ tf.transpose(Zc, perm=[0 ,2 ,1]) / tf.math.sqrt(norm_S)
        #        print(W @ tf.transpose(W, [0, 2, 1]))
        # W = oni_py.apply(weight, self.T, ctx.groups)
        return tf.reshape(tf.transpose(W, perm=[0 ,2 ,1]), shape=inputs.shape)

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
                 T=6, norm_groups=1, NScale=1.414 ,**kwargs):
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
        # print('ONI_Conv_Row:--T=', T, '----norm_groups:', norm_groups, '---NScale:', NScale, '---adjust:', adjustScale)
        self.weight_normalization = ONINorm(T=T, norm_groups=norm_groups)
        self.scale_ = np.ones((1, 1, 1, out_channels) )* NScale

        # self.WNScale = tf.constant(self.scale_, dtype=tf.float32)


        self.diag_w = tf.Variable(np.random.randn(out_channels), dtype=tf.float32)


    def call(self, inputs):
        weight_q = self.weight_normalization(self.kernel)

        weight_q =weight_q *tf.tile(tf.reshape(self.diag_w ,[1 ,1 ,1 ,-1]) ,[*weight_q.shape[0:3] ,1])


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
        self.scale_ = np.ones((1, units)) * NScale

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