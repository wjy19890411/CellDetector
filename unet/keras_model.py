import os

import numpy as np
import tensorflow as tf
from keras import backend as K
from keras import callbacks
from keras import models
from keras import layers
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.core import Dropout, Lambda
from keras.layers.merge import concatenate
from keras.layers.pooling import MaxPooling2D

import configs
import read_data

IMG_HEIGHT = configs.IMG_HEIGHT
IMG_WIDTH = configs.IMG_WIDTH
IMG_CHANNELS = configs.IMG_CHANNEL

MODEL_FILE = 'model-dsbowl2018-1.h5'


# Define IoU metric
def mean_iou(y_true, y_pred):
    prec = []
    for t in np.arange(0.5, 1.0, 0.05):
        y_pred_ = tf.to_int32(y_pred > t)
        score, up_opt = tf.metrics.mean_iou(y_true, y_pred_, 2)
        K.get_session().run(tf.local_variables_initializer())
        with tf.control_dependencies([up_opt]):
            score = tf.identity(score)
        prec.append(score)
    return K.mean(K.stack(prec), axis=0)


def _build_model():
    inputs = layers.Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
    s = Lambda(lambda x: x / 255)(inputs)

    c1 = Conv2D(
        16, (3, 3),
        activation='elu',
        kernel_initializer='he_normal',
        padding='same')(s)
    c1 = Dropout(0.1)(c1)
    c1 = Conv2D(
        16, (3, 3),
        activation='elu',
        kernel_initializer='he_normal',
        padding='same')(c1)
    p1 = MaxPooling2D((2, 2))(c1)

    c2 = Conv2D(
        32, (3, 3),
        activation='elu',
        kernel_initializer='he_normal',
        padding='same')(p1)
    c2 = Dropout(0.1)(c2)
    c2 = Conv2D(
        32, (3, 3),
        activation='elu',
        kernel_initializer='he_normal',
        padding='same')(c2)
    p2 = MaxPooling2D((2, 2))(c2)

    c3 = Conv2D(
        64, (3, 3),
        activation='elu',
        kernel_initializer='he_normal',
        padding='same')(p2)
    c3 = Dropout(0.2)(c3)
    c3 = Conv2D(
        64, (3, 3),
        activation='elu',
        kernel_initializer='he_normal',
        padding='same')(c3)
    p3 = MaxPooling2D((2, 2))(c3)

    c4 = Conv2D(
        128, (3, 3),
        activation='elu',
        kernel_initializer='he_normal',
        padding='same')(p3)
    c4 = Dropout(0.2)(c4)
    c4 = Conv2D(
        128, (3, 3),
        activation='elu',
        kernel_initializer='he_normal',
        padding='same')(c4)
    p4 = MaxPooling2D(pool_size=(2, 2))(c4)

    c5 = Conv2D(
        256, (3, 3),
        activation='elu',
        kernel_initializer='he_normal',
        padding='same')(p4)
    c5 = Dropout(0.3)(c5)
    c5 = Conv2D(
        256, (3, 3),
        activation='elu',
        kernel_initializer='he_normal',
        padding='same')(c5)

    u6 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c5)
    u6 = concatenate([u6, c4])
    c6 = Conv2D(
        128, (3, 3),
        activation='elu',
        kernel_initializer='he_normal',
        padding='same')(u6)
    c6 = Dropout(0.2)(c6)
    c6 = Conv2D(
        128, (3, 3),
        activation='elu',
        kernel_initializer='he_normal',
        padding='same')(c6)

    u7 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c6)
    u7 = concatenate([u7, c3])
    c7 = Conv2D(
        64, (3, 3),
        activation='elu',
        kernel_initializer='he_normal',
        padding='same')(u7)
    c7 = Dropout(0.2)(c7)
    c7 = Conv2D(
        64, (3, 3),
        activation='elu',
        kernel_initializer='he_normal',
        padding='same')(c7)

    u8 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c7)
    u8 = concatenate([u8, c2])
    c8 = Conv2D(
        32, (3, 3),
        activation='elu',
        kernel_initializer='he_normal',
        padding='same')(u8)
    c8 = Dropout(0.1)(c8)
    c8 = Conv2D(
        32, (3, 3),
        activation='elu',
        kernel_initializer='he_normal',
        padding='same')(c8)

    u9 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c8)
    u9 = concatenate([u9, c1], axis=3)
    c9 = Conv2D(
        16, (3, 3),
        activation='elu',
        kernel_initializer='he_normal',
        padding='same')(u9)
    c9 = Dropout(0.1)(c9)
    c9 = Conv2D(
        16, (3, 3),
        activation='elu',
        kernel_initializer='he_normal',
        padding='same')(c9)

    outputs = Conv2D(1, (1, 1), activation='sigmoid')(c9)

    model = models.Model(inputs=[inputs], outputs=[outputs])
    model.compile(
        optimizer='adam', loss='binary_crossentropy', metrics=[mean_iou])
    model.summary()
    return model


def train_input():
    X_train, Y_train = read_data.read_train_data()
    Y_train = np.expand_dims(Y_train, axis=-1)
    return X_train, Y_train


class KerasUnet(object):
    def __init__(self):
        self.model_file = 'model-dsbowl2018-1.h5'
        if self.model_file in os.listdir('.'):
            print('Loading existing model...')
            self.model = models.load_model(
                'model-dsbowl2018-1.h5', custom_objects={'mean_iou': mean_iou})
        else:
            print('Crreating new model...')
            self.model = build_model()

    def fit(self):
        X_train, Y_train = train_input()
        earlystopper = callbacks.EarlyStopping(patience=5, verbose=1)
        checkpointer = callbacks.ModelCheckpoint(
            self.model_file, verbose=1, save_best_only=True)
        results = self.model.fit(
            X_train,
            Y_train,
            validation_split=0.1,
            batch_size=16,
            epochs=5,
            callbacks=[earlystopper, checkpointer])

    def predict(self):
        X_test = read_data.read_test_data()
        return self.model.predict(X_test, verbose=1)
