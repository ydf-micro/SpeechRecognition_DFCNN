# *_*coding:utf-8 *_*

import keras
from keras.layers import Input, Conv2D, BatchNormalization, MaxPooling2D
from keras.layers import Reshape, Dense, Lambda
from keras.optimizers import Adam
from keras import backend as K
from keras.models import Model
from keras.utils import multi_gpu_model
from data_process import source_get, gen_label_data, mk_vocab, data_generator
from random import shuffle
from keras.callbacks import ModelCheckpoint

def conv2d(size):
    return Conv2D(size, (3, 3), use_bias=True, activation='relu',
                  padding='same', kernel_initializer='he_normal')

def norm(x):
    return BatchNormalization(axis=-1)(x)

def maxpool(x):
    return MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid')(x)

def dense(units, activation='relu'):
    return Dense(units, activation=activation, use_bias=True, kernel_initializer='he_normal')

def cnn_cell(size, x, pool=True):
    x = norm(conv2d(size)(x))
    x = norm(conv2d(size)(x))
    if pool:
        x = maxpool(x)

    return x

def ctc_lambda(args):
    labels, y_pred, input_length, label_length = args
    y_pred = y_pred[:, :, :]
    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)

class CNN_CTC_Model:
    def __init__(self, vocab_size):
        super().__init__()
        self.vocab_size = vocab_size
        self._model_init()
        self._ctc_init()
        self.opt_init()

    def _model_init(self):
        self.inputs = Input(name='the_inputs', shape=(None, 200, 1))
        self.h1 = cnn_cell(32, self.inputs)
        self.h2 = cnn_cell(64, self.h1)
        self.h3 = cnn_cell(128, self.h2)
        self.h4 = cnn_cell(128, self.h3, pool=False)

        self.h6 = Reshape((-1, 3200))(self.h4)
        self.h7 = dense(256)(self.h6)
        self.outputs = dense(self.vocab_size, activation='softmax')(self.h7)
        self.model = Model(inputs=self.inputs, outputs=self.outputs)

    def _ctc_init(self):
        self.labels = Input(name='the_labels', shape=[None], dtype='float32')
        self.input_length = Input(name='input_length', shape=[1], dtype='int64')
        self.label_length = Input(name='label_length', shape=[1], dtype='int64')
        self.loss_out = Lambda(ctc_lambda, output_shape=(1, ), name='ctc')\
            ([self.labels, self.outputs, self.input_length, self.label_length])
        self.ctc_model = Model(inputs=[self.labels, self.inputs, self.input_length, self.label_length],
                                outputs=self.loss_out)

    def opt_init(self):
        opt = Adam(lr=.0008, beta_1=.9, beta_2=.999, decay=.01, epsilon=10e-8)
        self.ctc_model.compile(loss={'ctc': lambda y_true, output: output}, optimizer=opt)

if __name__ == "__main__":
    # models = CNN_CTC_Model(1176)
    # models.ctc_model.summary()

    source_file = '/home/y_lab/data_thchs30'
    label_lst, wav_lst = source_get(source_file)

    label_data = gen_label_data(label_lst[:100])

    vocab = mk_vocab(label_data)

    total_nums = 100  # len(label_lst)
    batch_size = 4
    batch_num = total_nums // batch_size
    epochs = 50

    shuffle_list = [i for i in range(100)]

    model = CNN_CTC_Model(len(vocab))

    for k in range(epochs):
        print('this is the ', k+1, 'th epochs training !!!')
        shuffle(shuffle_list)
        batch = data_generator(batch_size, shuffle_list, wav_lst, label_data, vocab)
        filepath = '../logs/checkpoints/model_best.hdf5'
        checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True)
        model.ctc_model.fit_generator(batch, steps_per_epoch=batch_num, epochs=1, callbacks=[checkpoint])

    model.ctc_model.save('../models/model.h5')