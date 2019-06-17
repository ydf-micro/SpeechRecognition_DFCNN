# *_*coding:utf-8 *_*

import numpy as np
from keras import backend as K
from data_process import source_get, gen_label_data, mk_vocab, data_generator
from random import shuffle
from keras.models import Model, load_model
from keras.layers import Input, Reshape
from CNN_CTC import cnn_cell, dense

def decode_ctc(num_result, num2word):
    result = num_result[:, :, :]
    in_len = np.zeros((1), dtype = np.int32)
    in_len[0] = result.shape[1]
    r = K.ctc_decode(result, in_len, greedy=True, beam_width=10, top_paths=1)
    r1 = K.get_value(r[0][0])
    r1 = r1[0]
    text = []
    for i in r1:
        text.append(num2word[i])

    return r1, text


class predict_Model:
    def __init__(self, vocab_size):
        super().__init__()
        self.vocab_size = vocab_size
        self._model_init()

    def _model_init(self):
        self.inputs = Input(name='the_inputs', shape=(None, 200, 1))
        self.h1 = cnn_cell(32, self.inputs)
        self.h2 = cnn_cell(64, self.h1)
        self.h3 = cnn_cell(128, self.h2)
        self.h4 = cnn_cell(128, self.h3, pool=False)
        # 200 / 8 * 128 = 3200
        self.h6 = Reshape((-1, 3200))(self.h4)
        self.h7 = dense(256)(self.h6)
        self.outputs = dense(self.vocab_size, activation='softmax')(self.h7)
        self.model = Model(inputs=self.inputs, outputs=self.outputs)


if __name__ == '__main__':
    source_file = '/home/ydf_micro/datasets/data_thchs30'
    label_lst, wav_lst = source_get(source_file)

    label_data = gen_label_data(label_lst)

    vocab = mk_vocab(label_data)
    # print(len(vocab))

    shuffle_list = [i for i in range(len(label_data))]

    shuffle(shuffle_list)

    # model = load_model('../models/model_250.h5', custom_objects={'<lambda>': lambda y_true, output: output})
    # print(model.summary())
    # model.save_weights('../models/model_weights.h5')

    predict_model = predict_Model(len(vocab))
    predict_model.model.load_weights('../models/model_weights.h5', by_name=True)


    print(predict_model.model.summary())

    # 测试模型 predict(x, batch_size=None, verbose=0, steps=None)
    batch = data_generator(1, shuffle_list, wav_lst, label_data, vocab)
    for i in range(10):
        # 载入训练好的模型，并进行识别
        inputs, outputs = next(batch)
        x = inputs['the_inputs']
        y = inputs['the_labels'][0]

        result = predict_model.model.predict(x, steps=1)
        # 将数字结果转化为文本结果
        result, text = decode_ctc(result, vocab)
        print('数字结果： ', result)
        print('文本结果：', text)
        print('原文结果：', [vocab[int(i)] for i in y], '\n')
