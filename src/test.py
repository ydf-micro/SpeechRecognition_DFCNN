# *_*coding:utf-8 *_*

import numpy as np
from keras import backend as K
from data_process import source_get, gen_label_data, mk_vocab, data_generator
from random import shuffle
from keras.models import load_model
from CNN_CTC import ctc_lambda

def decode_ctc(num_result, num2word):
    result = num_result[:, :, :]
    in_len = np.zeros((1), dtype=np.int32)
    in_len[0] = result.shape[1]
    r = K.ctc_decode(result, in_len, greedy=True, beam_width=10, top_paths=1)
    r1 = K.get_value(r[0][0])
    r1 = r1[0]
    text = []
    for i in r1:
        text.append(num2word[i])

    return r1, text

if __name__ == '__main__':
    source_file = '/home/y_lab/data_thchs30'
    label_lst, wav_lst = source_get(source_file)

    label_data = gen_label_data(label_lst[:100])

    vocab = mk_vocab(label_data)

    shuffle_list = [i for i in range(100)]

    # shuffle(shuffle_list)
    batch = data_generator(1, shuffle_list, wav_lst, label_data, vocab)

    model = load_model('../models/model.h5', custom_objects={'<lambda>': lambda y_true, output: output})
    #
    for i in range(10):
        inputs, outputs = next(batch)
        x = inputs['the_inputs']
        y = inputs['the_labels'][0]

        result = model.predict(x, steps=1)
        #
        # result, text = decode_ctc(result, vocab)
        #
        # print('数字结果：', result)
        # print('文本结果：', text)
        # print('原文结果：', [vocab[int(i)] for i in y])