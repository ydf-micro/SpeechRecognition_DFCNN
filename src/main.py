# *_*coding:utf-8 *_*

import os
import numpy as np
from scipy.fftpack import fft
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt
from random import shuffle

# 获取信号的时频图
def compute_fbank(file_path):
    fs, wavsignal = wav.read(file_path)

    x = np.linspace(0, 400 - 1, 400, dtype=np.int64)
    w = .54 - .46 * np.cos(2 * np.pi * x / (400 - 1))  # Hamming window

    time_window = 25   # 一帧25ms
    window_length = fs // 1000 * time_window

    wav_arr = np.array(wavsignal)
    wav_length = len(wavsignal)
    range0_end = int(wav_length / fs * 1000 - time_window) // 10  # 计算有多少窗
    data_input = np.zeros((range0_end, 200), dtype=np.float)  # 存放最终频率特征数据
    # data_line = np.zeros((1, 400), dtype=np.float)

    for i in range(range0_end):
        p_start = i * 160
        p_end = p_start + 400
        data_line = wav_arr[p_start: p_end]
        data_line = data_line * w  # 加窗
        data_line = np.abs(fft(data_line))  # 傅里叶变换
        data_input[i] = data_line[0: 200]  # 傅里叶变换关于y轴对称

    data_input = np.log(data_input + 1)

    return data_input

# 获取音频文件及标签文件列表
def source_get(source_file):
    train_file = source_file + '/data'
    label_lst = []
    wav_lst = []

    for root, dirs, files in os.walk(train_file):
        for file in files:
            if file.endswith('wav') or file.endswith('WAV'):
                wav_file = os.sep.join([root, file])
                label_file = wav_file + '.trn'
                wav_lst.append(wav_file)
                label_lst.append(label_file)

    for i in range(len(label_lst)):
        wavname = (wav_lst[i].split('/')[-1]).split('.')[0]
        labelname = (label_lst[i].split('/')[-1]).split('.')[0]

        if wavname != labelname:
            print('error')

    return label_lst, wav_lst

# 读取音频文件对应的拼音label
def read_label(label_file):
    with open(label_file, 'r', encoding='utf-8') as f:
        data = f.readlines()
        return data[1]

def gen_label_data(label_lst):
    label_data = []
    for label_file in label_lst:
        pny = read_label(label_file)
        label_data.append(pny.strip('\n'))

    return label_data

# 建立label到id的映射字典
def mk_vocab(label_data):
    vocab = []
    for line in label_data:
        line = line.split(' ')
        for pny in line:
            if pny not in vocab:
                vocab.append(pny)

    vocab.append('_')

    return vocab

# 将读取的label映射到对应的id
def word2id(line, vocab):
    return [vocab.index(pny) for pny in line.split(' ')]

# 存放信号的时频图和标签数据
def get_batch(batch_size, shuffle_list, wav_lst, label_data, vocab):
    for i in range(len(wav_lst) // batch_size):
        wav_data_lst = []
        label_data_lst = []
        begin = i * batch_size
        end = begin + batch_size
        sub_list = shuffle_list[begin: end]
        for index in sub_list:
            fbank = compute_fbank(wav_lst[index])
            fbank = fbank[:fbank.shape[0]//8*8, :]
            label = word2id(label_data[index], vocab)
            wav_data_lst.append(fbank)
            label_data_lst.append(label)

        yield wav_data_lst, label_data_lst

def wav_padding(wav_data_lst):
    wav_lens = [len(data) for data in wav_data_lst]
    wav_max_len = max(wav_lens)
    wav_lens = np.array([leng//8 for leng in wav_lens])
    new_wav_data_lst = np.zeros((len(wav_data_lst), wav_max_len, 200, 1))
    for i in range(len(wav_data_lst)):
        new_wav_data_lst[i, :wav_data_lst[i].shape[0], :, 0] = wav_data_lst[i]

    return new_wav_data_lst, wav_lens

def label_padding(laebl_data_lst):
    label_lens = np.array([len(label) for label in label_data_lst])
    max_label_len = max(label_lens)
    new_label_data_lst = np.zeros((len(label_data_lst), max_label_len))
    for i in range(len(label_data_lst)):
        new_label_data_lst[i, :len(label_data_lst[i])] = label_data_lst[i]

    return new_label_data_lst, label_lens

def data_generator(batch_size, shuffle_list, wav_lst, label_data, vocab):
    for i in range(len(wav_lst) // batch_size):
        wav_data_lst = []
        label_data_lst = []
        begin = i * batch_size
        end = begin + batch_size
        sub_list = shuffle_list[begin: end]
        for index in sub_list:
            fbank = compute_fbank(wav_lst[index])
            pad_fbank = np.zeros((fbank.shape[0] // 8*8+8, fbank.shape[1]))
            pad_fbank[:fbank.shape[0], :] = fbank
            label = word2id(label_data[index], vocab)
            wav_data_lst.append(pad_fbank)
            label_data_lst.append(label)
        pad_wav_data, input_length = wav_padding(wav_data_lst)
        pad_label_data, label_length = label_padding(label_data_lst)
        inputs = {'the_inputs': pad_wav_data,
                  'the_labels': pad_label_data,
                  'input_length': input_length,
                  'label_length': label_length}
        outputs = {'ctc': np.zeros(pad_wav_data.shape[0],)}
        yield inputs, outputs

if __name__ == '__main__':
    # wavepath = '../data/test.wav'
    #
    # a = compute_fbank(wavepath)
    # plt.imshow(a.T, origin='lower')
    # plt.show()

    source_file = '/home/y_lab/data_thchs30'
    label_lst, wav_lst = source_get(source_file)

    # 打乱数据
    shuffle_list = [i for i in range(len(label_lst))]
    shuffle(shuffle_list)

    # # print(label_lst[:10])
    # # print(wav_lst[:10])
    #
    # # print(read_label(label_lst[0]))
    #
    label_data = gen_label_data(label_lst)
    #
    # print(len(label_data))
    #
    vocab = mk_vocab(label_data)
    #
    # print(len(vocab))
    #
    # label_id = word2id(label_data[0], vocab)
    #
    # print(label_data[0])
    # print(label_id)
    #
    # print(vocab[:15])
    # print(label_data[10])
    # print(word2id(label_data[10], vocab))
    #
    # fbank = compute_fbank(wav_lst[0])
    #
    # print(fbank.shape)
    # plt.imshow(fbank.T, origin='lower')
    # plt.show()

    batch = get_batch(4, shuffle_list, wav_lst, label_data, vocab)
    wav_data_lst, label_data_lst = next(batch)
    # lens = [len(wav) for wav in wav_data_lst]
    # print(max(lens))
    # print(lens)
    # for wav_data in wav_data_lst:
    #     print(wav_data.shape)
    # for label_data in label_data_lst:
    #     print(label_data)

    pad_wav_data_lst, wav_lens = wav_padding(wav_data_lst)
    print(pad_wav_data_lst.shape)
    print(wav_lens)

    pad_label_data_lst, label_lens = label_padding(label_data_lst)
    print(pad_label_data_lst.shape)
    print(label_lens)