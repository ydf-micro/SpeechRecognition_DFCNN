# *_*coding:utf-8 *_*

import numpy as np
from scipy.fftpack import fft
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt

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

if __name__ == '__main__':
    wavepath = '../data/test.wav'

    a = compute_fbank(wavepath)
    plt.imshow(a.T, origin='lower')
    plt.show()
