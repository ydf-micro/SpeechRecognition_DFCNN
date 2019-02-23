# *_*coding:utf-8 *_*

import os
import numpy as np
from scipy.fftpack import fft
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt


wavpath = '../data/test.wav'

fs, wavsignal = wav.read(wavpath)

print(wavsignal.shape)

plt.plot(wavsignal)
plt.title('Waveprint')
plt.show()

x = np.linspace(0, 400-1, 400, dtype=np.int64)
w = .54 - .46 * np.cos(2 * np.pi * x / (400 - 1))   # Hamming window

time_window = 25
window_length = fs // 1000 * time_window

# 分帧
p_begin = 0
p_end = p_begin + window_length
frame = wavsignal[p_begin: p_end]

plt.plot(frame)
plt.title('Framing')
plt.show()

# 加窗
frame = frame * w

plt.plot(frame)
plt.title('Windowing')
plt.show()

# 进行傅里叶变换
frame_fft = np.abs(fft(frame))[:200]
plt.plot(frame_fft)
plt.title('FFT')
plt.show()

# 取对数

frame_log = np.log(frame_fft)

plt.plot(frame_log)
plt.title('log power spectrum')
plt.show()