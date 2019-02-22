# *_*coding:utf-8 *_*

import os
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt

filepath = '../data/test.wav'

fs, wavsignal = wav.read(filepath)  # fs-->data  wavsignal-->sampling rate

plt.plot(wavsignal)
plt.show()

