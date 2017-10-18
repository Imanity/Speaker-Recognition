import wave
import configs
import scipy.io.wavfile
import numpy as np
import matplotlib.pyplot as plt 

def partition(filename):
    sample_rate, data = scipy.io.wavfile.read(filename)
    plt.subplot(2, 1, 1)
    plt.title("1")
    plt.plot(data)
    plt.show()

partition('output.wav')
