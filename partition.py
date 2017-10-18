import pyaudio
import wave
import configs
import scipy.io.wavfile
import numpy as np
import matplotlib.pyplot as plt 

def partition(in_filename, output_filename):
    (sample_rate, data) = scipy.io.wavfile.read(in_filename)
    
    begin = -1
    blank_time = 0
    partitions = []

    for i in range(0, len(data)):
        if data[i] > configs.min_volume:
            blank_time = 0
            if begin < 0:
                begin = i
        else:
            blank_time += 1
        
        if blank_time >= configs.blank_time_gap and begin > 0:
            end = i - configs.blank_time_gap
            if end - begin > configs.min_continuous_time:
                partitions.append((begin, end))
            begin = -1
    
    '''
    print(partitions)

    plt.subplot(2, 1, 1)
    plt.title("1")
    plt.plot(data)
    plt.show()
    '''
    
    for i in range(0, len(partitions)):
        filename = output_filename + str(i) + '.wav'
        scipy.io.wavfile.write(filename, sample_rate, data[(partitions[i][0]):(partitions[i][1])])
    
if __name__ == "__main__":
    partition('dialogue.wav', 'part_')
