import scipy.io.wavfile
import numpy as np
import matplotlib.pyplot as plt
import configs

def partition(in_filename, output_filename):
    (sample_rate, data) = scipy.io.wavfile.read(in_filename)

    # Lower sample rate
    wav_data_origin = []
    wav_data_index = 0
    wav_data_value = 0
    for i in range(0, len(data)):
        wav_data_value += abs(data[i])
        wav_data_index += 1
        if wav_data_index >= configs.audioSegmentLength:
            wav_data_origin.append(wav_data_value / wav_data_index)
            wav_data_index = 0
            wav_data_value = 0
    if wav_data_index != 0:
        wav_data_origin.append(wav_data_value / wav_data_index)

    # Conv Layer
    wav_data = []
    curr_window_sum = 0
    half_window_size = int(configs.windowSize / 2)
    curr_window_size = half_window_size + 1
    for i in range(0, curr_window_size):
        curr_window_sum += wav_data_origin[i]
    wav_data.append(curr_window_sum / curr_window_size)
    for i in range(1, len(wav_data_origin)):
        if i > half_window_size:
            curr_window_sum -= wav_data_origin[i - half_window_size - 1]
            curr_window_size -= 1
        if i < len(wav_data_origin) - half_window_size:
            curr_window_sum += wav_data_origin[i + half_window_size]
            curr_window_size += 1
        wav_data.append(curr_window_sum / curr_window_size)

    # Partition
    curr_start = 0
    curr_status = 0
    partitionOrigin = []
    for i in range(0, len(wav_data)):
        if wav_data[i] > configs.volumeThreshold:
            if curr_status == 0:
                if i > curr_start:
                    partitionOrigin.append({ 'start': curr_start, 'end': i - 1, 'myType': 'empty' })
                curr_status = 1
                curr_start = i
        else:
            if curr_status == 1:
                if i > curr_start:
                    partitionOrigin.append({ 'start': curr_start, 'end': i - 1, 'myType': 'speech' })
                curr_status = 0
                curr_start = i
    if len(wav_data) - 1 > curr_start:
        if curr_status == 0:
            partitionOrigin.append({ 'start': curr_start, 'end': len(wav_data) - 1, 'myType': 'empty' })
        else:
            partitionOrigin.append({ 'start': curr_start, 'end': len(wav_data) - 1, 'myType': 'speech'})
    partition_tmp = []
    for single in partitionOrigin:
        single_len = single['end'] - single['start']
        if single_len > configs.minSize:
            partition_tmp.append(single)
    partitions = []
    last_type = -1
    for single in partition_tmp:
        if single['myType'] != last_type:
            partitions.append(single)
        else:
            partitions[len(partitions) - 1]['end'] = single['end']
        last_type = single['myType']
    
    # Partition
    print(partitions)

    # Output wav
    wav_index = 0
    for wav_segment_info in partitions:
        if wav_segment_info['myType'] == 'speech':
            wav_index += 1
            begin = wav_segment_info['start'] * configs.audioSegmentLength
            end = wav_segment_info['end'] * configs.audioSegmentLength
            scipy.io.wavfile.write(output_filename + str(wav_index) + '.wav', sample_rate, data[begin:end])
    
    
    # Plot
    plt.subplot(2, 1, 1)
    plt.title("1")
    plt.plot(wav_data)
    plt.show()
    
def getParts(in_filename):
    (sample_rate, data) = scipy.io.wavfile.read(in_filename)

    # Lower sample rate
    wav_data_origin = []
    wav_data_index = 0
    wav_data_value = 0
    for i in range(0, len(data)):
        wav_data_value += abs(data[i])
        wav_data_index += 1
        if wav_data_index >= configs.audioSegmentLength:
            wav_data_origin.append(wav_data_value / wav_data_index)
            wav_data_index = 0
            wav_data_value = 0
    if wav_data_index != 0:
        wav_data_origin.append(wav_data_value / wav_data_index)

    # Conv Layer
    wav_data = []
    curr_window_sum = 0
    half_window_size = int(configs.windowSize / 2)
    curr_window_size = half_window_size + 1
    for i in range(0, curr_window_size):
        curr_window_sum += wav_data_origin[i]
    wav_data.append(curr_window_sum / curr_window_size)
    for i in range(1, len(wav_data_origin)):
        if i > half_window_size:
            curr_window_sum -= wav_data_origin[i - half_window_size - 1]
            curr_window_size -= 1
        if i < len(wav_data_origin) - half_window_size:
            curr_window_sum += wav_data_origin[i + half_window_size]
            curr_window_size += 1
        wav_data.append(curr_window_sum / curr_window_size)

    # Partition
    curr_start = 0
    curr_status = 0
    partitionOrigin = []
    for i in range(0, len(wav_data)):
        if wav_data[i] > configs.volumeThreshold:
            if curr_status == 0:
                if i > curr_start:
                    partitionOrigin.append({ 'start': curr_start, 'end': i - 1, 'myType': 'empty' })
                curr_status = 1
                curr_start = i
        else:
            if curr_status == 1:
                if i > curr_start:
                    partitionOrigin.append({ 'start': curr_start, 'end': i - 1, 'myType': 'speech' })
                curr_status = 0
                curr_start = i
    if len(wav_data) - 1 > curr_start:
        if curr_status == 0:
            partitionOrigin.append({ 'start': curr_start, 'end': len(wav_data) - 1, 'myType': 'empty' })
        else:
            partitionOrigin.append({ 'start': curr_start, 'end': len(wav_data) - 1, 'myType': 'speech'})
    partition_tmp = []
    for single in partitionOrigin:
        single_len = single['end'] - single['start']
        if single_len > configs.minSize:
            partition_tmp.append(single)
    partitions = []
    last_type = -1
    for single in partition_tmp:
        if single['myType'] != last_type:
            partitions.append(single)
        else:
            partitions[len(partitions) - 1]['end'] = single['end']
        last_type = single['myType']
    
    # Partition
    #print(partitions)

    # Output wav
    results = []
    wav_index = 0
    for wav_segment_info in partitions:
        if wav_segment_info['myType'] == 'speech':
            wav_index += 1
            begin = wav_segment_info['start'] * configs.audioSegmentLength
            end = wav_segment_info['end'] * configs.audioSegmentLength
            results.append(data[begin:end])
            
            scipy.io.wavfile.write('part_' + str(wav_index) + '.wav', 16000, data[begin:end])
    return results

if __name__ == "__main__":
    partition('output.wav', 'part_')
