import scipy.io.wavfile as wav
import numpy as np
import configs
import math
import matplotlib.pyplot as plt

def wav2bmp(wav_filename):
    (sample_rate, wav_data) = wav.read(wav_filename)

    datas = []
    data_len = len(wav_data)
    n = int(data_len / configs.single_img_rate)
    imgs = np.zeros((n, configs.img_h, configs.img_w))
    img_max = 0

    for i in range(0, n):
        datas.append(wav_data[i * configs.single_img_rate : (i + 1) * configs.single_img_rate])

    for (img_id, data) in enumerate(datas):
        n = int(configs.single_img_rate / configs.single_pixel_rate)
        for i in range(0, n):
            ftt_data = data[i * configs.single_pixel_rate : (i + 1) * configs.single_pixel_rate]
            ftt_res = np.fft.fft(ftt_data)
            for j in range(0, configs.single_pixel_rate):
                data_tmp = ftt_res[j]
                data_tmp_val = math.sqrt(data_tmp.real * data_tmp.real + data_tmp.imag * data_tmp.imag)
                if data_tmp_val > img_max:
                    img_max = data_tmp_val
                imgs[img_id][j][i] = data_tmp_val
    
    imgs = imgs * 255 / img_max

    '''
    for img in imgs:
        fig = plt.figure()
        ax = fig.add_subplot(221)
        ax.imshow(img)
        plt.show()
    '''

    return imgs

if __name__ == "__main__":
    print(wav2bmp('output.wav'))
