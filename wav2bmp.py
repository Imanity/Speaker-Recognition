import scipy.io.wavfile as wav
import numpy as np
import configs
import math
import matplotlib.pyplot as plt
from PIL import Image

def wav2bmp(wav_filename, output_path):
    (sample_rate, wav_data) = wav.read(wav_filename)

    datas = []
    data_len = len(wav_data)
    n = int(data_len / configs.single_img_rate)
    imgs = np.zeros((n, configs.img_h, configs.img_w))

    for i in range(0, n):
        datas.append(wav_data[i * configs.single_img_rate : (i + 1) * configs.single_img_rate])

    for (img_id, data) in enumerate(datas):
        n = int(configs.single_img_rate / configs.single_pixel_rate)
        for i in range(0, n):
            ftt_data = data[i * configs.single_pixel_rate : (i + 1) * configs.single_pixel_rate]
            ftt_res = np.fft.fft(ftt_data)
            for j in range(0, n):
                data_tmp = ftt_res[j]
                data_tmp_val = math.sqrt(data_tmp.real * data_tmp.real + data_tmp.imag * data_tmp.imag)
                imgs[img_id][j][i] = data_tmp_val * 255.0 / configs.max_volume
                if imgs[img_id][j][i] > 255.0:
                    imgs[img_id][j][i] = 255.0

    '''     
    for img in imgs:
        fig = plt.figure()
        ax = fig.add_subplot(221)
        ax.imshow(img)
        plt.show()
    '''

    for (imgId, img) in enumerate(imgs):
        pilImg = Image.fromarray(img.astype(np.uint8))
        pilImg.thumbnail((64, 64), Image.ANTIALIAS)
        pilImg.save(output_path + str(imgId) + '.bmp', "BMP")
    
    return imgs

def wavArray2bmpArray(wav_data):
    datas = []
    data_len = len(wav_data)
    n = int(data_len / configs.single_img_rate)
    imgs = np.zeros((n, configs.img_h, configs.img_w))

    for i in range(0, n):
        datas.append(wav_data[i * configs.single_img_rate : (i + 1) * configs.single_img_rate])
    
    for (img_id, data) in enumerate(datas):
        n = int(configs.single_img_rate / configs.single_pixel_rate)
        for i in range(0, n):
            ftt_data = data[i * configs.single_pixel_rate : (i + 1) * configs.single_pixel_rate]
            ftt_res = np.fft.fft(ftt_data)
            for j in range(0, n):
                data_tmp = ftt_res[j]
                data_tmp_val = math.sqrt(data_tmp.real * data_tmp.real + data_tmp.imag * data_tmp.imag)
                imgs[img_id][j][i] = data_tmp_val * 255.0 / configs.max_volume
                if imgs[img_id][j][i] > 255.0:
                    imgs[img_id][j][i] = 255.0
    
    return imgs

if __name__ == "__main__":
    wav2bmp('E:/tang3.wav', 'E:/wavImg/tang3/')
