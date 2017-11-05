from python_speech_features import mfcc
import numpy as np
import wave
from pyaudio import PyAudio, paInt16
import configs
import scipy.io.wavfile as wav

def feature_extract(file):
    (rate,sig) = wav.read(file)
    mfcc_feat = mfcc(sig,rate)
    return mfcc_feat
    #f = open(out, "wb")
    #np.savetxt(f, mfcc_feat)
    #print(rate)
    #print(type(sig))
    #print(sig.shape)
    #print(type(mfcc_feat))
    

