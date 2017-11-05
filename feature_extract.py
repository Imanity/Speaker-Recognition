from python_speech_features import mfcc
import numpy as np
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA

def feature_extract(files):
    mfccs = []
    for file in files:
        (rate,sig) = wav.read(file)
        mfcc_feat = mfcc(sig, rate)
        mfccs.append(mfcc_feat)
    # draw3d(mfccs)
    return mfccs

def pca_extract(mfcc_feat):
    pca = PCA(n_components = 3)
    pca.fit(mfcc_feat)
    '''
    print(pca.explained_variance_ratio_)
    print(pca.explained_variance_)
    
    print(mfcc_feat.shape)
    print(pca.transform(mfcc_feat).shape)
    '''
    return pca.transform(mfcc_feat)

def draw3d(mfccs):
    fig = plt.figure()
    ax = Axes3D(fig)
    colors = ['r', 'r', 'g', 'g', '#ffff00', '#ff00ff', '#00ffff']
    for i, mfcc in enumerate(mfccs):
        ax.scatter(mfcc[:, 0], mfcc[:, 1], mfcc[:, 2], c = colors[i])
    plt.show()

if __name__ == "__main__":
    feature_extract(['wei_1.wav', 'wei_2.wav', 'tang_1.wav', 'tang_2.wav'])
