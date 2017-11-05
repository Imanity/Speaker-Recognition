from python_speech_features import mfcc
import numpy as np
import scipy.io.wavfile as wav
#import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA

from scipy import spatial
from scipy.cluster.vq import kmeans


def getKMeansVec(array):
    (res, distortion)= kmeans(array, 1)
    #print(res.shape)
    #print(distortion)
    return res

def getCosineDistance(vec1, vec2):
    return 1 - spatial.distance.cosine(vec1, vec2)

def extract(file):
    (rate,sig) = wav.read(file)
    mfcc_feat = mfcc(sig, rate)
    return mfcc_feat

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
    #feature_extract(['wei_1.wav', 'wei_2.wav', 'tang_1.wav', 'tang_2.wav'])
    a1 = getKMeansVec(extract("wei_1.wav"))
    a2 = getKMeansVec(extract("wei_2.wav"))
    b1 = getKMeansVec(extract("tang_1.wav"))
    b2 = getKMeansVec(extract("tang_2.wav"))
    print(getCosineDistance(a1, a2))
    print(getCosineDistance(b1, b2))
    print(getCosineDistance(a1, b1))
    print(getCosineDistance(a1, b2))
    print(getCosineDistance(a2, b1))
    print(getCosineDistance(a2, b2))

