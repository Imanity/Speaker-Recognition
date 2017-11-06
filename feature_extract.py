from python_speech_features import mfcc
import numpy as np
import scipy.io.wavfile as wav
#import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA

from scipy import spatial, stats
from scipy.cluster.vq import kmeans, kmeans2


def getKMeansVec(array, k = 1):
    (res, distortion)= kmeans(array, k)
    #print(res.shape)
    #print(distortion)
    return res

def getKMeansVec2(array, k = 1 ,minit ="points"):
    (centroid, label)= kmeans2(array, k)
    (res, count) = stats.mode(label)

    minDistance = -1
    minIndex = -1
    groups = []
    for i in range(k):
        groups = []
        for j in range(array.shape[0]):
            if label[j] == i:
                groups.append(array[j])
        if len(groups) == 1 or len(groups) == 0:
            continue
        tmp = getAverageDistance(centroid[i], groups)
        if minDistance == -1:
            minDistance = tmp
            minIndex = i
        elif minDistance > tmp:
            minDistance = tmp
            minIndex = i

    print(minIndex)
    print(label)
    return centroid

def getRecommandVec(array):
    k = 3
    (centroid, label)= kmeans2(array, k)
    (res, count) = stats.mode(label)

    minDistance = -1
    minIndex = -1
    groups = []
    for i in range(k):
        groups = []
        for j in range(array.shape[0]):
            if label[j] == i:
                groups.append(array[j])
        if len(groups) == 1 or len(groups) == 0:
            continue
        tmp = getAverageDistance(centroid[i], groups)
        if minDistance == -1:
            minDistance = tmp
            minIndex = i
        elif minDistance > tmp:
            minDistance = tmp
            minIndex = i
    if minIndex == -1:
        return centroid[res[0]]
    return centroid[minIndex]

def getLabels(ararys):
    k = 4
    (centroid, label)= kmeans2(array, k)
    #(res, count) = stats.mode(label)
    #indexs = np.zeros(array.shape[0])
    dicts = {}
    index = 0
    for i in label:
        if dicts.get(str(i)) == None:
            index = index+1
            dicts[str(i)] = index
    for i in range(len(label)):
        label[i] = dicts.get(str(label[i]))
    return label

def getAverageDistance(target, array):
    dis = 0
    for i in range(len(array)):
        dis = dis + np.sqrt(np.sum(np.square(target - array[i])))
    return dis/len(array)

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

