import tensorflow as tf
import numpy as np
from PIL import Image
from feature_extract import getKMeansVec, getCosineDistance, getRecommandVec
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

from wav2bmp import wavArray2bmpArray

sess = tf.Session()
saver = tf.train.import_meta_graph('E:/model/model.meta')
saver.restore(sess,tf.train.latest_checkpoint('E:/model/'))

graph = tf.get_default_graph()
xs = graph.get_tensor_by_name("xs_to_restore:0")
keep_prob = graph.get_tensor_by_name("prob_to_restore:0")
prediction = graph.get_tensor_by_name("prediction_to_restore:0")
h_fc1 = graph.get_tensor_by_name("h_fc1_to_restore:0")

def getVec(path, begin, end):
    bmps = []
    for i in range(begin, end):
        img = Image.open(path + str(i) + '.bmp')
        bmp = np.array(img)
        bmps.append(bmp)
    vecs = sess.run(prediction, feed_dict = {xs: bmps, keep_prob: 1})
    print(getKMeansVec(vecs))
    return vecs

def draw(ps):
    colors = ['r', 'g', 'b', '#000000', '#ffff00', '#ff00ff', '#00ffff']
    fig = plt.figure()
    ax = Axes3D(fig)
    for i, p in enumerate(ps):
        ax.scatter(p[:,0], p[:,1], p[:,2], c = colors[i])
    plt.show()
vecs
def getVecFromArray(bmps):
    vecs = sess.run(prediction, feed_dict = {xs: bmps, keep_prob: 1})
    return 

def predict(audio_parts):
    core_vecs = []
    for i in audio_parts:
        bmps = wavArray2bmpArray(audio_parts)
        vecs = getVecFromArray(bmps)
        vec = getRecommandVec(vecs)
        core_vecs.append(vec)
    return core_vecs

if __name__ == "__main__":
    v1 = getVec('E:/wavImg/tangTest/', 0, 28)
    v2 = getVec('E:/wavImg/weiTest/', 0, 28)
    v3 = getVec('E:/wavImg/luoTest/', 0, 28)
    v4 = getVec('E:/wavImg/tangPredict/', 0, 8)
    v5 = getVec('E:/wavImg/weiPredict/', 0, 8)
    v6 = getVec('E:/wavImg/luoPredict/', 0, 8)
    v = np.array([v1])
    draw(v)
    v = np.array([v2])
    draw(v)
    v = np.array([v3])
    draw(v)
    v = np.array([v4])
    draw(v)
    v = np.array([v5])
    draw(v)
    v = np.array([v6])
    draw(v)
    print('--------')
