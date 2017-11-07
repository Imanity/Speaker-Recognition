import tensorflow as tf
import numpy as np
import random
from PIL import Image

# Read bmp

def getBMP(path, num):
    bmps = []
    labels = []
    for i in range(0, num):
        foldertType = random.randint(1, 3)
        if foldertType == 3:
            folderId = random.randint(1, 3)
            imgId = random.randint(0, 33)
            names = ['tang', 'wei', 'luo']
            img = Image.open(path + names[folderId - 1] + '/' + str(imgId) + '.bmp')
            bmp = np.array(img)
            bmps.append(bmp)
            if folderId == 1:
                labels.append([1.0, 0.0, 0.0])
            elif folderId == 2:
                labels.append([0.0, 1.0, 0.0])
            else:
                labels.append([0.0, 0.0, 1.0])
        else:
            folderId = random.randint(1, 3)
            imgId = random.randint(0, 291)
            names = ['tang', 'wei', 'luo']
            img = Image.open(path + names[folderId - 1] + str(foldertType) + '/' + str(imgId) + '.bmp')
            bmp = np.array(img)
            bmps.append(bmp)
            if folderId == 1:
                labels.append([1.0, 0.0, 0.0])
            elif folderId == 2:
                labels.append([0.0, 1.0, 0.0])
            else:
                labels.append([0.0, 0.0, 1.0])
    return np.array(bmps), np.array(labels)

def getTestBMP(path):
    bmps = []
    labels = []
    for i in range(0, 28):
        img = Image.open(path + 'tangTest/' + str(i) + '.bmp')
        bmp = np.array(img)
        bmps.append(bmp)
        labels.append([1.0, 0.0, 0.0])
    for i in range(0, 28):
        img = Image.open(path + 'weiTest/' + str(i) + '.bmp')
        bmp = np.array(img)
        bmps.append(bmp)
        labels.append([0.0, 1.0, 0.0])
    for i in range(0, 28):
        img = Image.open(path + 'luoTest/' + str(i) + '.bmp')
        bmp = np.array(img)
        bmps.append(bmp)
        labels.append([0.0, 0.0, 1.0])
    for i in range(0, 24):
        img = Image.open(path + 'tangtest1/' + str(i) + '.bmp')
        bmp = np.array(img)
        bmps.append(bmp)
        labels.append([1.0, 0.0, 0.0])
    for i in range(0, 14):
        img = Image.open(path + 'weitest1/' + str(i) + '.bmp')
        bmp = np.array(img)
        bmps.append(bmp)
        labels.append([0.0, 1.0, 0.0])
    for i in range(0, 21):
        img = Image.open(path + 'luotest1/' + str(i) + '.bmp')
        bmp = np.array(img)
        bmps.append(bmp)
        labels.append([0.0, 0.0, 1.0])
    return np.array(bmps), np.array(labels)

# Start training

def compute_accuracy(v_xs, v_ys):
    global prediction
    y_pre = sess.run(prediction, feed_dict={xs: v_xs, keep_prob: 1})
    correct_prediction = tf.equal(tf.argmax(y_pre,1), tf.argmax(v_ys,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    result = sess.run(accuracy, feed_dict={xs: v_xs, ys: v_ys, keep_prob: 1})
    return result

def weight_variable(shape, w):
    initial = tf.truncated_normal(shape, stddev = w)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape = shape)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

def avg_pool_2x2(x):
    return tf.nn.avg_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

xs = tf.placeholder(tf.float32, [None, 64, 64], name = 'xs_to_restore')
ys = tf.placeholder(tf.float32, [None, 3])
keep_prob = tf.placeholder(tf.float32, name = 'prob_to_restore')
x_image = tf.reshape(xs, [-1, 64, 64, 1])

## conv1 layer ##
W_conv1 = weight_variable([5, 5, 1, 32], 0.0001) # patch 5x5, in size 1, out size 32
b_conv1 = bias_variable([32])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1) # output size 64x64x32
h_pool1 = max_pool_2x2(h_conv1) # output size 32x32x32

## conv2 layer ##
W_conv2 = weight_variable([5, 5, 32, 32], 0.01) # patch 5x5, in size 32, out size 32
b_conv2 = bias_variable([32])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2) # output size 32x32x32
h_pool2 = avg_pool_2x2(h_conv2) # output size 16x16x32

## conv3 layer ##
W_conv3 = weight_variable([5, 5, 32, 64], 0.01) # patch 5x5, in size 32, out size 64
b_conv3 = bias_variable([64])
h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3, name = 'h_conv3_to_restore') # output size 16x16x64
h_pool3 = avg_pool_2x2(h_conv3) # output size 8x8x64

## func1 layer ##
W_fc1 = weight_variable([8 * 8 * 64, 4096], 0.1)
b_fc1 = bias_variable([4096])
# [n_samples, 8, 8, 64] ->> [n_samples, 8*8*64]
h_pool3_flat = tf.reshape(h_pool3, [-1, 8 * 8 * 64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool3_flat, W_fc1) + b_fc1, name = 'h_fc1_to_restore')
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

## func2 layer ##
W_fc2 = weight_variable([4096, 3], 0.1)
b_fc2 = bias_variable([3])
prediction = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2, name = 'prediction_to_restore')

## arguments ##
cross_entropy = -tf.reduce_sum(ys * tf.log(prediction))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

sess = tf.Session()

sess.run(tf.global_variables_initializer())

for i in range(2000):
    batch_xs, batch_ys = getBMP('E:/wavImg/', 100)
    sess.run(train_step, feed_dict={xs: batch_xs, ys: batch_ys, keep_prob: 0.5})
    if i % 100 == 0:
        test_xs, test_ys = getTestBMP('E:/wavImg/')
        print('step ' + str(i))
        print(compute_accuracy(test_xs, test_ys))

saver = tf.train.Saver()
saver.save(sess, 'E:/model/model')
