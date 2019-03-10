import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
import  tensorflow as tf
sess = tf.InteractiveSession()
import datetime
start = datetime.datetime.now()
import numpy as np


Xtr, Ytr = mnist.train.next_batch(5000)  #5000 条候选样本
Xte, Yte = mnist.test.next_batch(200)    #200 条测试样本
# tf Graph Input，占位符，用来feed数据
xtr = tf.placeholder("float", [None, 100])
xte = tf.placeholder("float", [100])
distance = tf.reduce_sum(tf.abs(tf.add(xtr, tf.negative(xte))), reduction_indices=1)
# 预测: 获取离测试样本具有最小L1距离的样本(1-NN），此样本的类别作为test样本的类别
pred = tf.argmin(distance, 0)
accuracy = 0.

# 初始化图
init = tf.global_variables_initializer()

# 发布图
with tf.Session() as sess:
    sess.run(init)

    #循环测试集
    for i in range(len(Xte)):
        # Get nearest neighbor
        nn_index = sess.run(pred, feed_dict={xtr: Xtr, xte: Xte[i, :]})  #每次循环feed数据，候选Xtr全部，测试集Xte一次循环输入一条
        # 获得与测试样本最近样本的类别，计算与真实类别的误差
        print("Test", i, "Prediction:", np.argmax(Ytr[nn_index]), \
              "True Class:", np.argmax(Yte[i]))
        # 计算误差率
        if np.argmax(Ytr[nn_index]) == np.argmax(Yte[i]):
            accuracy += 1. / len(Xte)
    print("Done!")
    print("Accuracy:", accuracy)



# def weight_variable(shape):
#     initial = tf.truncated_normal(shape, stddev=0.1)
#     return tf.Variable(initial)
#
#
# def bias_variable(shape):
#     initial = tf.constant(0.1, shape=shape)
#     return tf.Variable(initial)
#
#
# def conv2d(x, W):
#     return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
#
#
# def max_pool_2x2(x):
#     return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
#                           strides=[1, 2, 2, 1], padding='SAME')

# x = tf.placeholder('float', shape=[None, 100])
# y_ = tf.placeholder('float', shape=[None, 10])
# W_conv1 = weight_variable([5, 5, 1, 32])
# b_conv1 = bias_variable([32])
#x_image = tf.reshape(x, [-1, 10, 10, 1])  # 28x28,通道数为1


# h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
# h_pool1 = max_pool_2x2(h_conv1)
#
# W_conv2 = weight_variable([5, 5, 32, 64])
# b_conv2 = bias_variable([64])
#
# h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
# h_pool2 = max_pool_2x2(h_conv2)
#
# W_fc1 = weight_variable([3*3*64, 1024])
# b_fc1 = bias_variable([1024])
#
# h_pool2_flat = tf.reshape(h_pool2, [-1, 3*3*64])
# h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
#
# keep_prob = tf.placeholder('float')
# h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
#
# W_fc2 = weight_variable([1024, 10])
# b_fc2 = bias_variable([10])
#
# y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
#
# cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv))
# train_step = tf.train.AdamOptimizer(0.0001).minimize(cross_entropy)
# correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
# accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))
# sess.run(tf.global_variables_initializer())

# for i in range(1000):
#     batch = mnist.train.next_batch(50)
#     if i % 100 == 0:
#         train_accuracy = accuracy.eval(feed_dict={
#             x:batch[0], y_:batch[1], keep_prob: 1.0})
#         print('step {}, training accuracy {}'.format(i, train_accuracy))
#     train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
#
# print('test accuracy {}'.format(accuracy.eval(feed_dict={
#     x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0})))
#
# end = datetime.datetime.now()
# print((end-start).seconds)
