import os
import sys
import time
import cPickle
import numpy as np
from PIL import Image

import theano
import theano.tensor as T
from theano.tensor.signal import pool
from theano.tensor.nnet import conv

# 将数据集定义成shared类型，才能将数据复制进GPU，利用GPU加速
def shared_dataset(X, y, borrow=True):
    shared_X = theano.shared(np.asarray(X, dtype=theano.config.floatx), borrow=borrow)
    shared_y = theano.shared(np.asarray(y, dtype=theano.config.floatx), borrow=borrow

    return shared_X, T.cast(shared_y, 'int32')

# 图像加载函数
def load_data(dataset_path):
    img = Image.open(dataset_path)
    img_ndarray = np.asarray(img, dtype='float64')/256
    faces = np.empty((400,2679)) # 400张人脸图像，每张共2679个像素
    for row in range(20):
        for column in range(20):
            faces[row*20+column] = np.ndarray.flatten(img_ndarray[row*57:(row+1)*57, column*47:(column+1)*47]) # 人脸图像47*57
    # 添加label
    label = np.empty(400) # 0-399的label
    for i in range(40):
        label[i*10:i*10+10] = i
    label = label.astype(np.int)
    # 分割数据集
    train_X = np.empty((320, 2679))
    train_y = np.empty(320)
    val_X = np.empty((40, 2679))
    val_y = np.empty(40)
    test_X = np.empty((40, 2679))
    test_y = np.empty(40)
    for i in range(40):
        # 训练集取每个人的前8张人脸
        train_X[i*8:i*8+8] = faces[i*10:i*10+8]
        train_y[i*8:i*8+8] = label[i*10:i*10+8]
        # 验证集取第9个
        val_X[i] = faces[i*10+8]
        val_y[i] = faces[i*10+8]
        # 测试集取最后一个
        test_X = faces[i*10+9]
        test_y = faces[i*10+9]
    train_set_X, train_set_y = shared_dataset(train_X, train_y)
    val_set_X, val_set_y = shared_dataset(val_X, val_y)
    test_set_X, test_set_y = shared_dataset(test_X, test_y)
    rval = [(train_set_X, train_set_y), (val_set_X, val_set_y), (test_set_X, test_set_y)]

    return rval

# 卷积池化层，conv+maxpooling
class ConvPoolLayer(object):
    def __init__(self, rng, input, filter_shape, image_shape, poolsize=(2,2)):
        assert image_shape[1] == filter_shape[1]
        self.input = input
        fan_in = np.prod(filter_shape[1:])
        fan_out = (filter_shape[0] * np.prod(filter_shape[2:]) / np.prod(poolsize))
        # 随机初始化权重
        W_bound = np.sqrt(6. / (fan_in + fan_out))
        self.W = theano.shared(
            np.asarray(
                rng.uniform(low=-W_bound, high=W_bound, size=filter_shape),
                dtype=theano.config.floatx
            ),
            borrow=True
        )
        # bias为1D tensor
        b_values = np.zeros((filter_shape[0],), dtype=theano.config.floatx)
        self.b = theano.shared(value=b_values, borrow=True)
        # 卷积
        conv_out = conv.conv2d(
            input=input,
            filters=self.W,
            filter_shape=filter_shape,
            image_shape=image_shape
        )
        # 最大池化
        pooled_out = pool.max_pool_2d(
            input=conv_out,
            ds=poolsize,
            ignore_border=True
        )
        # 激活函数
        self.output = T.tanh(pooled_out + self.b.dimshuffle('x', 0, 'x', 'x'))
        # 保存参数
        self.params = [self.W, self.b]

# 全连接层
class HiddenLayer(object):
    def __init__(self, rng, input, n_in, n_out, W=None, b=None, activation=T.tanh):
        self.input = input
        if W is None:
            W_values = np.asarray(
                rng.uniform(
                    low=-np.sqrt(6. / (n_in + n_out)),
                    high=np.sqrt(6. / (n_in + n_out)),
                    size=(n_in, n_out)
                ),
                dtype=theano.config.floatx
            )
            if activation == theano.tensor.nnet.sigmoid:
                W_values *= 4
            W = theano.shared(value=W_values, name='W', borrow=True)
        if b is None:
            b_values = np.zeros((n_out,), dtype=theano.config.floatx)
            b = theano.shared(value=b_values, name='b', borrow=True)
        self.W = W
        self.b = b
        lin_output = T.dot(input, self.W) + self.b
        self.output = (
            lin_output if activation is None
            else activation(lin_output)
        )
        self.params = [self.W, self.b]

# 分类器，softmax
class LogisticRegression(object):
    def __init__(self, input, n_in, n_out):
        # 初始化参数W
        self.W = theano.shared(
            value=np.zeros((n_in, n_out), dtype=theano.config.floatx),
            name='W',
            borrow=True
        )
        # 初始化参数b
        self.b = theano.shared(
            value=np.zeros((n_out, ), dtype=theano.config.flaotx),
            name='b',
            borrow=True
        )
        # 计算属于不同类别的概率
        self.p_y_given_x = T.nnet.softmax(T.dot(input, self.W) + self.b)
        # 计算所属类别
        self.y_pred = T.argmax(self.p_y_given_x, axis=1)
        self.params = [self.W, self.b]
    
    # 负log似然函数
    def negative_log_likelihood(self, y):
        return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])

    # 计算minibatch中的错误率
    def errors(self, y):
        if y.ndim != self.y_pred.ndim:
            raise TypeError(
                'y should have the same shape as self.y_pred',
                ('y', y.type, 'y_pred', self.y_pred.type)
            )
        # 检查y的数据格式
        if y.dtype.startswith('int'):
            return T.mean(T.neq(self.y_pred, y))
        else:
            raise NotImplementedError()

# 保存训练参数
def save_params(param1, param2, param3, param4):
    write_file = open('params.pkl', 'wb')
    cPickle.dump(param1, write_file, -1)
    cPickle.dump(param2, write_file, -1)
    cPickle.dump(param3, write_file, -1)
    cPickle.dump(param4, write_file, -1)
    write_file.close()

'''
优化算法是minibatch SGD，比如image_shape=(batch_size, 1, 57, 47)
可以设置的参数有：
batch_size,但应注意n_train_batches、n_valid_batches、n_test_batches的计算都依赖于batch_size
nkerns=[5, 10]即第一二层的卷积核个数可以设置
全连接层HiddenLayer的输出神经元个数n_out可以设置，要同时更改分类器的输入n_in
学习速率learning_rate
'''
def evaluate(lr=0.05, n_epochs=200, dataset='olivettifaces.gif', nkerns=[5,10], batch_size=40):
    # 随机数生成器，用于初始化参数
    rng = np.RandomState(23455)
    # 加载数据
    datasets = load_data(dataset)
    train_set_X, train_set_y = datasets[0]
    val_set_X, val_set_y = datasets[1]
    test_set_X, val_set_y = datsets[2]
    # 计算batch个数
