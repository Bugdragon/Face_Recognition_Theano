# Face_Recognition_Theano
使用Theano实现类似LeNet5的人脸识别器

### 人脸图像库Olivetti Faces
纽约大学的小型人脸库，由40个人的400张图片构成，即每个人的人脸图片为10张。每张图片的灰度级为8位，每个像素的灰度大小位于0-255之间，每张图片大小为64×64。
如下图，图片大小是1190*942，一共有20*20张人脸，每张人脸大小是57*47：
![Image text](https://raw.githubusercontent.com/Bugdragon/Face_Recognition_Theano/master/olivettifaces.gif)

### 代码实现
1. 实现图像加载函数，添加label，分割数据集☑
2. 参考LeNet5，定义ConvPoolLayer、HiddenLayer、LogisticRegression三个layer
3. 构建网络架构：input->layer0(ConvPoolLayer)->layer1(ConvPoolLayer)->layer2(HiddenLayer)->layer3(LogisticRegression)
4. 设置优化算法，应用于Olivetti Faces进行人脸识别
5. 训练结果以及参数设置的讨论

### 版本条件
* Ubuntu 18.04LTS(64-bit)
* Python 3.6.5(pip3)
* Numpy 1.15.4
* Theano 1.0.3
