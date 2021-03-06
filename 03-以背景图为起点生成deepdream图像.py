# 导入相关的库
from __future__ import print_function
import os
from io import BytesIO
import numpy as np
from functools import partial
import PIL.Image
import scipy.misc
import tensorflow as tf

# 创建图和会话
graph = tf.Graph()
sess = tf.InteractiveSession(graph=graph)

# 导入模型：tensorflow提供了一种以.pb为扩展名的文件，可以事先将模型导入到pb文件中，再在需要的时候导出
model_fn = 'tensorflow_inception_graph.pb' # 导入inception网络
# tensorflow_inception_graph.pb文件的下载：
# https://storage.googleapis.com/download.tensorflow.org/models/inception5h.zip

with tf.gfile.GFile(model_fn, "rb") as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())

# 定义输入图像的占位符
t_input = tf.placeholder(np.float32, name='input')

# 图像预处理——减均值(像素的均值)
imagenet_mean = 117.0 #在训练Inception模型时做了减均值预处理，此处也需要减同样的均值以保持一致

# 图像预处理——增加维度
# 图像的格式通常是三维(height, wight, channles)，这只能表示一张图像；而Inception模型要求输入格式是(batch,height,wight,channels),这样才能将多张图片送入网络
# 变为(batch, height, weight, channels)
# tf.expand_dims(input, dim, name=None)表示向tensor中插入一维，插入的位置就是dim参数代表的位置(维数从0开始)
# 函数返回的是一个张量，与input是一样的，只不过多了一个维度
t_preprocessed = tf.expand_dims(t_input - imagenet_mean, 0)  #这里同时完成了减均值的操作

# 导入模型并将经过预处理的图像送入网络中
tf.import_graph_def(graph_def, {'input': t_preprocessed})

# 找出所有卷积层
layers = [op.name for op in graph.get_operations() if op.type=='Conv2D']
# 输出卷积层层数
print('Number of layers: ', len(layers))     # 输出结果 59
# 输出所有卷积层名称
print(layers)
# 还可以输出指定卷积层的参数
# name1 = 'mixed4d_3x3_bottleneck_pre_relu'
# name2 = 'mixed4d_5x5_bottleneck_pre_relu'
# print('shape of %s : %s' % (name1, str(graph.get_tensor_by_name('import/' + name1 + ':0').get_shape())))
# print('shape of %s : %s' % (name2, str(graph.get_tensor_by_name('import/' + name2 + ':0').get_shape())))
# 输出结果：(?, ?, ?, 144)(?, ?, ?, 32)。卷积层shape格式一般是(batch,height,weight,channels)，因此时还不知道输入图像的数量和尺寸，所以前三个维度不确定，显示?。由于导入的是事先训练好的模型，所以卷积层的通道数量是固定的

"""以背景图片为起点生成deepdream图像"""
# 定义卷积层，并取出相应的tensor
name = 'mixed4c'
layer_output = graph.get_tensor_by_name('import/%s:0' % name)

# 用一张背景图片（不是随机噪声图像）作为起点，对图像进行优化
img_test = PIL.Image.open('bgimg-mountain.jpg')
# 以噪声图像为起点的代码：img_noise = np.random.uniform(size=(224,224,3)) + 100.0

"""定义render_naive渲染函数"""
# 定义savearray函数，用于把一个numpy.ndarray保存成为图像文件
def savearray(img_array, img_name):
    scipy.misc.toimage(img_array).save(img_name)
    print('img saved: %s' % img_name)

# 定义render_naive渲染函数
def render_naive(t_obj, img0, iter_n=20, step=1.0):
    """参数说明：
    t_obj: 是layer_output[:,:,:,channel]，即卷积层的某个通道的值
    img0: 初始图像（噪声图像）
    iter_n: 迭代次数
    step: 用于控制每次迭代的步长，可以看作学习率
    """
    t_score = tf.reduce_mean(t_obj)
    # t_score是t_obj的平均值
    # 由于我们的目标是调整输入图像，使卷积层激活值尽可能大，即最大化t_score
    # 为达到这个目标，可以使用梯度下降，计算t_score对t_input的梯度
    t_grad = tf.gradients(t_score, t_input)[0]  # t_input在主程序中定义
    # t_input = tf.placeholder(np.float32, name='input')，是输入图像的占位符

    img = img0.copy() # 复制图像，以免影响原图像的值
    for i in range(iter_n):
        # 在sess计算梯度，以及当前的t_score
        g, score = sess.run([t_grad, t_score], {t_input:img})
        # 对img应用梯度
        # 首先对梯度进行归一化处理
        g /= g.std() + 1e-8
        # 将正规化处理后的梯度应用在图像上，step用于控制每次迭代的步长，此处为1.0
        img += g * step
        print('iter: %d' % (i+1), 'score(mean)=%f' % score)



"""调用render_naive函数进行渲染"""
render_naive(layer_output, img_test, iter_n=100)
# 不用指定特定通道，即表示利用所有通道
# 单通道的时候，参数layer_output[:,:,:,channel]，指定某一个特定通道

# 保存并显示图片
im = PIL.Image.open('naive_deepdream.jpg')
im.show()
im.save('mountain_naive.jpg')

