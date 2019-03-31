from __future__ import print_function
import os
from io import BytesIO
import numpy as np
from functools import partial
import PIL.Image
import scipy.misc
import tensorflow as tf


graph = tf.Graph()
sess = tf.InteractiveSession(graph=graph)
model_fn = 'tensorflow_inception_graph.pb'

with tf.gfile.GFile(model_fn, "rb") as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())

t_input = tf.placeholder(np.float32, name='input')
imagenet_mean = 117.0
t_preprocessed = tf.expand_dims(t_input - imagenet_mean, 0)
tf.import_graph_def(graph_def, {'input': t_preprocessed})


"""定义相关函数"""
# 定义savearray函数，用于把一个numpy.ndarray保存成为图像文件
def savearray(img_array, img_name):
    scipy.misc.toimage(img_array).save(img_name)
    print('img saved: %s' % img_name)

# 将图像放大ratio倍
def resize_ratio(img, ratio):
    min = img.min()
    max = img.max()
    img = (img - min) / (max - min) * 255
    img = np.float32(scipy.misc.imresize(img, ratio))
    img = img / 255 * (max - min) + min
    return img

# 调整图像尺寸
def resize(img, hw):
    min = img.min()
    max = img.max()
    img = (img - min) / (max - min) * 255
    img = np.float32(scipy.misc.imresize(img, hw))
    img = img / 255 * (max - min) + min
    return img

# 原始图像尺寸可能很大，从而可能导致内存耗尽的问题
# 原理就是把大图像分解成小图像，每次只对其中一张小图像做优化
# 每次只对tile_size * tile_size 大小的图像计算梯度，避免上面的内存问题
def calc_grad_tiled(img, t_grad, tile_size=512):
    # 参数：要分解的图像，梯度，要分解成的小图像的尺寸
    sz = tile_size
    h, w = img.shape[:2]
    sx, sy = np.random.randint(sz, size=2)
    img_shift = np.roll(np.roll(img, sx, 1), sy, 0) # 先在行上做整体移动，再在列上做整体移动
    grad = np.zeros_like(img)
    for y in range(0, max(h - sz // 2, sz), sz):
        for x in range(9, max(w - sz // 2, sz), sz):
            sub = img_shift[y: y + sz, x: x + sz]
            g = sess.run(t_grad, {t_input: sub})
            grad[y: y + sz, x: x + sz] = g
    return np.roll(np.roll(grad, -sx, 1), -sy, 0)


def render_deepdream(t_obj, img0, iter_n=20, step=1.5, octave_n=4, octave_scale=1.4):
    """参数说明：
    t_obj: 是layer_output[:,:,:,channel]，即卷积层的某个通道的值
    img0: 初始图像（噪声图像）
    iter_n: 迭代次数
    step: 用于控制每次迭代的步长，可以看作学习率
    octave_n: 拉普拉斯金字塔分解后的总层数
    octave_scale：层与层之间的倍数，乘以这个倍数图像放大，除以这个倍数图像缩小
    """
    t_score = tf.reduce_mean(t_obj)
    # t_score是t_obj的平均值
    # 由于我们的目标是调整输入图像，使卷积层激活值尽可能大，即最大化t_score
    # 为达到这个目标，可以使用梯度下降，计算t_score对t_input的梯度
    t_grad = tf.gradients(t_score, t_input)[0]  # t_input在主程序中定义
    # t_input = tf.placeholder(np.float32, name='input')，是输入图像的占位符

    img = img0.copy() # 复制图像，以免影响原图像的值

    # 将图像进行金字塔分解：分为高频、低频部分
    octaves = []
    for i in range(octave_n - 1):
        hw = img.shape[:2]
        lo = resize(img, np.int32(np.float32(hw) / octave_scale))  # 将原图resize,将低频成分缩小
        hi = img - resize(lo, hw) # 原图减去低频成分就得到高频成分
        # 但此处不能直接减，因为低频成分已经缩小了，需要resize成为和原图一样大小的，之后再相减
        img = lo
        octaves.append(hi)

    # 首先生成低频图像，再依次放大并加上高频
    for octave in range(octave_n):
        if octave > 0:
            hi = octaves[-octave]
            img = resize(img, hi.shape[:2]) + hi
        for i in range(iter_n):
            g = calc_grad_tiled(img, t_grad)  # calc_grad_tiled函数作用是避免内存耗尽的问题
            img += g * (step / (np.abs(g).mean() + 1e-7))

    img = img.clip(0,255)
    savearray(img, 'mountain_deepdream.jpg')
    im = PIL.Image.open('mountain_deepdream.jpg').show()


"""以背景图片为起点生成deepdream图像"""
# 定义卷积层，并取出相应的tensor
name = 'mixed4c'
layer_output = graph.get_tensor_by_name('import/%s:0' % name)

# 用一张背景图片作为起点，对图像进行优化
img0 = PIL.Image.open('bgimg-mountain.jpg')
img0 = np.float32(img0)

#调用render_naive函数进行渲染"
render_deepdream(tf.square(layer_output), img0)

# 保存并显示图片
# im = PIL.Image.open('naive_deepdream.jpg')
# im.show()
# im.save('mountain_naive_deepdream.jpg')

