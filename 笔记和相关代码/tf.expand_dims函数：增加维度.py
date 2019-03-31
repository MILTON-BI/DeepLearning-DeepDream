# 图像的格式通常是三维(height, wight, channles)，这只能表示一张图像；而Inception模型要求输入格式是(batch,height,wight,channels),这样才能将多张图片送入网络
# 变为(batch, height, weight, channels)
# tf.expand_dims(input, dim, name=None)表示向tensor中插入一维，插入的位置就是dim参数代表的位置(维数从0开始)

import tensorflow as tf
import numpy as np

# 使用tf.expand_dims(input, dim, name=None)
t = [[2,3,4],[5,6,7]]
print('t_shape=', np.shape(t))

t1 = tf.expand_dims(t, 0)
print('t1_shape=', np.shape(t1))

t2= tf.expand_dims(t, 1)
print('t2_shape=', np.shape(t2))

t3 = tf.expand_dims(t, 2)
print('t3_shape=', np.shape(t3))

t4 = tf.expand_dims(t, -1)   # 参数-1表示最后一维
print('t4_shape=', np.shape(t4))
