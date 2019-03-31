
"""图像处理函数
   1. 图像编码/解码
   2. 图形缩放
   3. 图像的裁剪和填充
   4. 图像的水平和上下翻转
   5. 改变对比度
   6. 白化处理
"""

import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

# 读取图像的原始数据
image_raw_data = tf.gfile.FastGFile('dog.jpg', 'rb').read()

"""图像编码"""
# with tf.Session() as sess:
#     # 对图像进行jpeg的格式解码，从而得到图像对应的三维矩阵
#     img_data = tf.image.decode_jpeg(image_raw_data)
#     # img_data = tf.image.decode_png(image_raw_data)
#
#     # 解码后的结果是一个张量，需要通过eval()函数来获取它的值
#     print(img_data.eval())
#     print(img_data.eval().shape)  # 结果是shape为(1080, 1920, 3)的张量（对应原图片像素1920*1080）
#
#     # 可视化
#     plt.imshow(img_data.eval())
#     plt.show()


"""图像缩放：四种方法，根据函数参数method不同的设置"""
# 双线性插值法：ResizeMethod.BILINEAR(默认设置)，对应的method=0
# with tf.Session() as sess:
#     img_data = tf.image.decode_jpeg(image_raw_data)
#     resized1 = tf.image.resize_images(img_data, [256, 256], method=0)
#     # tensorflow的函数处理图片后存储的数据是float32格式的，需要转换成uint8才能正确显示图片
#     resized1 = np.asarray(resized1.eval(), dtype='uint8')
#     plt.imshow(resized1)
#     plt.show()

# 最邻近插值法：NEAREST_NEIGHBOR,对应的method=1
# with tf.Session() as sess:
#     img_data = tf.image.decode_jpeg(image_raw_data)
#     resized2 = tf.image.resize_images(img_data, [256, 256], method=1)
#     resized2 = np.asarray(resized2.eval(), dtype='uint8')
#     plt.imshow(resized2)
#     plt.show()
#
# # 双立方插值法：BICUBIC,对应的method=2
# with tf.Session() as sess:
#     img_data = tf.image.decode_jpeg(image_raw_data)
#     resized3 = tf.image.resize_images(img_data, [256, 256], method=2)
#     resized3 = np.asarray(resized3.eval(), dtype='uint8')
#     plt.imshow(resized3)
#     plt.show()
#
# # 像素区域插值法：AREA,对应的method=3
# with tf.Session() as sess:
#     img_data = tf.image.decode_jpeg(image_raw_data)
#     resized4 = tf.image.resize_images(img_data, [256, 256], method=3)
#     resized4 = np.asarray(resized4.eval(), dtype='uint8')
#     plt.imshow(resized4)
#     plt.show()


"""图像裁剪和填充，然后缩放
   1.常规的裁剪方式
   （1）函数：tf.image.resize_image_with_crop_or_pad(image, target_height, target_width)
   （2）如果目标图像尺寸小于原始图像尺寸，则在中心位置裁剪，反之则用黑色像素进行填充  
"""
# 裁剪
# with tf.Session() as sess:
#     img_data = tf.image.decode_jpeg(image_raw_data)
#     croped = tf.image.resize_image_with_crop_or_pad(img_data, 400, 400)
#     plt.imshow(croped.eval())
#     plt.show()
# # 填充
# with tf.Session() as sess:
#     img_data = tf.image.decode_jpeg(image_raw_data)
#     padded = tf.image.resize_image_with_crop_or_pad(img_data, 2000, 2000)
#     plt.imshow(padded.eval())
#     plt.show()

""" 2.随机裁剪：
   （1）函数：tf.image.random_crop(image, size, seed=None, name=None)
        size是图像裁剪后的尺寸[长,宽,通道数]
   """
# 随机裁剪
# with tf.Session() as sess:
#     img_data = tf.image.decode_jpeg(image_raw_data)
#     random_croped1 = tf.image.random_crop(img_data, [800, 800, 3])
#     plt.imshow(random_croped1.eval())
#     plt.show()
# # 再次随机裁剪，验证随机性
# with tf.Session() as sess:
#     img_data = tf.image.decode_jpeg(image_raw_data)
#     random_croped2 = tf.image.random_crop(img_data, [800, 800, 3])
#     plt.imshow(random_croped2.eval())
#     plt.show()

""" 水平翻转：tf.image.flip_left_right(image_data)"""
# with tf.Session() as sess:
#     img_data = tf.image.decode_jpeg(image_raw_data)
#     plt.imshow(img_data.eval())
#     plt.axis('off')
#     plt.show()
#     flip_left_right = tf.image.flip_left_right(img_data)
#     plt.imshow(flip_left_right.eval())
#     plt.axis('off')
#     plt.show()

""" 垂直（上下）翻转：tf.image.flip_up_down(image_data)"""
# with tf.Session() as sess:
#     img_data = tf.image.decode_jpeg(image_raw_data)
#     plt.imshow(img_data.eval())
#     plt.axis('off')
#     plt.show()
#     flip_up_down = tf.image.flip_up_down(img_data)
#     plt.imshow(flip_up_down.eval())
#     plt.axis('off')
#     plt.show()

""" 改变对比度：
    (1)tf.image.adjust_contrast: 按给定值调整图像的对比度
    (2)tf.image.random_contrast：在某一个范围内随机调整对比度"""
# with tf.Session() as sess:
#     img_data = tf.image.decode_jpeg(image_raw_data)
#     plt.imshow(img_data.eval())
#     plt.show()
    # 将图像的对比度降至原来的二分之一
    # contrast = tf.image.adjust_contrast(img_data, 0.5)
    # 将图像对比度提高至原来的5倍
    # contrast = tf.image.adjust_contrast(img_data, 5)
    # # 在[lower,upper]范围内随机调整图像对比度
    # contrast = tf.image.random_contrast(img_data, lower=0.2, upper=3)
    #
    # plt.imshow(contrast.eval())
    # plt.show()

""" 白化（标准化或归一化）处理：tf.image.per_image_standardization(image_data)
    (1)将图像的像素值转换为零均值和单位方差
"""
with tf.Session() as sess:
    img_data = tf.image.decode_jpeg(image_raw_data)
    plt.imshow(img_data.eval())
    plt.show()
    standardization = tf.image.per_image_standardization(img_data)
    plt.imshow(np.asarray(standardization.eval(), dtype='uint8'))
    plt.show()
