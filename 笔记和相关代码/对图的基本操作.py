"""
图的基本操作：
1.建立图、获得默认图、重置默认图
tf.Graph(), tf.get_default_grapy(), tf.reset_default_grapy()
2. 获取张量操作
3. 获取节点操作
"""

import tensorflow as tf
import numpy as np

"""----------------------------1.建立新图、获得默认图、重置默认图-------------------------------------------"""
# g = tf.Graph()
# with g.as_default():
#     c1 = tf.constant(0.0) # 在新图中添加变量
#     print(c1)    # 输出  Tensor("Const:0", shape=(), dtype=float32)
#     print('c1.graph:', c1.graph) # 通过变量的".graph"获得其所在的图
#     # 输出：c1.graph: <tensorflow.python.framework.ops.Graph object at 0x0000022D1DCA0B00>
#
# # 重置默认图
tf.reset_default_graph()
g2 = tf.get_default_graph() # 获得当前的默认图
# print('g2: ', g2)  # 输出  g2: <tensorflow.python.framework.ops.Graph object at 0x0000022D30AD2668>

"""-----------------------------------2.获取张量-----------------------------------------------------"""
# 先获得张量的名字
# print(c1.name)   # 输出  Const:0
# # 用get_tensor_by_name()函数获取张量
# t = g.get_tensor_by_name(name='Const:0')
# # 通过打印t验证get_tensor_by_name所获得的张量就是前面定义的张量c1
# print(t)    # 输出  Tensor("Const:0", shape=(), dtype=float32)

"""
不必花太多精力去关注tensorflow中默认的命名规则。一般在使用名字时，都会在定义的同时为其指定好名称；
如果真的不清楚某个元素的名称，可以通过"张量名.name"获得，并回填到代码中即可
"""
"""
# ------------------------------------3.获取节点------------------------------------------------
1.获取节点操作：get_operation_by_name
2. 获取节点操作方法与获取张量的方法非常类似，先将op的名字打印出来，然后使用get_operation_by_name获取节点
"""
# 下面将获取张量和获取节点的例子放在一起
a = tf.constant([[1.0, 2.0]])
b = tf.constant([[3.0], [4.0]])
tensor1 = tf.matmul(a, b, name='example_op')
print(tensor1)    # 输出： Tensor("example_op:0", shape=(1, 1), dtype=float32)
print(tensor1.name)  # 输出：example_op:0
print(tensor1.op.name)   # 输出：example_op  操作（节点）是矩阵乘法，其名称也就是定义时指定的name='example_op'

# 然后使用get_operation_by_name函数
test_op = g2.get_operation_by_name('example_op')
print(test_op)