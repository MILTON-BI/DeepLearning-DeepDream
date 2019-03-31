import tensorflow as tf

# 图的保存
v = tf.Variable(1.0, name='my_variable')
with tf.Session() as sess:
    tf.train.write_graph(sess.graph_def, './tfmodel', 'test_pb.pb', as_text=False)

# 图的加载
with tf.Session() as sess:
    with tf.gfile.FastGFile('./tfmodel/test_pb.pb', "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        sess.graph.as_default()
        tf.import_graph_def(graph_def, name="tf.graph")
        print(graph_def)