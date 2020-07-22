import tensorflow.compat.v1 as tf
from tensorflow.python.framework import graph_io

sess = tf.Session()
tf.disable_eager_execution()
saver = tf.train.import_meta_graph("../mixnet-l/TF/model.ckpt.meta")
saver.restore(sess, tf.train.latest_checkpoint('../MixNet/mixnet-l/TF/'))
frozen = tf.graph_util.convert_variables_to_constants(sess, sess.graph_def,
                                                      ['logits'])
graph_io.write_graph(frozen, './', 'inference_graph.pb', as_text=False)