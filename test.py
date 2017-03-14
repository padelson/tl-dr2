import numpy as np
import tensorflow as tf

# W = tf.constant(np.random.rand([num_encoder_symbols,
#                                 embedding_size]))
# ids = [tf.placeholder(tf.int32, [sequence_length, ])]
#
# embeddings = tf.nn.embedding_lookup(W, ids)
# sess = tf.Session()
# embeddings.eval(session=sess)

x = [tf.placeholder(tf.float32, shape=[None]) for _ in range(4)]

# Use tf.shape() to get the runtime size of `x` in the 0th dimension.
zeros_dims = tf.pack([tf.shape(x)[0], 7])

y = tf.fill(zeros_dims, 0.0)
z = tf.pack(x)

sess = tf.Session()
result = sess.run([x, z], feed_dict={i: np.random.rand(5) for i in x})
print result, result[1].shape
