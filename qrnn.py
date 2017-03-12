import numpy as np
import tensorflow as tf


class QRNN(object):
    def __init__(self, num_encoder_symbols, num_decoder_symbols,
                 embedding_size, num_layers, conv_size, num_convs):
        # TODO assign args
        self.num_encoder_symbols = num_encoder_symbols
        self.num_decoder_symbols = num_decoder_symbols
        self.embedding_size = embedding_size
        self.encode_layers = num_layers
        self.decode_layers = num_layers
        self.conv_size = conv_size
        self.num_convs = num_convs
        self.initializer = None
        pass

    def seq2seq_f(self, encoder_inputs, decoder_inputs,
                  output_projection=None, training=False):
        # TODO embed inputs??
        # TODO what do i do about output_projection
        encode_outputs = []
        for i in range(self.encode_layers):
            inputs = encoder_inputs if i == 0 else encode_outputs[-1]
            encode_outputs.append(self.conv_layer(inputs))
        decode_outputs = []
        for i in range(self.decode_layers):
            enc_out = encode_outputs[i][-1]
            # TODO what do you feed in during decode lol
            if not training:
                decode_inputs = None
            decode_outputs.append(self.conv_with_encode_output(decoder_inputs,
                                                               enc_out))
        last_hidden = self.attention_output(encode_outputs[-1],
                                            decode_outputs[-1])
        return self.transform_output(last_hidden)

    def fo_pool(self, Z, F, O):
        H = [np.zeros(Z[0].shape)]
        C = [np.zeros(Z[0].shape)]
        for i in range(1, self.num_convs):
            C.append(tf.mul(F[i], C[i-1]) + tf.mul(1-F[i], Z[i]))
            H.append(tf.mul(O[i], C[i]))
        return H

    def f_pool(self, Z, F):
        H = [np.zeros(Z[0].shape)]
        for i in range(1,):
            H.append(tf.mul(F[i], H[i-1]) + tf.mul(1-F[i]))
        return H

    def conv_layer(self, layer_id):
        pass

    def linear_layer(self, layer_id, inputs):
        with tf.variable_scope('QRNN/Linear/'+str(layer_id)):
            # TODO shape
            W = tf.get_variable('W', [], initializer=self.initializer)
            b = tf.get_variable('b', [], initializer=self.initializer)

            _weighted = tf.add(tf.mul(inputs, W), b)
            Z, F, O = tf.split(1, 3, _weighted)
            return self.fo_pool(tf.tanh(Z), tf.sigmoid(F), tf.sigmoid(O))

    def conv_with_encode_output(self, layer_id):
        pass

    def linear_with_encode_output(self, layer_id, inputs=None, h_t):
        with tf.variable_scope('QRNN/Linear_with_encode/'+str(layer_id)):
            # TODO shape
            W = tf.get_variable('W', [], initializer=self.initializer)
            V = tf.get_variable('V', [], initializer=self.initializer)
            b = tf.get_variable('b', [], initializer=self.initializer)

            _sum = tf.mul(h_t, V)
            if inputs is not None:
                _sum = tf.add(_sum, tf.mul(inputs, W))
            _weighted = tf.add(_sum, b)
            Z, F, O = tf.split(1, 3, _weighted)
            return self.fo_pool(tf.tanh(Z), tf.sigmoid(F), tf.sigmoid(O))

    def attention_output(self):
        pass

    def transform_output(self):
        pass
