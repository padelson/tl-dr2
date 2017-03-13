import numpy as np
import tensorflow as tf


class QRNN(object):
    def __init__(self, num_encoder_symbols, num_decoder_symbols, batch_size,
                 embedding_size, enc_input_size, dec_input_size,
                 num_layers, conv_size, num_convs):
        self.num_encoder_symbols = num_encoder_symbols
        self.num_decoder_symbols = num_decoder_symbols
        self.batch_size = batch_size
        self.embedding_size = embedding_size
        self.enc_input_size = enc_input_size
        self.dec_input_size = dec_input_size
        self.encode_layers = num_layers
        self.decode_layers = num_layers
        self.conv_size = conv_size
        self.num_convs = num_convs
        self.initializer = None

    def get_embeddings(word_ids):
        with tf.variable_scope('QRNN/embeddings'):
            W = tf.get_variable('W', [self.num_encoder_symbols,
                                      self.embedding_size],
                                initializer=self.initializer)
            return tf.nn.embedding_lookup(W, word_ids)

    def seq2seq_f(self, encoder_inputs, decoder_inputs,
                  output_projection=None, training=False):
        # TODO what do i do about output_projection
        encode_outputs = []
        embedded_inputs = get_embeddings(encoder_inputs)
        for i in range(self.encode_layers):
            inputs = embedded_inputs if i == 0 else encode_outputs[-1]
            encode_outputs.append(self.conv_layer(i, inputs))

        decode_outputs = []
        for i in range(self.decode_layers):
            enc_out = encode_outputs[i][-1]
            # TODO what do you feed in during decode lol
            if not training:
                decode_inputs = None
            is_last_layer = i == self.decode_layers - 1
            if not last_layer:
                decode_outputs.append(self.conv_with_encode_output(
                                      i,
                                      decoder_inputs,
                                      enc_out))
            else:
                last_state = self.conv_with_attention(encode_outputs,
                                                      decode_outputs)
        return self.transform_output(last_state)

    def fo_pool(self, Z, F, O):
        # Z, F, O dims: [batch_size, sequence_length, num_convs]
        H = np.zeros(Z.shape)
        C = np.zeros(Z.shape)
        for i in range(1, self.enc_input_size):
            C[:, i, :] = tf.mul(F[:, i, :], C[:, i-1, :]) + \
                         tf.mul(1-F[:, i, :], Z[:, i, :])
            H[:, i, :] = tf.mul(O[:, i, :], C[:, i, :])
        # i think we want output [batch, seq_len, num_convs, 1]
        return tf.expand_dims(np.array(H), -1)

    def f_pool(self, Z, F):
        # Z, F dims: [batch_size, sequence_length, num_convs]
        H = H = np.zeros(Z.shape)
        for i in range(1, self.enc_input_size):
            H[:, i, :] = tf.mul(F[:, i, :], H[:, i-1, :]) + \
                         tf.mul(1-F[:, i, :])
        return tf.expand_dims(np.array(H), -1)

    def _get_filter_shape(vec_size):
        return [self.conv_size, vec_size, 1, self.num_convs*3]

    # convolution dimension results maths
    # out_height = ceil(float(in_height - filter_height + 1) /
    #                   float(strides[1])) = sequence_length
    # out_width  = ceil(float(in_width - filter_width + 1) /
    #                   float(strides[2])) = 1
    # in_height = sequence_length + filter_height - 1
    # filter_height = conv_size
    # in_width = embedding_size
    # filter_width = embedding_size

    def conv_layer(self, layer_id, inputs):
        with tf.variable_scope("QRNN/Variable/Convolution/"+str(layer_id)):
            filter_shape = _get_filter_shape(inputs.shape[2])
            W = tf.get_variable('W', self.filter_shape,
                                initializer=self.initializer)
            b = tf.get_variable('b', [num_filters],
                                initializer=self.initializer)
            num_pads = self.conv_size - 1
            # input dims ~should~ now be [batch_size, sequence_length,
            #                             embedding_size, 1]
            padded_input = tf.pad(tf.expand_dims(inputs, -1),
                                  [[0, 0], [num_pads, 0],
                                   [0, 0], [0, 0]],
                                  "CONSTANT")
            conv = tf.nn.conv2d(
                padded_input,
                W,
                strides=[1, 1, 1, 1],
                padding="VALID",
                name="conv") + b
            # conv dims: [batch_size, sequence_length,
            #             1, num_convs*3]
            # squeeze out 3rd D
            # split 4th (now 3rd) dim into 3
            Z, F, O = tf.split(2, 3, tf.squeeze(conv))
            return self.fo_pool(tf.tanh(Z), tf.sigmoid(F), tf.sigmoid(O))

    def linear_layer(self, layer_id, inputs):
        with tf.variable_scope('QRNN/Linear/'+str(layer_id)):
            # TODO shape
            W = tf.get_variable('W', [], initializer=self.initializer)
            b = tf.get_variable('b', [], initializer=self.initializer)

            _weighted = tf.add(tf.mul(inputs, W), b)
            Z, F, O = tf.split(1, 3, _weighted)
            return self.fo_pool(tf.tanh(Z), tf.sigmoid(F), tf.sigmoid(O))

    def conv_with_encode_output(self, layer_id):
        raise NotImplementedError

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

    def conv_with_attention(self, encode_outputs, decode_outputs):
        with tf.variable_scope('QRNN/Conv_with_attention/'):
            # TODO shape
            W_conv = tf.get_variable('W', [], initializer=self.initializer)
            # TODO get conv variables
            W_k = tf.get_variable('W_k', [], initializer=self.initializer)
            W_c = tf.get_variable('W_c', [], initializer=self.initializer)
            b_o = tf.get_variable('b_o', [], initializer=self.initializer)

            # do normal conv with encode_output
            Z, F, O = None
            Z = tf.tanh(Z)
            F = tf.sigmoid(F)
            O = tf.sigmoid(O)

            C = []
            H = []
            enc_final_state = encode_outputs[-1]
            for i in range(self.num_convs):
                if i == 0:
                    C.append(np.zeros(Z[0].shape))
                else:
                    C.append(tf.mul(F[i], C[i-1]) + tf.mul(1-F[i], Z[i]))
                # TODO transpose one?
                # TODO make this more efficient
                alpha = tf.nn.softmax(tf.matmul(C[i], enc_final_state,
                                                transpose_b=True))
                k_t = np.sum(tf.matmul(alpha, enc_final_state))
                _weights = tf.add(tf.matmul(k_t, Wk), tf.matmul(C[i], W_c))
                H.append(tf.mul(O[i], tf.add(_weights, b)))
            return H

    def transform_output(self, inputs):
        with tf.variable_scope('QRNN/Transform_output'):
            # TODO shapes
            W = tf.get_variable('W', [], initializer=self.initializer)
            b = tf.get_variable('b', [], initializer=self.initializer)
        return tf.add(tf.matmul(inputs, W), b)
