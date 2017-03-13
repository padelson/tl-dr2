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
        self.initializer = tf.random_normal_initializer()

    def get_embeddings(self, word_ids):
        with tf.variable_scope('QRNN/embeddings'):
            W = tf.get_variable('W', [self.num_encoder_symbols,
                                      self.embedding_size],
                                initializer=self.initializer)
            return tf.nn.embedding_lookup(W, word_ids)

    def fo_pool(self, Z, F, O):
        # Z, F, O dims: [batch_size, sequence_length, num_convs]
        H = np.zeros(Z.shape)
        C = np.zeros(Z.shape)
        for i in range(1, Z.shape[1]):
            C[:, i, :] = tf.mul(F[:, i, :], C[:, i-1, :]) + \
                         tf.mul(1-F[:, i, :], Z[:, i, :])
            H[:, i, :] = tf.mul(O[:, i, :], C[:, i, :])
        # i think we want output [batch, seq_len, num_convs]
        return np.array(H)

    def f_pool(self, Z, F):
        # Z, F dims: [batch_size, sequence_length, num_convs]
        H = H = np.zeros(Z.shape)
        for i in range(1, Z.shape[1]):
            H[:, i, :] = tf.mul(F[:, i, :], H[:, i-1, :]) + \
                         tf.mul(1-F[:, i, :])
        return np.array(H)

    def _get_filter_shape(self, inputs):
        return tf.pack([self.conv_size, tf.shape(inputs)[2], 1, self.num_convs*3])

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
            filter_shape = self._get_filter_shape(inputs)
            W = tf.get_variable('W', filter_shape,
                                initializer=self.initializer)
            b = tf.get_variable('b', [self.num_convs*3],
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
        # input dim [batch, seq_len, num_convs or embedding_size]
        shape = (inputs.shape[2], self.num_convs*3)
        with tf.variable_scope('QRNN/Linear/'+str(layer_id)):
            W = tf.get_variable('W', shape,
                                initializer=self.initializer)
            b = tf.get_variable('b', [self.num_convs*3],
                                initializer=self.initializer)
            result = np.zeros([inputs.shape[:2]]+[self.num_convs*3])
            # TODO do this efficiently
            for i in range(inputs.shape[0]):
                input_i = inputs[i, :, :]
                result[i, :, :] = tf.nn.xw_plus_b(input_i, W, b)
            Z, F, O = tf.split(1, 3, result)
            return self.fo_pool(tf.tanh(Z), tf.sigmoid(F), tf.sigmoid(O))

    def conv_with_encode_output(self, layer_id, h_t, inputs=None):
        with tf.variable_scope("QRNN/Variable/Conv_w_enc_out/"+str(layer_id)):
            v_shape = (self.num_convs, self.num_convs*3)
            if inputs is not None:
                filter_shape = self._get_filter_shape(inputs.shape[2])
                W = tf.get_variable('W', filter_shape,
                                    initializer=self.initializer)
            V = tf.get_variable('V', v_shape,
                                initializer=self.initializer)
            b = tf.get_variable('b', [self.num_convs*3],
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
            Z_conv, F_conv, O_conv = tf.split(2, 3, tf.squeeze(conv))
            Z_v, F_v, O_v = tf.split(2, 3, tf.matmul(h_t, V))
            return self.fo_pool(tf.tanh(Z_conv - Z_v),
                                tf.sigmoid(F_conv - F_v),
                                tf.sigmoid(O_conv - O_v))

    def linear_with_encode_output(self, layer_id, h_t, inputs=None):
        # input dim [batch, seq_len, num_convs or embedding_size]
        # h_t dim [batch, 1, num_convs]
        # if inputs is not None:
        #     w_shape = (inputs.shape[2], self.num_convs*3)
        # v_shape = (self.num_convs, self.num_convs*3)
        # with tf.variable_scope('QRNN/Linear_with_encode/'+str(layer_id)):
        #     if inputs is not None:
        #         W = tf.get_variable('W', w_shape,
        #                             initializer=self.initializer)
        #     V = tf.get_variable('V', v_shape,
        #                         initializer=self.initializer)
        #     b = tf.get_variable('b', [self.num_convs*3],
        #                         initializer=self.initializer)
        #
        #     # idk if these matrix multiplications are right
        #     _sum = tf.matmul(h_t, V)
        #     if inputs is not None:
        #         _sum = tf.add(_sum, tf.matmul(inputs, W))
        #     _weighted = tf.add(_sum, b)
        #
        #     result = np.zeros([inputs.shape[:2]]+[self.num_convs*3])
        #     # TODO: do this efficiently
        #     for i in range(inputs.shape[0]):
        #         result_i = tf.matmul(h_t, V) + b
        #         if inputs is not None:
        #             input_i = inputs[i, :, :]
        #             result_i += tf.matmul(input_i, W)
        #         result[i, :, :] = result_i
        #
        #     Z, F, O = tf.split(1, 3, result)
        #     return self.fo_pool(tf.tanh(Z), tf.sigmoid(F), tf.sigmoid(O))
        pass

    def conv_with_attention(self, encode_outputs, inputs):
        # input dim [batch, seq_len, num_convs]
        with tf.variable_scope('QRNN/Conv_with_attention/'):
            filter_shape = self._get_filter_shape(inputs.shape[2])
            attn_weight_shape = [self.num_convs, self.num_convs]

            W_k = tf.get_variable('W_k', attn_weight_shape,
                                  initializer=self.initializer)
            W_c = tf.get_variable('W_c', attn_weight_shape,
                                  initializer=self.initializer)
            b_o = tf.get_variable('b_o', [self.num_convs],
                                  initializer=self.initializer)
            W_conv = tf.get_variable('W_conv', filter_shape,
                                     initializer=self.initializer)
            b = tf.get_variable('b_conv', [self.num_convs*3],
                                initializer=self.initializer)

            num_pads = self.conv_size - 1
            # input dims ~should~ now be [batch_size, sequence_length,
            #                             num_convs, 1]
            padded_input = tf.pad(tf.expand_dims(inputs, -1),
                                  [[0, 0], [num_pads, 0],
                                   [0, 0], [0, 0]],
                                  "CONSTANT")
            conv = tf.nn.conv2d(
                padded_input,
                W_conv,
                strides=[1, 1, 1, 1],
                padding="VALID",
                name="conv") + b
            # conv dims: [batch_size, sequence_length,
            #             1, num_convs*3]
            # squeeze out 3rd D
            # split 4th (now 3rd) dim into 3
            Z, F, O = tf.split(2, 3, tf.squeeze(conv))

            # do normal conv with encode_output
            Z = tf.tanh(Z)
            F = tf.sigmoid(F)
            O = tf.sigmoid(O)

            # calculate attention
            enc_final_state = encode_outputs[-1]
            C = np.zeros(Z.shape)
            H = np.zeros(Z.shape)
            for i in range(1, self.dec_input_size):
                c_i = tf.mul(F[:, i, :], C[:, i-1, :]) + \
                      tf.mul(1-F[:, i, :], Z[:, i, :])
                C[:, i, :] = c_i
                # C_i dim [batch, 1, num_convs]
                # enc_final_state dim [batch, seq_len, num_convs]
                c_dot_h = tf.reduce_sum(tf.mul(c_i, enc_final_state), axis=2)
                # alpha dim [batch, seq_len]
                alpha = tf.nn.softmax(c_dot_h)
                k_t = tf.mul(alpha, enc_final_state)
                h_i = tf.mul(O[:, i, :], (tf.matmul(k_t, W_k) +
                                          tf.matmul(c_i, W_c)+b_o))
                H[:, i, :] = h_i
            return H

    def transform_output(self, inputs):
        # input dim [batch, seq_len, num_convs]
        shape = (inputs.shape[2], self.num_decoder_symbols)
        with tf.variable_scope('QRNN/Transform_output'):
            W = tf.get_variable('W', shape,
                                initializer=self.initializer)
            b = tf.get_variable('b', [self.num_decoder_symbols],
                                initializer=self.initializer)
            # TODO: do efficiently
            result = np.zeros(inputs.shape[:2] + [self.num_decoder_symbols])
            for i in range(inputs.shape[0]):
                input_i = inputs[i, :, :]
                result[i, :, :] = tf.nn.xw_plus_b(input_i, W, b)
        return result

    def seq2seq_f(self, encoder_inputs, decoder_inputs,
                  output_projection=None, training=False):
        # TODO what do i do about output_projection
        encode_outputs = []
        embedded_inputs = np.array(self.get_embeddings(encoder_inputs))
        encoder_inputs = np.array(encoder_inputs)
        decoder_inputs = np.array(decoder_inputs)
        for i in range(self.encode_layers):
            inputs = embedded_inputs if i == 0 else encode_outputs[-1]
            encode_outputs.append(self.conv_layer(i, inputs))
        decode_outputs = []
        for i in range(self.decode_layers):
            # list index i of dim [batch, seq_len, state_size]
            enc_out = tf.squeeze(encode_outputs[i][:, -1, :])
            # TODO what do you feed in during decode lol
            if not training:
                decoder_inputs = None
            is_last_layer = i == (self.decode_layers - 1)
            if not is_last_layer:
                decode_outputs.append(self.conv_with_encode_output(
                                      i,
                                      enc_out,
                                      decoder_inputs))
            else:
                last_state = self.conv_with_attention(encode_outputs,
                                                      decoder_inputs)
        if output_projection is not None:
            return last_state
        else:
            return self.transform_output(last_state)
