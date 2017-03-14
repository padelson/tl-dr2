import numpy as np
import tensorflow as tf


class QRNN(object):
    def __init__(self, num_symbols, batch_size, seq_length,
                 embedding_size, num_layers, conv_size, num_convs):
        self.num_symbols = num_symbols
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.embedding_size = embedding_size
        self.num_layers = num_layers
        self.conv_size = conv_size
        self.num_convs = num_convs
        self.initializer = tf.random_normal_initializer()

    def get_embeddings(self, embeddings, word_ids):
        if word_ids is None:
            return None
        # with tf.variable_scope('QRNN/embeddings', reuse=True):
        #     W = tf.get_variable('W', [self.num_symbols,
        #                               self.embedding_size],
        #                         initializer=self.initializer)
        return tf.nn.embedding_lookup(embeddings, word_ids)

    def fo_pool(self, Z, F, O):
        # Z, F, O dims: [batch_size, sequence_length, num_convs]
        # H = tf.fill(tf.shape(Z), 0.0)
        # C = tf.fill(tf.shape(Z), 0.0)
        H = [tf.fill(tf.pack([tf.shape(Z)[0], 1, tf.shape(Z)[2]]), 0.0)]
        C = [tf.fill(tf.pack([tf.shape(Z)[0], 1, tf.shape(Z)[2]]), 0.0)]
        for i in range(1, self.seq_length):
            c_i = tf.mul(F[:, i, :], C[-1]) + \
                         tf.mul(1-F[:, i, :], Z[:, i, :])
            # C[:, i, :] = c_i
            C.append(c_i)
            h_i = tf.mul(O[:, i, :], c_i)
            # H[:, i, :] = h_i
            H.append(tf.squeeze(h_i))
        # i think we want output [batch, seq_len, num_convs]
        return tf.reshape(tf.pack(H), tf.shape(Z))

    def f_pool(self, Z, F, sequence_length):
        # Z, F dims: [batch_size, sequence_length, num_convs]
        H = tf.fill(tf.shape(Z), 0)
        for i in range(1, self.seq_length):
            H[:, i, :] = tf.mul(F[:, i, :], H[:, i-1, :]) + \
                         tf.mul(1-F[:, i, :])
        return np.array(H)

    def _get_filter_shape(self, input_shape):
        return [self.conv_size, input_shape, 1, self.num_convs*3]

    # convolution dimension results maths
    # out_height = ceil(float(in_height - filter_height + 1) /
    #                   float(strides[1])) = sequence_length
    # out_width  = ceil(float(in_width - filter_width + 1) /
    #                   float(strides[2])) = 1
    # in_height = sequence_length + filter_height - 1
    # filter_height = conv_size
    # in_width = embedding_size
    # filter_width = embedding_size

    def conv_layer(self, layer_id, inputs, input_shape):
        with tf.variable_scope("QRNN/Variable/Convolution/"+str(layer_id)):
            filter_shape = self._get_filter_shape(input_shape)
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

    # def linear_layer(self, layer_id, inputs):
    #     # input dim [batch, seq_len, num_convs or embedding_size]
    #     shape = (inputs.shape[2], self.num_convs*3)
    #     with tf.variable_scope('QRNN/Linear/'+str(layer_id)):
    #         W = tf.get_variable('W', shape,
    #                             initializer=self.initializer)
    #         b = tf.get_variable('b', [self.num_convs*3],
    #                             initializer=self.initializer)
    #         result = np.zeros([inputs.shape[:2]]+[self.num_convs*3])
    #         # TODO do this efficiently
    #         for i in range(inputs.shape[0]):
    #             input_i = inputs[i, :, :]
    #             result[i, :, :] = tf.nn.xw_plus_b(input_i, W, b)
    #         Z, F, O = tf.split(1, 3, result)
    #         return self.fo_pool(tf.tanh(Z), tf.sigmoid(F), tf.sigmoid(O))

    def conv_with_encode_output(self, layer_id, h_t, inputs=None,
                                input_shape=None, pool=True):
        pooling = self.fo_pool if pool else lambda x, y, z: (x, y, z)
        with tf.variable_scope("QRNN/Variable/Conv_w_enc_out/"+str(layer_id)):
            v_shape = (self.num_convs, self.num_convs*3)
            V = tf.get_variable('V', v_shape,
                                initializer=self.initializer)
            b = tf.get_variable('b', [self.num_convs*3],
                                initializer=self.initializer)
            Z_v, F_v, O_v = tf.split(2, 3, tf.matmul(h_t, V))
            if inputs is not None:
                filter_shape = self._get_filter_shape(input_shape)
                W = tf.get_variable('W', filter_shape,
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
                return pooling(tf.tanh(Z_conv + Z_v),
                               tf.sigmoid(F_conv + F_v),
                               tf.sigmoid(O_conv + O_v))
            else:
                return pooling(tf.tanh(Z_v),
                               tf.sigmoid(F_v),
                               tf.sigmoid(O_v))

    # def linear_with_encode_output(self, layer_id, h_t, inputs=None):
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

    def conv_with_attention(self, layer_id, encode_outputs, inputs,
                            input_shape):
        # input dim [batch, seq_len, num_convs]
        with tf.variable_scope('QRNN/Conv_with_attention/'):
            attn_weight_shape = [self.num_convs, self.num_convs]

            W_k = tf.get_variable('W_k', attn_weight_shape,
                                  initializer=self.initializer)
            W_c = tf.get_variable('W_c', attn_weight_shape,
                                  initializer=self.initializer)
            b_o = tf.get_variable('b_o', [self.num_convs],
                                  initializer=self.initializer)

            h_t = tf.squeeze(encode_outputs[-1][:, -1, :])
            Z, F, O = self.conv_with_encode_output(layer_id, h_t, inputs,
                                                   input_shape, pool=False)

            # calculate attention
            enc_final_state = encode_outputs[-1]
            C = tf.fill(tf.shape(Z), 0.0)
            H = tf.fill(tf.shape(Z), 0.0)
            for i in range(1, self.seq_length):
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


def init_encoder_and_decoder(num_encoder_symbols, num_decoder_symbols,
                             batch_size, enc_seq_length, dec_seq_length,
                             embedding_size, num_layers, conv_size, num_convs):
    encoder = QRNN(num_encoder_symbols, batch_size, enc_seq_length,
                   embedding_size, num_layers, conv_size, num_convs)
    decoder = QRNN(num_decoder_symbols, batch_size, dec_seq_length,
                   embedding_size, num_layers, conv_size, num_convs)
    return encoder, decoder


def seq2seq_f(encoder, decoder, encoder_inputs, decoder_inputs,
              output_projection, training, embeddings):
    # TODO what do i do about output_projection
    # inputs are lists of placeholders, each one is shape [None]
    # self.enc_input_size = len(encoder_inputs)
    # self.dec_input_size = len(decoder_inputs)
    encode_outputs = []
    # pack inputs to be shape [sequence_length, batch_size]
    encoder_inputs = tf.transpose(tf.pack(encoder_inputs))
    decoder_inputs = tf.transpose(tf.pack(decoder_inputs))

    # embed to be shape [batch_size, sequence_length, embed_size]
    embedded_enc_inputs = encoder.get_embeddings(embeddings, encoder_inputs)
    # TODO put embeddings in a centralized location
    embedded_dec_inputs = decoder.get_embeddings(embeddings, decoder_inputs)

    for i in range(encoder.num_layers):
        inputs = embedded_enc_inputs if i == 0 else encode_outputs[-1]
        input_shape = encoder.embedding_size if i == 0 else encoder.num_convs
        encode_outputs.append(encoder.conv_layer(i, inputs, input_shape))

    decode_outputs = []
    for i in range(decoder.num_layers):
        # list index i of dim [batch, seq_len, state_size]
        enc_out = tf.squeeze(encode_outputs[i][:, -1, :])
        # TODO what do you feed in during decode lol
        inputs = embedded_dec_inputs if i == 0 else decode_outputs[-1]
        input_shape = decoder.embedding_size if i == 0 else decoder.num_convs
        inputs, input_shape = tf.cond(True, lambda: (inputs, input_shape),
                                      lambda: (None, None))
        is_last_layer = i == (decoder.num_layers - 1)
        if not is_last_layer:
            decode_outputs.append(decoder.conv_with_encode_output(
                                  i,
                                  enc_out,
                                  inputs,
                                  input_shape))
        else:
            last_state = decoder.conv_with_attention(i, encode_outputs,
                                                     inputs, input_shape)
    if output_projection is not None:
        return last_state
    else:
        return decoder.transform_output(last_state)
