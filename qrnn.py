import tensorflow as tf


class QRNN(object):
    def __init__(self, num_encoder_symbols, num_decoder_symbols,
                 embedding_size, num_layers, conv_size):
        # TODO assign args
        self.encode_layers = num_layers
        self.decode_layers = num_layers
        pass

    def seq2seq_f(self, encoder_inputs, decoder_inputs,
                  output_projection=None, feed_previous=False):
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
            decode_outputs.append(self.conv_with_encode_output(decoder_inputs,
                                                               feed_previous,
                                                               enc_out))
        last_hidden = self.attention_output(encode_outputs[-1],
                                            decode_outputs[-1])
        return self.transform_output(last_hidden)

    def fo_pool(Z, F, O, C_prev, H_prev):
        C = tf.mul(F,)
        pass

    def f_pool(Z, F):
        pass

    def conv_layer(self, layer_id):
        pass

    def linear_layer(self, layer_id):
        pass

    def conv_with_encode_output(self, layer_id):
        pass

    def linear_with_encode_output(self, layer_id):
        pass

    def attention_output(self):
        pass

    def transform_output(self):
        pass
