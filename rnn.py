import copy

import tensorflow as tf


def seq2seq(encoder_inputs,
            decoder_inputs,
            cell,
            num_encoder_symbols,
            num_decoder_symbols,
            embedding_size,
            embeddings,
            output_projection=None,
            feed_previous=None):

    with tf.variable_scope('rnn_seq2seq'):
        # encode
        encoder_cell = copy.deepcopy(cell)
        embedded_enc_input = tf.nn.embedding_lookup(embeddings, encoder_inputs)
        attention_states, encoder_state = tf.nn.dynamic_rnn(encoder_cell,
                                                            embedded_enc_input,
                                                            dtype=tf.float32)

        # decode
        embedded_dec_input = [tf.nn.embedding_lookup(embeddings, d)
                              for d in decoder_inputs]

        def decode_with_attention(feed_prev):
            reuse = None if feed_prev else True
            with tf.variable_scope(tf.get_variable_scope(), reuse=reuse):
                loop_f = tf.nn.seq2seq._extract_argmax_and_embed(
                                embeddings,
                                output_projection,
                                True) \
                    if feed_prev else None

                o, s = tf.nn.seq2seq.attention_decoder(embedded_dec_input,
                                                       encoder_state,
                                                       attention_states,
                                                       cell,
                                                       loop_function=loop_f)
                return o

        outputs = tf.cond(feed_previous, lambda: decode_with_attention(True),
                          lambda: decode_with_attention(False))
        # output_len = len(decoder_inputs)
        # outputs = o_s[:, :output_len, :]
        # state = tf.squeeze(o_s[: output_len, :])
        return outputs, None
