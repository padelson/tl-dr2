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

    with tf.get_variable_scope('rnn_seq2seq'):
        # encode
        encoder_cell = copy.deepcopy(cell)
        embedded_enc_input = tf.nn.embedding_lookup(embeddings, encoder_inputs)
        encoder_outputs, encoder_state = tf.nn.dynamic_rnn(cell,
                                                           embedded_enc_input)

        # reshape attention states
        top_states = [tf.reshape(e, [-1, 1, cell.output_size]
                                     for e in encoder_outputs)]
        attention_states = tf.concat(top_states, 1)

        # decode
        embedded_dec_input = tf.nn.embedding_lookup(embeddings, decoder_inputs)
        loop_function = tf.nn.seq2seq._extract_argmax_and_embed(
                            embeddings,
                            output_projection,
                            True) \
                            if feed_previous else None

        return tf.nn.seq2seq.attention_decoder(embedded_dec_input,
                                               encoder_state,
                                               attention_states,
                                               cell,
                                               loop_function=loop_function)
