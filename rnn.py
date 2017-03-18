import copy

from tensorflow.python.util import nest
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

    with tf.variable_scope('seq2seq_rnn'):
        # encode
        encoder_cell = copy.deepcopy(cell)
        embedded_enc_input = [tf.nn.embedding_lookup(embeddings, i)
                              for i in encoder_inputs]
        encoder_outputs, encoder_state = tf.nn.rnn(encoder_cell,
                                                   embedded_enc_input,
                                                   dtype=tf.float32)

        # decode
        top_states = [tf.reshape(e, [-1, 1, cell.output_size])
                      for e in encoder_outputs]
        attention_states = tf.concat(1, top_states)

        def decode(feed_prev_bool):
            reuse = None if feed_prev_bool else True
            with tf.variable_scope(tf.get_variable_scope(), reuse=reuse):
                loop_function = tf.nn.seq2seq._extract_argmax_and_embed(
                                    embeddings,
                                    output_projection,
                                    True) if feed_previous else None
                embedded_dec_inputs = [tf.nn.embedding_lookup(embeddings, i)
                                       for i in decoder_inputs]
                outputs, state = tf.nn.seq2seq.attention_decoder(
                    embedded_dec_inputs,
                    encoder_state,
                    attention_states,
                    cell,
                    loop_function=loop_function)
                state_list = [state]
                if nest.is_sequence(state):
                    state_list = nest.flatten(state)
                return outputs + state_list

        outputs_and_state = tf.cond(feed_previous,
                                    lambda: decode(True),
                                    lambda: decode(False))
        outputs_len = len(decoder_inputs)
        state_list = outputs_and_state[outputs_len:]
        state = state_list[0]
        if nest.is_sequence(encoder_state):
            state = nest.pack_sequence_as(structure=encoder_state,
                                          flat_sequence=state_list)
        return outputs_and_state[:outputs_len], state
