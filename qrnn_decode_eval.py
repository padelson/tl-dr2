import tensorflow as tf


def get_input_from_state(state, embeddings, output_projection, batch_size):
    vocab = tf.nn.xw_plus_b(state, output_projection[0], output_projection[1])
    # word_ids = [tf.argmax(tf.squeeze(i)) for i
    #             in tf.split(0, batch_size, vocab)]
    word_ids = tf.argmax(vocab, axis=1)
    return tf.nn.embedding_lookup(embeddings, tf.arg_max(word_ids))


def advance_step_input(step_input, new_input):
    result = tf.concat(1, [step_input, tf.expand_dims(new_input, 1)])
    return result[:, 1:, :]


def decode_evaluate(decoder, encode_outputs, embedded_dec_inputs,
                    embeddings):
    H = []
    layer_inputs = {}
    layer_outputs = {}
    for i in range(decoder.seq_length):
        if i == 0:
            step_input = tf.fill([decoder.batch_size, decoder.conv_size,
                                  decoder.embedding_size], 0.0)
            layer_inputs[0] = step_input
            new_input = embedded_dec_inputs[:, 0, :]
        else:
            step_input = layer_inputs[0]
            new_input = get_input_from_state(H[-1], embeddings,
                                             decoder.output_projection,
                                             decoder.batch_size)
        step_input = advance_step_input(step_input, new_input)

        for j in range(decoder.num_layers):
            enc_out = tf.squeeze(encode_outputs[j][:, -1, :])
            if i == 0:
                if j < decoder.num_layers-1:
                    step_input, c_t = decoder.conv_with_encode_output(
                                        j,
                                        enc_out,
                                        layer_inputs[j],
                                        decoder.embedding_size,
                                        seq_len=decoder.conv_size)
                    layer_inputs[j+1] = step_input
                    layer_outputs[j] = c_t
                else:
                    h_t, c_t = decoder.conv_with_attention(
                                j, encode_outputs,
                                layer_inputs[j],
                                decoder.num_convs,
                                seq_len=decoder.conv_size)
                    H.append(tf.squeeze(h_t[:, -1:, :]))
                    layer_outputs[j] = c_t
            else:
                if j < decoder.num_layers-1:
                    input_shape = decoder.embedding_size if j == 0 else \
                        decoder.num_convs
                    h_t, c_t = decoder.eval_conv_with_encode_output(
                                j,
                                enc_out,
                                layer_inputs[j],
                                input_shape,
                                layer_outputs[j])
                    layer_inputs[j+1] = advance_step_input(layer_inputs[j],
                                                           h_t)
                    layer_outputs[j] = c_t
                else:
                    h_t, c_t = decoder.eval_conv_with_attention(
                                j,
                                encode_outputs,
                                layer_inputs[j],
                                decoder.num_convs,
                                layer_outputs[j])
                    H.append(tf.squeeze(h_t))
                    layer_outputs[j] = c_t
    return tf.reshape(tf.pack(H), [decoder.batch_size,
                                   decoder.seq_length, decoder.num_convs])
