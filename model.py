import os

import numpy as np
import tensorflow as tf

import config
import data


class Encoder(object):
    def __init__(self, size):
        self.size = size

    def encode(self, inputs, masks, encoder_state_input):
        """
        In a generalized encode function, you pass in your inputs,
        masks, and an initial
        hidden state input into this function.

        :param inputs: Symbolic representations of your input
        :param masks: this is to make sure tf.nn.dynamic_rnn doesn't iterate
                      through masked steps
        :param encoder_state_input: (Optional) pass this as initial hidden
                                    state to tf.nn.dynamic_rnn to build
                                    conditional representations
        :return: an encoded representation of your input.
                 It can be context-level representation, word-level
                 representation, or both.
        """
        # TODO
        return


class Decoder(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def decode(self, knowledge_rep):
            """
            takes in a knowledge representation
            and output a probability estimation over
            all paragraph tokens on which token should be
            the start of the answer span, and which should be
            the end of the answer span.

            :param knowledge_rep: it is a representation of the paragraph and
                                  question, decided by how you choose to
                                  implement the encoder
            :return:
            """
            # TODO
            return


class Summarizer(object):
    def _seq_f(encoder_inputs, decoder_inputs, do_decode):
        return tf.contrib.legacy_seq2seq.embedding_attention_seq2seq(
            encoder_inputs,
            decoder_inputs,
            self.cell,
            num_encoder_symbols=config.ENC_VOCAB,
            num_decoder_symbols=config.DEC_VOCAB,
            embedding_size=config.HIDDEN_SIZE,
            output_projection=self.output_projection,
            feed_previous=do_decode)

    def _setup_data(self):
        data_dir = os.path.listdir(self.data_path)
        if 'DATA_PROCESSED' not in data_dir:
            data.process_data(self.data_path)
        self.train_data, self.dev_data, self.test_data = data.split_data(
                                                            self.data_path)

    def _setup_checkpoints(self):
        self.checkpoint_path = 'checkpoint_' + self.data_path
        data.make_dir(self.checkpoint_path)

    def _create_placeholders(self):
        print 'Creating placeholders'
        self.encoder_inputs = []
        self.decoder_inputs = []
        self.decoder_masks = []
        for i in xrange(config.BUCKETS[-1][0]):  # Last bucket is the biggest.
            self.encoder_inputs.append(tf.placeholder(tf.int32, shape=[None],
                                       name='encoder{}'.format(i)))
        for i in xrange(config.BUCKETS[-1][1] + 1):
            self.decoder_inputs.append(tf.placeholder(tf.int32, shape=[None],
                                       name='decoder{}'.format(i)))
            self.decoder_masks.append(tf.placeholder(tf.float32, shape=[None],
                                      name='mask{}'.format(i)))

        # Our targets are decoder inputs shifted by one (to ignore <s> symbol)
        self.targets = [self.decoder_inputs[i + 1]
                        for i in xrange(len(self.decoder_inputs) - 1)]

    def _create_loss(self):
        print 'Creating loss'
        if config.NUM_SAMPLES > 0 and config.NUM_SAMPLES < config.DEC_VOCAB:
            w = tf.get_variable('proj_w', [config.HIDDEN_SIZE,
                                           config.DEC_VOCAB])
            b = tf.get_variable('proj_b', [config.DEC_VOCAB])
            self.output_projection = (w, b)

        def sampled_loss(inputs, labels):
            labels = tf.reshape(labels, [-1, 1])
            return tf.nn.sampled_softmax_loss(tf.transpose(w), b, inputs,
                                              labels, config.NUM_SAMPLES,
                                              config.DEC_VOCAB)
        self.softmax_loss = sampled_loss

        single_cell = tf.nn.rnn_cell.GRUCell(config.HIDDEN_SIZE)
        self.cell = tf.nn.rnn_cell.MultiRNNCell([single_cell] *
                                                config.NUM_LAYERS)

        if self.update_params:
            self.outputs, self.losses = tf.nn.seq2seq.model_with_buckets(
                                        self.encoder_inputs,
                                        self.decoder_inputs,
                                        self.targets,
                                        self.decoder_masks,
                                        config.BUCKETS,
                                        lambda x, y: self._seq_f(x, y, False),
                                        softmax_loss_function=self.softmax_loss
                                        )
        else:
            self.outputs, self.losses = tf.nn.seq2seq.model_with_buckets(
                                        self.encoder_inputs,
                                        self.decoder_inputs,
                                        self.targets,
                                        self.decoder_masks,
                                        config.BUCKETS,
                                        lambda x, y: self._seq_f(x, y, True),
                                        softmax_loss_function=self.softmax_loss
                                        )

    def _create_optimizer(self):
        print 'Creating optimizer'
        with tf.variable_scope('training') as scope:
            self.global_step = tf.Variable(0, dtype=tf.int32, trainable=False,
                                           name='global_step')
            if self.update_params:
                self.optimizer = tf.train.GradientDescentOptimizer(config.LR)
                trainables = tf.trainable_variables()
                self.gradient_norms = []
                self.train_ops = []
                for bucket in xrange(len(config.BUCKETS)):
                    grads = tf.gradients(self.losses[bucket], trainables)
                    max_g = config.MAX_GRAD_NORM
                    clipped_grads, norm = tf.clip_by_global_norm(grads, max_g)
                    self.gradient_norms.append(norm)
                    self.train_ops.append(self.optimizer.apply_gradients(
                                          zip(clipped_grads,
                                              trainables),
                                          global_step=self.global_step))
                    print('Created opt for bucket {}'.format(bucket))

    def __init__(self, encoder, decoder, data_path, update_params):
        self.encoder = encoder
        self.decoder = decoder
        self.data_path = data_path
        self.update_params = update_params

        self._setup_data()
        self._setup_checkpoints()
        self._create_placeholders()
        self._create_loss()
        self._create_optimizer()

    def _check_restore_parameters(sess, saver):
        ckpt = tf.train.get_checkpoint_state(os.path.dirname(config.CKPT_PATH +
                                                             '/checkpoint'))
        if ckpt and ckpt.model_checkpoint_path:
            print "Loading parameters for the Chatbot"
            saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            print "Initializing fresh parameters for the Chatbot"

    def _get_skip_step(iteration):
        # How many steps should the model train before it saves weights
        if iteration < 100:
            return 30
        return 100

    def run_step(self, sess, encoder_inputs, decoder_inputs, decoder_masks,
                 bucket_id, update_params):
        encoder_size, decoder_size = config.BUCKETS[bucket_id]
        input_feed = {}
        for step in xrange(encoder_size):
            input_feed[self.encoder_inputs[step].name] = encoder_inputs[step]
        for step in xrange(decoder_size):
            input_feed[self.decoder_inputs[step].name] = decoder_inputs[step]
            input_feed[self.decoder_masks[step].name] = decoder_masks[step]
        last_target = self.decoder_inputs[decoder_size].name
        input_feed[last_target] = np.zeros([config.BATCH_SIZE], dtype=np.int32)

        # output feed: depends on whether we do a backward step or not.
        if update_params:
            output_feed = [self.train_ops[bucket_id],  # update that does SGD.
                           self.gradient_norms[bucket_id],  # gradient norm.
                           self.losses[bucket_id]]  # loss for this batch.
        else:
            output_feed = [self.losses[bucket_id]]  # loss for this batch.
            for step in xrange(decoder_size):  # output logits.
                output_feed.append(self.outputs[bucket_id][step])

        outputs = sess.run(output_feed, input_feed)
        if update_params:
            return outputs[1], outputs[2], None  # Grad norm, loss, no outputs.
        else:
            return None, outputs[0], outputs[1:]  # No grad norm, loss, outputs

    def train(self):
        assert self.update_params
        saver = tf.train.Saver()
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            self._check_restore_parameters(sess, saver)
            iteration = self.global_step.eval()
            total_loss = 0
            for epoch in range(config.NUM_EPOCHS):
                bucket_index = 0
                while True:
                    skip_step = self._get_skip_step(iteration)
                    batch_data = data.get_batch(self.train_data, bucket_index,
                                                config.BATCH_SIZE)
                    encoder_inputs = batch_data[0]
                    decoder_inputs = batch_data[1]
                    decoder_masks = batch_data[2]
                    next_bucket = batch_data[3]
                    if next_bucket:
                        bucket_index += 1
                    if bucket_index > len(config.BUCKETS):
                        break
                    _, step_loss, _ = self.run_step(sess, encoder_inputs,
                                                    decoder_inputs,
                                                    decoder_masks,
                                                    bucket_index, True)
                    total_loss += step_loss
                    iteration += 1
                    if iteration % skip_step == 0:
                        print  # TODO print info
                        saver.save(sess, self.checkpoint_path,
                                   global_step=self.global_step)
                        if iteration % (10 * skip_step) == 0:
                            self.evaluate(sess)
            self.evaluate(sess, test=True)

    def evaluate(self, sess, test=False):
        for bucket_index in xrange(len(config.BUCKETS)):
            if len(self.test_data[bucket_index]) == 0:
                print 'Test: empty bucket', bucket_index
                continue
            eval_data = self.test_data if test else self.dev_data
            batch_data = data.get_batch(eval_data, bucket_index,
                                        config.BATCH_SIZE)
            encoder_inputs = batch_data[0]
            decoder_inputs = batch_data[1]
            decoder_masks = batch_data[2]
            encoder_inputs, decoder_inputs, decoder_masks = data.get_batch()
            _, step_loss, _ = self.run_step(sess, encoder_inputs,
                                            decoder_inputs,
                                            decoder_masks,
                                            bucket_index, False)
            print 'Test bucket:', bucket_index, 'Loss:', step_loss

    def _construct_title(self, output_logits):
        outputs = [int(np.argmax(logit, axis=1)) for logit in output_logits]
        # If there is an EOS symbol in outputs, cut them at that point.
        if config.EOS_ID in outputs:
            outputs = outputs[:outputs.index(config.EOS_ID)]
        # Print out sentence corresponding to outputs.
        return " ".join([tf.compat.as_str(self.inv_dec_vocab[output])
                         for output in outputs])

    def summarize(self, input):
        saver = tf.train.Saver()

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            self._check_restore_parameters(sess, saver)
            bucket_index, input_data = data.process_input(input)
            encoder_inputs = input_data[0]
            decoder_inputs = input_data[1]
            decoder_masks = input_data[2]
            _, _, output_logits = self.run_step(sess, encoder_inputs,
                                                decoder_inputs,
                                                decoder_masks,
                                                bucket_index, True)
            title = self.construct_title(output_logits)
            print(title)
