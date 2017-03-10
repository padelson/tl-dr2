import os
import time

import numpy as np
import tensorflow as tf

import config
import data
import utils


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
    def _seq_f(self, encoder_inputs, decoder_inputs, do_decode):
        return tf.nn.seq2seq.embedding_attention_seq2seq(
            encoder_inputs,
            decoder_inputs,
            self.cell,
            num_encoder_symbols=self.enc_vocab,
            num_decoder_symbols=self.dec_vocab,
            embedding_size=config.HIDDEN_SIZE,
            output_projection=self.output_projection,
            feed_previous=do_decode)

    def _construct_title(self, output_logits):
        with open('output_logits.txt', 'w') as f:
            f.write(str(output_logits))
        output_logits = np.array(output_logits)
        if len(output_logits.shape) > 1:
            outputs = [int(np.argmax(logit, axis=1))
                       for logit in output_logits]
        else:
            outputs = [int(np.argmax(logit)) for logit in output_logits]
        # If there is an EOS symbol in outputs, cut them at that point.
        if config.EOS_ID in outputs:
            outputs = outputs[:outputs.index(config.EOS_ID)]
        # Print out sentence corresponding to outputs.
        return " ".join([tf.compat.as_str(self.inv_dec_dict[output])
                         for output in outputs])

    def _save_gt_headlines(self, set_name, headlines):
        dir_path = os.path.join(self.sess_dir, set_name + '_headlines')
        if set_name == 'dev':
            self.dev_headlines_path = dir_path
        elif set_name == 'test':
            self.test_headlines_path = dir_path
        data.make_dir(dir_path)
        for i, hl in enumerate(headlines):
            filepath = os.path.join(dir_path, str(i) + '.txt')
            with open(filepath, 'w') as f:
                f.write(hl + '\n')

    def _setup_data(self):
        print 'Setting up data...',
        start = time.time()
        with open(os.path.join(self.sess_dir, 'data_path'), 'w') as f:
            f.write(self.data_path)
        data_dir = os.listdir(self.data_path)
        if 'DATA_PROCESSED' not in data_dir:
            raise Exception('Data not processed')
        meta_data = data.split_data(self.data_path, config.BUCKETS)
        self.train_data = meta_data[0]
        self.dev_data = meta_data[1]
        self.test_data = meta_data[2]
        self.enc_dict = meta_data[3]
        self.dec_dict = meta_data[4]
        self.num_train_points = meta_data[5]
        self.dev_headlines = meta_data[6]
        self.test_headlines = meta_data[7]
        self._save_gt_headlines('dev', self.dev_headlines)
        self._save_gt_headlines('test', self.test_headlines)
        self.inv_dec_dict = {v: k for k, v in self.dec_dict.iteritems()}
        self.enc_vocab = len(self.enc_dict)
        self.dec_vocab = len(self.dec_dict)
        print 'Vocab sizes:', self.enc_vocab, self.dec_vocab
        # self._save_dev_gt_headlines()
        print 'Setting up data took', time.time() - start, 'seconds'

    def _setup_sess_dir(self):
        print 'Setting up directory for session'
        self.sess_dir = self.sess_name
        data.make_dir(self.sess_dir)

    def _setup_checkpoints(self):
        print 'Setting up checkpoints directory'
        self.checkpoint_path = os.path.join(self.sess_dir, 'checkpoint')
        data.make_dir(self.checkpoint_path)

    def _setup_results(self):
        print 'Setting up results directory'
        self.results_path = os.path.join(self.sess_dir, 'results')
        data.make_dir(self.results_path)

    def _create_placeholders(self):
        print 'Creating placeholders...  ',
        start = time.time()
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
        print 'Took', time.time() - start, 'seconds'

    def _create_loss(self):
        print 'Creating loss...  ',
        start = time.time()
        if config.NUM_SAMPLES > 0 and config.NUM_SAMPLES < self.dec_vocab:
            w = tf.get_variable('proj_w', [config.HIDDEN_SIZE,
                                           self.dec_vocab])
            b = tf.get_variable('proj_b', [self.dec_vocab])
            self.output_projection = (w, b)

        def sampled_loss(inputs, labels):

            labels = tf.reshape(labels, [-1, 1])
            return tf.nn.sampled_softmax_loss(tf.transpose(w), b, inputs,
                                              labels, config.NUM_SAMPLES,
                                              self.dec_vocab)
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
        print 'Took', time.time() - start, 'seconds'

    def _create_optimizer(self):
        print 'Creating optimizer...  ',
        start = time.time()
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
        print 'Took', time.time() - start, 'seconds'

    def __init__(self, encoder, decoder, data_path, update_params, sess_name):
        self.encoder = encoder
        self.decoder = decoder
        self.data_path = data_path
        self.update_params = update_params
        self.sess_name = sess_name

        self._setup_sess_dir()
        self._setup_data()
        self._setup_checkpoints()
        self._setup_results()
        self._create_placeholders()
        self._create_loss()
        self._create_optimizer()

    def _check_restore_parameters(self, sess, saver):
        ckpt = tf.train.get_checkpoint_state(self.checkpoint_path)
        if ckpt and ckpt.model_checkpoint_path:
            print "Loading parameters"
            saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            print "Initializing fresh parameters"

    def _get_skip_step(self, iteration):
        # How many steps should the model train before it saves weights
        if iteration <= 1:
            return 1  # TODO change this back
        if iteration < self.num_train_points:
            return self.num_train_points / 10
        return self.num_train_points / 2

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

    def evaluate(self, sess, iteration, test=False):
        bucket_losses = []
        summaries = []
        for bucket_index in xrange(len(config.BUCKETS)):
            if len(self.test_data[bucket_index]) == 0:
                print 'Test: empty bucket', bucket_index
                continue
            eval_data = self.test_data if test else self.dev_data
            batch_data = data.get_batch(eval_data, bucket_index,
                                        config.BUCKETS, config.BATCH_SIZE)
            encoder_inputs = batch_data[0]
            decoder_inputs = batch_data[1]
            decoder_masks = batch_data[2]
            _, step_loss, output_logits = self.run_step(sess, encoder_inputs,
                                                        decoder_inputs,
                                                        decoder_masks,
                                                        bucket_index, False)
            loss_text = 'Test bucket:', bucket_index, 'Loss:', step_loss
            print loss_text
            bucket_losses.append(loss_text)
            print np.array(output_logits).shape
            summaries.append(self._construct_title(output_logits))
        path = os.path.join(self.results_path,
                            'iter_' + str(iteration))
        if test:
            path += '_test'
        gt_path = self.test_headlines_path if test else self.dev_headlines_path
        utils.write_results(summaries, bucket_losses, path, gt_path)
        print 'Wrote results to', path

    def train(self):
        assert self.update_params
        saver = tf.train.Saver()
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            self._check_restore_parameters(sess, saver)
            iteration = self.global_step.eval()
            print 'Starting at iteration', iteration
            total_loss = 0
            cur_epoch = iteration / self.num_train_points
            for epoch in range(cur_epoch, config.NUM_EPOCHS):
                print '\n', 'Epoch:', epoch+1
                target = self.num_train_points / config.BATCH_SIZE
                prog = utils.Progbar(target=target)
                prog.update((iteration+1) % target)
                bucket_index = 0
                while True:
                    skip_step = self._get_skip_step(iteration)
                    batch_data = data.get_batch(self.train_data, bucket_index,
                                                config.BUCKETS,
                                                config.BATCH_SIZE,
                                                iteration %
                                                self.num_train_points)
                    encoder_inputs = batch_data[0]
                    decoder_inputs = batch_data[1]
                    decoder_masks = batch_data[2]
                    next_bucket = batch_data[3]
                    _, step_loss, _ = self.run_step(sess, encoder_inputs,
                                                    decoder_inputs,
                                                    decoder_masks,
                                                    bucket_index, True)
                    if next_bucket:
                        bucket_index += 1
                    iteration += 1
                    prog.update((iteration + 1) % target,
                                [("train loss", step_loss)])
                    total_loss += step_loss
                    if bucket_index >= len(config.BUCKETS):
                        saver.save(sess, os.path.join(self.checkpoint_path,
                                                      'summarizer'),
                                                      global_step=iteration)
                        self.evaluate(sess, iteration)
                        break
            self.evaluate(sess, iteration, test=True)

    def summarize(self, inputs):
        saver = tf.train.Saver()

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            self._check_restore_parameters(sess, saver)
            bucket_index, input_data = data.process_input(inputs,
                                                          config.BUCKETS)
            encoder_inputs = input_data[0]
            decoder_inputs = input_data[1]
            decoder_masks = input_data[2]
            _, _, output_logits = self.run_step(sess, encoder_inputs,
                                                decoder_inputs,
                                                decoder_masks,
                                                bucket_index, True)
            title = self.construct_title(output_logits)
            print(title)
