import os
import time

import numpy as np
import tensorflow as tf

import config
import data
import utils

import rnn
from qrnn import init_encoder_and_decoder, seq2seq_f


class Summarizer(object):
    def _seq_f(self, encoder_inputs, decoder_inputs, do_decode):
        if self.model == 'rnn':
            return rnn.seq2seq(
                encoder_inputs,
                decoder_inputs,
                self.cell,
                num_encoder_symbols=self.enc_vocab,
                num_decoder_symbols=self.dec_vocab,
                embedding_size=config.EMBED_SIZE,
                embeddings=self.embeddings,
                output_projection=self.output_projection,
                feed_previous=do_decode)

        enc_seq_length = len(encoder_inputs)
        dec_seq_length = len(decoder_inputs)
        encoder, decoder = init_encoder_and_decoder(self.enc_vocab,
                                                    self.dec_vocab,
                                                    enc_seq_length,
                                                    dec_seq_length,
                                                    config.EMBED_SIZE,
                                                    config.NUM_LAYERS,
                                                    config.CONV_SIZE,
                                                    config.HIDDEN_SIZE,
                                                    self.output_projection)
        return seq2seq_f(encoder, decoder, encoder_inputs, decoder_inputs,
                         do_decode, self.embeddings, self.center_conv)

    def _construct_seq(self, output_logits):
        output_logits = np.array(output_logits)
        outputs = [int(np.argmax(logit)) for logit in output_logits]
        # If there is an EOS symbol in outputs, cut them at that point.
        if config.EOS_ID in outputs:
            outputs = outputs[:outputs.index(config.EOS_ID)+1]
        # Print out sentence corresponding to outputs.
        return " ".join([tf.compat.as_str(self.inv_dec_dict[output])
                         for output in outputs])

    def _setup_data(self):
        print 'Setting up data...',
        start = time.time()
        with open(os.path.join(self.sess_dir, 'data_path'), 'w') as f:
            f.write(self.data_path)
        meta_data = data.load_data(self.data_path, config.BUCKETS)
        self.train_data = meta_data[0]
        self.dev_data = meta_data[1]
        self.test_data = meta_data[2]
        self.enc_dict = meta_data[3]
        self.dec_dict = meta_data[4]
        self.num_train_points = meta_data[5]
        self.dev_headlines_path = meta_data[6]
        self.test_headlines_path = meta_data[7]
        self.inv_dec_dict = {v: k for k, v in self.dec_dict.iteritems()}
        self.enc_vocab = len(self.enc_dict)
        self.dec_vocab = len(self.dec_dict)
        print 'Setting up data took', time.time() - start, 'seconds'
        print 'Encoder vocab size:', len(self.enc_dict)
        print 'Encoder vocab size:', len(self.dec_dict)
        print 'Number of training samples', [len(x['dec_input']) for i, x in
                                             self.train_data.iteritems()]

    def _setup_sess_dir(self):
        print 'Setting up directory for session'
        self.sess_dir = os.path.join('/datadrive', self.sess_name)
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
        self.feed_prev_placeholder = tf.placeholder(tf.bool, shape=[],
                                                    name='feed_prev')
        self.learning_rate = tf.placeholder(tf.float32, shape=[], name='lr')
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
            proj_w_size = config.HIDDEN_SIZE
            w = tf.get_variable('proj_w', [proj_w_size,
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
        embed_init = tf.contrib.layers.xavier_initializer()
        if self.pretrained:
            pad = tf.zeros([1, config.EMBED_SIZE])
            flags = tf.Variable(embed_init([3, config.EMBED_SIZE],
                                dtype=tf.float32))
            embeddings = tf.constant(data.load_embeddings(self.data_path),
                                     dtype=tf.float32)
            self.embeddings = tf.concat(0, [pad, flags, embeddings])
        else:
            self.embeddings = tf.Variable(embed_init([self.enc_vocab,
                                                      config.EMBED_SIZE]),
                                          dtype=tf.float32)
        feed_prev = self.feed_prev_placeholder
        self.outputs, self.losses = tf.nn.seq2seq.model_with_buckets(
                                    self.encoder_inputs,
                                    self.decoder_inputs,
                                    self.targets,
                                    self.decoder_masks,
                                    config.BUCKETS,
                                    lambda x, y: self._seq_f(x, y, feed_prev),
                                    softmax_loss_function=self.softmax_loss
                                    )
        # If we use output projection, we need to project outputs for decoding.
        cur = None
        bucket = 0

        def do_nothing(): return cur

        def project_outputs():
            if self.output_projection:
                    return [tf.matmul(output,
                            self.output_projection[0]) +
                            self.output_projection[1]
                            for output in self.outputs[bucket]]
            return tf.constant(False)

        for bucket in xrange(len(config.BUCKETS)):
            cur = self.outputs[bucket]
            self.outputs[bucket] = tf.cond(self.feed_prev_placeholder,
                                           do_nothing,
                                           project_outputs)

        print 'Took', time.time() - start, 'seconds'

    def _create_optimizer(self):
        print 'Creating optimizer...  ',
        start = time.time()
        with tf.variable_scope('training'):
            self.global_step = tf.Variable(0, dtype=tf.int32, trainable=False,
                                           name='global_step')
            self.bucket_step = tf.Variable(0, dtype=tf.int32, trainable=False,
                                           name='bucket_step')
            self.epoch = tf.Variable(0, dtype=tf.int32, trainable=False,
                                     name='epoch')
            self.bucket_index = tf.Variable(0, dtype=tf.int32, trainable=False,
                                            name='bucket_index')
            if self.create_opt:
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

    def __init__(self, data_path, create_opt, sess_name, model, pretrained,
                 center_conv=False):
        self.data_path = data_path
        self.create_opt = create_opt
        self.sess_name = sess_name
        self.model = model
        self.pretrained = pretrained
        self.center_conv = center_conv
        self.lr = config.LR
        self.dev_loss = None
        print '###Initializing', model, 'model'

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

    def run_step(self, sess, encoder_inputs, decoder_inputs, decoder_masks,
                 batch_size, bucket_id, update_params):
        encoder_size, decoder_size = config.BUCKETS[bucket_id]
        input_feed = {}
        for step in xrange(encoder_size):
            input_feed[self.encoder_inputs[step].name] = encoder_inputs[step]
        for step in xrange(decoder_size):
            input_feed[self.decoder_inputs[step].name] = decoder_inputs[step]
            input_feed[self.decoder_masks[step].name] = decoder_masks[step]
        last_target = self.decoder_inputs[decoder_size].name
        input_feed[last_target] = np.zeros([batch_size], dtype=np.int32)
        input_feed[self.learning_rate] = self.lr
        input_feed[self.feed_prev_placeholder] = not update_params
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

    def evaluate(self, sess, train_losses, iteration, test=False):
        bucket_loss_texts = []
        bucket_losses = []
        summaries = []
        eval_iter = 0
        avg_loss = np.sum(train_losses) / len(train_losses)
        print 'Average train loss', avg_loss
        eval_start = time.time()
        for bucket_index in xrange(len(config.BUCKETS)):
            bucket_count = len(self.test_data[bucket_index])
            if bucket_count == 0:
                print 'Test: empty bucket', bucket_index
                continue
            bucket_loss = 0
            while True:
                eval_data = self.test_data if test else self.dev_data
                batch_data = data.get_batch(eval_data, bucket_index,
                                            config.BUCKETS, config.BATCH_SIZE,
                                            eval_iter)
                encoder_inputs = batch_data[0]
                decoder_inputs = batch_data[1]
                decoder_masks = batch_data[2]
                done = batch_data[3]
                _, step_loss, output_logits = self.run_step(sess,
                                                            encoder_inputs,
                                                            decoder_inputs,
                                                            decoder_masks,
                                                            config.BATCH_SIZE,
                                                            bucket_index,
                                                            False)
                bucket_loss += step_loss

                output_logits = np.array(output_logits)
                for i in xrange(config.BATCH_SIZE):
                    summaries.append(self._construct_seq(
                                     output_logits[:, i, :]))
                eval_iter += 1
                if done:
                    eval_iter = 0
                    break
            loss_text = 'Test bucket:', bucket_index, 'Avg Loss:', \
                (bucket_loss / bucket_count)
            print loss_text
            bucket_losses.append(bucket_loss)
            bucket_loss_texts.append(loss_text)
        path = os.path.join(self.results_path,
                            'iter_' + str(iteration))
        if self.dev_loss is not None:
            if sum([self.dev_loss[i] < bucket_losses[i]
                    for i in range(len(bucket_losses))]) > 0:
                self.lr /= 2
                print 'Learning rate adjusted'
        self.dev_loss = bucket_losses
        if test:
            path += '_test'
        data.make_dir(path)
        gt_path = self.test_headlines_path if test else self.dev_headlines_path
        utils.write_results(summaries, avg_loss, bucket_loss_texts,
                            path, gt_path)
        print 'Wrote results to', path
        print 'Evaluation took', time.time() - eval_start

    def train(self):

        def num_steps(bucket):
            return int(np.ceil(len(bucket) / float(config.BATCH_SIZE)))

        saver = tf.train.Saver()
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            self._check_restore_parameters(sess, saver)
            iteration = self.global_step.eval()
            bucket_sizes = [num_steps(v['dec_input']) for k, v
                            in self.train_data.iteritems()]
            cur_epoch = self.epoch.eval()
            bucket_index = self.bucket_index.eval()
            step_iter = self.bucket_step.eval()
            print 'Starting at iter:', iteration, 'epoch:', cur_epoch+1, \
                  'bucket index:', bucket_index, 'bucket step:', step_iter
            for epoch in range(cur_epoch, config.NUM_EPOCHS):
                epoch_start = time.time()
                total_losses = []
                sess.run(tf.assign(self.epoch, epoch))
                print '\n', 'Epoch:', epoch+1
                print 'Bucket sizes', bucket_sizes
                print 'Learning rate:', self.lr
                if sum(bucket_sizes) > 1:
                    prog = utils.Progbar(target=bucket_sizes[bucket_index])
                end_while = False
                while True:
                    batch_start = time.time()
                    batch_data = data.get_batch(self.train_data, bucket_index,
                                                config.BUCKETS,
                                                config.BATCH_SIZE,
                                                step_iter)
                    encoder_inputs = batch_data[0]
                    decoder_inputs = batch_data[1]
                    decoder_masks = batch_data[2]
                    next_bucket = batch_data[3]
                    _, step_loss, _ = self.run_step(sess, encoder_inputs,
                                                    decoder_inputs,
                                                    decoder_masks,
                                                    config.BATCH_SIZE,
                                                    bucket_index, True)
                    if next_bucket:
                        step_iter = sess.run(tf.assign(self.bucket_step, 0))
                        bucket_index = sess.run(tf.assign(self.bucket_index,
                                                          bucket_index+1))
                    else:
                        step_iter = sess.run(tf.assign(self.bucket_step,
                                                       step_iter+1))
                    if bucket_index >= len(config.BUCKETS):
                        end_while = True
                        bucket_index = sess.run(tf.assign(self.bucket_index,
                                                          0))
                    if next_bucket and sum(bucket_sizes) > 1:
                        print
                        prog.update(bucket_sizes[bucket_index-1],
                                    [("train loss", step_loss),
                                     ('batch runtime',
                                      time.time() - batch_start)])
                        prog = utils.Progbar(target=bucket_sizes[bucket_index])
                    total_losses.append(step_loss)
                    if end_while or \
                       iteration == 200 or \
                       (iteration > 0 and iteration % 1000 == 0):
                        saver.save(sess, os.path.join(self.checkpoint_path,
                                                      'summarizer'),
                                   global_step=iteration)
                        if iteration == 200 or end_while or epoch < 2 \
                                or iteration % 2000 == 0:
                            self.evaluate(sess, total_losses, iteration)
                    iteration += 1
                    if sum(bucket_sizes) == 1:
                        print 'Train loss', step_loss
                    if end_while:
                        print 'Epoch', epoch+1, 'took', time.time()-epoch_start
                        break
                    if sum(bucket_sizes) > 1:
                        prog.update(step_iter,
                                    [("train loss", step_loss),
                                     ('batch runtime',
                                      time.time() - batch_start)])

            self.evaluate(sess, total_losses, iteration, test=True)
            saver.save(sess, os.path.join(self.checkpoint_path,
                                          'summarizer'),
                       global_step=iteration)

    def summarize(self, inputs):
        saver = tf.train.Saver()

        with tf.Session() as sess:
            summaries = []
            sess.run(tf.global_variables_initializer())
            self._check_restore_parameters(sess, saver)
            bucket_index, input_data = data.process_input(inputs,
                                                          config.BUCKETS,
                                                          self.enc_dict,
                                                          self.dec_dict)
            encoder_inputs = input_data[0]
            decoder_inputs = input_data[1]
            decoder_masks = input_data[2]
            _, _, output_logits = self.run_step(sess, encoder_inputs,
                                                decoder_inputs,
                                                decoder_masks,
                                                len(input_data[0]),
                                                bucket_index, False)
            output_logits = np.array(output_logits)
            for i in xrange(output_logits.shape[1]):
                summaries.append(self._construct_seq(
                                 output_logits[:, i, :]))
            return summaries
