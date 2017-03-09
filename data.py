import os

import numpy as np


def _read_and_split_file(path):
    with open(path) as f:
        return f.read().split('\n')


def _get_bucket(size, buckets):
    for i in range(len(buckets)):
        s1, s2 = size
        b1, b2 = buckets[i]
        if s1 <= b1 and s2 <= b2:
            return i
    raise Exception('No bucket found for size: ' + str(size))


def _pad_vec(vec, size):
    return vec + [0] * (size - len(vec))


def _vectorize(vec):
    # TODO turn indices into one-hot vectors???
    return vec


def _one_hot_and_mask_data(hl, txt, mask_size, enc_dict, dec_dict):
    txt_size, hl_size = mask_size
    hl_vec = _vectorize(_pad_vec([dec_dict.get(w, enc_dict['<unk>'])
                                  for w in hl.split(' ')], hl_size))
    txt_vec = _pad_vec([enc_dict.get(w, enc_dict['<unk>'])
                        for w in txt.split(' ')], txt_size)
    txt_vec = _vectorize(list(reversed(txt_vec)))
    mask = np.ones(hl_size)
    for i in range(len(hl_vec), hl_size):
        mask[i] = 0
    return np.array(hl_vec), np.array(txt_vec), np.array(mask)


def init_data_buckets(n):
    return {i: {'enc_input': [], 'dec_input': [], 'dec_masks': []}
            for i in range(n)}


def _bucketize_and_split_data(headlines, text, buckets, enc_dict, dec_dict):
    train = init_data_buckets(len(buckets))
    dev = init_data_buckets(len(buckets))
    test = init_data_buckets(len(buckets))
    data_by_bucket = init_data_buckets(len(buckets))
    dev_headlines = []
    test_headlines = []
    for i in range(len(headlines)):
        hl = headlines[i]
        txt = text[i]
        size = (len(txt), len(hl))
        bucket_index = _get_bucket(size, buckets)
        data_by_bucket[bucket_index]['enc_input'].append(txt)
        data_by_bucket[bucket_index]['dec_input'].append(hl)

    for key in data_by_bucket:
        bucket = data_by_bucket[key]
        num_samples = len(bucket['enc_input'])
        split_data_indices = np.random.permutation(np.arange(num_samples))
        for i in range(len(split_data_indices)):
            data_index = split_data_indices[i]
            txt = bucket['enc_input'][data_index]
            hl = bucket['dec_input'][data_index]
            hl_vec, txt_vec, mask = _one_hot_and_mask_data(hl, txt,
                                                           buckets[key],
                                                           enc_dict, dec_dict)
            if i < num_samples * 8 / 10:
                train[key]['enc_input'].append(txt_vec)
                train[key]['dec_input'].append(hl_vec)
                train[key]['dec_masks'].append(mask)
            elif i % 2 == 0:
                dev[key]['enc_input'].append(txt_vec)
                dev[key]['dec_input'].append(hl_vec)
                dev[key]['dec_masks'].append(mask)
                dev_headlines.append(hl)
            else:
                test[key]['enc_input'].append(txt_vec)
                test[key]['dec_input'].append(hl_vec)
                test[key]['dec_masks'].append(mask)
                test_headlines.append(hl)

    return train, dev, test, dev_headlines, test_headlines


def split_data(data_path, buckets):
    # TODO comply with actual format
    headlines = _read_and_split_file(os.path.join(data_path, 'headlines2.txt'))
    text = _read_and_split_file(os.path.join(data_path, 'text2.txt'))
    enc_vocab = _read_and_split_file(os.path.join(data_path, 'enc_vocab.txt'))
    dec_vocab = _read_and_split_file(os.path.join(data_path, 'dec_vocab.txt'))
    enc_dict = {enc_vocab[i]: i for i in range(len(enc_vocab))}
    dec_dict = {dec_vocab[i]: i for i in range(len(dec_vocab))}
    train, dev, test, dev_headlines, test_headlines = \
        _bucketize_and_split_data(headlines, text, buckets,
                                  enc_dict, dec_dict)
    num_samples = len(headlines) * 8 / 10
    return train, dev, test, enc_dict, dec_dict, num_samples, \
        dev_headlines, test_headlines


def make_dir(path):
    """ Create a directory if there isn't one already. """
    try:
        os.mkdir(path)
    except OSError:
        pass


def _reshape(inputs, size, batch_size):
    """ Create batch-major inputs. Batch inputs are just re-indexed inputs
    """
    batch_inputs = []
    for length_id in xrange(size):
        batch_inputs.append(np.array([inputs[batch_id][length_id]
                                      for batch_id in xrange(batch_size)],
                                     dtype=np.int32))
    return batch_inputs


def get_batch(data_buckets, bucket_index, buckets, batch_size, iteration=0):
    bucket = data_buckets[bucket_index]
    next_bucket = False
    start_i = iteration * batch_size
    end_i = (iteration + 1) * batch_size
    if end_i > len(bucket['enc_input']):
        next_bucket = True
        start_i = -batch_size
        end_i = None
        enc_input = bucket['enc_input'][start_i:]
        dec_input = bucket['dec_input'][start_i:]
        dec_masks = bucket['dec_masks'][start_i:]
    else:
        enc_input = bucket['enc_input'][start_i:end_i]
        dec_input = bucket['dec_input'][start_i:end_i]
        dec_masks = bucket['dec_masks'][start_i:end_i]
    enc_size, dec_size = buckets[bucket_index]
    enc_matrix = _reshape(enc_input, enc_size, batch_size)
    dec_matrix = _reshape(dec_input, dec_size, batch_size)
    mask_matrix = _reshape(dec_masks, dec_size, batch_size)
    return enc_matrix, dec_matrix, mask_matrix, next_bucket


def process_input(inputs, buckets, enc_dict, dec_dict):
    txt_size = len(inputs)
    bucket_index = _get_bucket((txt_size, 0), buckets)
    bucket = buckets[bucket_index]
    hl_vec, txt_vec, mask = _one_hot_and_mask_data('', inputs, bucket,
                                                   enc_dict, dec_dict)
    input_data = ([txt_vec], [hl_vec], [mask])
    return bucket_index, input_data
