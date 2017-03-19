'''Data processing for 2l-dr: headline generation (take 2)
Functions to load dataset and process
'''
import json
import os

import numpy as np

import config


def _read_and_split_file(path):
    '''  Open a file and split by newlines  '''
    with open(path) as f:
        result = f.read().split('\n')
        if result[-1] == '':
            return result[:-1]
        return result


def load_embeddings(path):
    '''  Load word embeddings from file path  '''
    with open(os.path.join(path, 'embeddings.txt')) as f:
        vecs = json.loads(f.read())
        if vecs[-1] == '':
            return np.stack(vecs[:-1])
        return np.stack(vecs)


def _get_bucket(size, buckets):
    '''  Find the smallest bucket greater than or equal to the given size.  '''
    for i in range(len(buckets)):
        s1, s2 = size
        b1, b2 = buckets[i]
        if s1 <= b1 and s2 <= b2:
            return i
    raise Exception('No bucket found for size: ' + str(size))


def _pad_vec(vec, size):
    '''  Pad vec to be the appropriate size  '''
    return vec + [config.PAD_ID] * (size - len(vec))


def _ids_and_mask_data(hl, txt, mask_size, enc_dict, dec_dict):
    '''  translate headlines and text to word_ids and create mask  '''
    txt_size, hl_size = mask_size
    hl_vec = _pad_vec([dec_dict['<s>']] +
                      [dec_dict.get(w, dec_dict['<unk>'])
                       for w in hl.split(' ')] + [dec_dict['<\s>']], hl_size)
    txt_vec = _pad_vec([enc_dict.get(w, enc_dict['<unk>'])
                        for w in txt.split(' ')], txt_size)
    txt_vec = list(reversed(txt_vec))
    mask = np.ones(hl_size)
    for i in range(len(hl_vec), hl_size):
        mask[i] = 0
    return np.array(hl_vec), np.array(txt_vec), np.array(mask)


def init_data_buckets(n):
    '''  init datasets  '''
    return {i: {'enc_input': [], 'dec_input': [], 'dec_masks': []}
            for i in range(n)}


def _bucketize_data(headlines, text, buckets, enc_dict, dec_dict):
    '''  sort data into appropriate bucket sizes  '''
    data_by_bucket = init_data_buckets(len(buckets))
    for i in range(len(headlines)):
        hl = headlines[i]
        txt = text[i]
        size = (len(txt.split()), len(hl.split()))
        bucket_index = _get_bucket(size, buckets)
        hl_vec, txt_vec, mask = _ids_and_mask_data(hl, txt,
                                                   buckets[bucket_index],
                                                   enc_dict, dec_dict)
        data_by_bucket[bucket_index]['enc_input'].append(txt_vec)
        data_by_bucket[bucket_index]['dec_input'].append(hl_vec)
        data_by_bucket[bucket_index]['dec_masks'].append(mask)

    return data_by_bucket


def load_one_set(data_path, name, buckets, enc_dict, dec_dict):
    '''  load one subset of the data (train/dev/test)  '''
    headlines = []
    text = []
    headline_path = os.path.join(data_path, name+'/headlines')
    text_path = os.path.join(data_path, name+'/texts')
    for filename in os.listdir(headline_path):
        headlines += _read_and_split_file(os.path.join(headline_path,
                                                       filename))
    for filename in os.listdir(text_path):
        text += _read_and_split_file(os.path.join(text_path, filename))

    return _bucketize_data(headlines, text, buckets, enc_dict, dec_dict)


def load_data(data_path, buckets):
    '''  read in data at data_path and sort them into appropriate buckets  '''
    enc_vocab = _read_and_split_file(os.path.join(data_path, 'enc_vocab.txt'))
    dec_vocab = _read_and_split_file(os.path.join(data_path, 'dec_vocab.txt'))
    enc_dict = {enc_vocab[i]: i for i in range(len(enc_vocab))}
    dec_dict = {dec_vocab[i]: i for i in range(len(dec_vocab))}
    train = load_one_set(data_path, 'train', buckets, enc_dict, dec_dict)
    dev = load_one_set(data_path, 'dev', buckets, enc_dict, dec_dict)
    test = load_one_set(data_path, 'test', buckets, enc_dict, dec_dict)
    num_samples = sum([len(train[i]['dec_input'])
                       for i in range(len(buckets))])
    dev_headlines_path = os.path.join(data_path, 'dev/headlines')
    test_headlines_path = os.path.join(data_path, 'test/headlines')
    return train, dev, test, enc_dict, dec_dict, num_samples, \
        dev_headlines_path, test_headlines_path


def make_dir(path):
    ''' Create a directory if there isn't one already. '''
    try:
        os.mkdir(path)
    except OSError:
        pass


def _reshape(inputs, size, batch_size):
    ''' Create batch-major inputs. Batch inputs are just re-indexed inputs  '''
    batch_inputs = np.array(inputs).T
    return batch_inputs


def get_batch(data_buckets, bucket_index, buckets, batch_size, iteration=0):
    '''  Get one batch of data from given bucket  '''
    bucket = data_buckets[bucket_index]
    next_bucket = False
    start_i = iteration * batch_size
    end_i = (iteration + 1) * batch_size
    if end_i >= len(bucket['enc_input']):
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
    '''  Get word_ids, mask, and bucket for text input  '''
    texts = []
    headlines = []
    masks = []
    for inp in inputs:
        txt_size = len(inp)
        bucket_index = _get_bucket((txt_size, 0), buckets)
        bucket = buckets[bucket_index]
        hl_vec, txt_vec, mask = _ids_and_mask_data('', inp, bucket,
                                                   enc_dict, dec_dict)
        texts.append(txt_vec)
        headlines.append(hl_vec)
        masks.append(mask)
    return bucket_index, (_reshape(texts, bucket[0], len(inputs)),
                          _reshape(headlines, bucket[1], len(inputs)),
                          _reshape(masks, bucket[1], len(inputs)))
