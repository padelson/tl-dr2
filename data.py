import os

from numpy import np


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


def _one_hot_and_mask_data(hl, txt, mask_size, enc_dict, dec_dict):
    hl_vec = [dec_dict.get(w, enc_dict['<unk']) for w in hl.split(' ')]
    txt_vec = [enc_dict.get(w, enc_dict['<unk']) for w in txt.split(' ')]
    txt_size, hl_size = mask_size
    mask = np.ones(hl_size)
    for i in range(len(hl_vec), hl_size):
        mask[i] = 0
    return hl_vec, txt_vec, mask


def init_data_buckets(n):
    return {i: {'enc_input': [], 'dec_input': [], 'dec_masks': []}
            for i in range(n)}


def _bucketize_and_split_data(headlines, text, buckets, enc_dict, dec_dict):
    train = init_data_buckets(len(buckets))
    dev = init_data_buckets(len(buckets))
    test = init_data_buckets(len(buckets))
    data_by_bucket = init_data_buckets(len(buckets))
    for i in range(len(headlines)):
        hl = headlines[i]
        txt = text[i]
        size = (len(txt), len(hl))
        bucket_index = _get_bucket(size, buckets)
        hl, txt, mask = _one_hot_and_mask_data(hl, txt, buckets[bucket_index],
                                               enc_dict, dec_dict)
        data_by_bucket[bucket_index]['enc_input'].append(txt)
        data_by_bucket[bucket_index]['dec_input'].append(hl)
        data_by_bucket[bucket_index]['dec_masks'].append(mask)

    for key in data_by_bucket:
        bucket = data_by_bucket[key]
        num_samples = len(bucket['enc_input'])
        split_data_indices = np.random.permutation(np.arange(num_samples))
        for i in range(len(split_data_indices)):
            data_index = split_data_indices[i]
            if i < num_samples * 9 / 10:
                train[key]['enc_input'].append(bucket['enc_input'][data_index])
                train[key]['dec_input'].append(bucket['dec_input'][data_index])
                train[key]['dec_masks'].append(bucket['dec_masks'][data_index])
            elif i % 2 == 0:
                dev[key]['enc_input'].append(bucket['enc_input'][data_index])
                dev[key]['dec_input'].append(bucket['dec_input'][data_index])
                dev[key]['dec_masks'].append(bucket['dec_masks'][data_index])
            else:
                test[key]['enc_input'].append(bucket['enc_input'][data_index])
                test[key]['dec_input'].append(bucket['dec_input'][data_index])
                test[key]['dec_masks'].append(bucket['dec_masks'][data_index])

    return train, dev, test


def split_data(data_path, buckets, enc_dict, dec_dict):
    headlines = _read_and_split_file(os.path.join(data_path, 'headlines.txt'))
    text = _read_and_split_file(os.path.join(data_path, 'text.txt'))
    enc_vocab = _read_and_split_file(os.path.join(data_path, 'enc_vocab.txt'))
    dec_vocab = _read_and_split_file(os.path.join(data_path, 'dec_vocab.txt'))
    num_samples = len(headlines)
    train, dev, test = _bucketize_and_split_data(headlines, text, buckets,
                                                 enc_dict, dec_dict)
    enc_vocab_dict = {enc_vocab[i]: i for i in range(len(enc_vocab))}
    dec_vocab_dict = {dec_vocab[i]: i for i in range(len(dec_vocab))}
    return train, dev, test, enc_vocab_dict, dec_vocab_dict, num_samples


def make_dir(path):
    """ Create a directory if there isn't one already. """
    try:
        os.mkdir(path)
    except OSError:
        pass


def get_batch():
    pass


def process_input(input):
    pass
