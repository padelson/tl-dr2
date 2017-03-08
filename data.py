import os


def _read_and_split_file(path):
    with open(path) as f:
        return f.read().split('\n')


def _bucketize_and_split_data(headlines, text, buckets):
    train, dev, test = {}
    # TODO bucketize and mask
    # TODO split into train, dev, test
    return train, dev, test


def split_data(data_path, buckets):
    headlines = _read_and_split_file(os.path.join(data_path, 'headlines.txt'))
    text = _read_and_split_file(os.path.join(data_path, 'text.txt'))
    enc_vocab = _read_and_split_file(os.path.join(data_path, 'enc_vocab.txt'))
    dec_vocab = _read_and_split_file(os.path.join(data_path, 'dec_vocab.txt'))
    num_samples = len(headlines)
    train, dev, test = _bucketize_and_split_data(headlines, text, buckets)
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
