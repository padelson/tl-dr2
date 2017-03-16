import json
import sys


def load_and_split_file(path):
    with open(path) as f:
        result = f.read.split('\n')
        if result[-1] == '':
            return result[:-1]
        return result


def get_glove_vecs(vocab, glove_file):
    vecs = {}
    vocab_copy = list(vocab)
    with open(glove_file) as f:
        for line in f:
            split = line.split()
            word = split[0]
            if word in vocab:
                vecs[word] = map(float, split[1:])
                vocab.remove(word)
            if len(vocab) == 0:
                break
        result = [vecs[w] for w in vocab_copy]
        return result


def write_results(vecs, size, vocab_name):
    with open(vocab_name+str(size)+'.glove', 'w') as f:
        f.write(json.dumps(vecs))


if __name__ == '__main__':
    if len(sys.argv) != 1:
        print 'provide glove vec dims you want'
        sys.exit(0)
    enc_vocab = load_and_split_file('enc_vocab.txt')
    glove_path = '/datadrive/glove/'
