import collections

if __name__ == '__main__':
    n = 500
    vocab = collections.defaultdict(int)
    with open('text2.txt') as f:
        for line in f:
            for w in line.split():
                vocab[w] += 1
    with open('enc_vocab.txt', 'w') as f:
        enc_vocab = sorted(vocab.items(), key=lambda x: x[1], reverse=True)[:n]
        for word in enc_vocab:
            f.write(word[0]+'\n')
        f.write('<unk>')
    with open('headlines2.txt') as f:
        for line in f:
            for w in line.split():
                vocab[w] += 1
    with open('dec_vocab.txt', 'w') as f:
        dec_vocab = sorted(vocab.items(), key=lambda x: x[1], reverse=True)[:n]
        for word in dec_vocab:
            f.write(word[0]+'\n')
        f.write('<unk>')
