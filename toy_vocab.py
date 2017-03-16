import collections
import json

def get_vecs(vocab, num_to_keep):
    vecs_path = '/datadrive/glove/glove.6B.200d.txt'
    vecs = {}
    vocab_copy = list(vocab)
    with open(vecs_path) as f:
        for line in f:
            split = line.split()
            word = split[0]
            if word in vocab:
                vecs[word] = map(float, split[1:])
                vocab.remove(word)
            if len(vocab) == 0:
                break
        if (len(vocab) != 0):
            print 'didnt find vecs for', vocab
        result = [(vecs[w], w) for w in vocab_copy if w in vecs][:num_to_keep]
        return [x[0] for x in result], [x[1] for x in result]

print 'main'
n = 1000
vocab = collections.defaultdict(int)
with open('toy_texts.txt') as f:
    print 'coutning shit'
    for line in f:
        for w in line.split():
            vocab[w] += 1
with open('toy_headlines.txt') as f:
    for line in f:
        for w in line.split():
            vocab[w] += 1
dec_vocab = sorted(vocab.items(), key=lambda x: x[1], reverse=True)[:2000]
vecs, vocab = get_vecs(dec_vocab, n)
with open('embeddings.txt', 'w') as f:
    f.write(json.dumps(vecs))
with open('vocab.txt', 'w') as f:
    f.write('<pad>' + '\n')
    f.write('<unk>' + '\n')
    f.write('<s>' + '\n')
    f.write('<\s>' + '\n')
    for word in vocab:
        f.write(word[0]+'\n')
