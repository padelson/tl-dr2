# parse the gigaword files
import json
import os
import re
import time
import collections
from string import punctuation

def make_dir(path):
    """ Create a directory if there isn't one already. """
    try:
        os.mkdir(path)
    except OSError:
        pass

def makeDirs():
    make_dir('/datadrive/gigaword_parsed')
    make_dir('/datadrive/gigaword_parsed/train')
    make_dir('/datadrive/gigaword_parsed/dev')
    make_dir('/datadrive/gigaword_parsed/test')
    make_dir('/datadrive/gigaword_parsed/train/headlines')
    make_dir('/datadrive/gigaword_parsed/dev/headlines')
    make_dir('/datadrive/gigaword_parsed/test/headlines')
    make_dir('/datadrive/gigaword_parsed/train/texts')
    make_dir('/datadrive/gigaword_parsed/dev/texts')
    make_dir('/datadrive/gigaword_parsed/test/texts')

def readline(f):
    line = f.readline()
    if line == "</DOC>\n":
        raise Exception("end of document")
    return line

def grabContents(f, tag):
    line = readline(f)
    # check for correct tag (return None if incorrect)
    if (line != "<%s>\n" % tag):
        return None
    contents = ""
    line = readline(f)
    while line != "</%s>\n" % tag:
        contents += line.replace('\n', ' ')
        line = readline(f)
    return contents.split()

# assuming you have an open readable file...
# read the <TEXT> until </TEXT>
# throw Exception if the read is bad
def getFirstSentence(f):
    text = grabContents(f, "P")
    while text is None:
        text = grabContents(f, "P")
    # second = grabContents(f, "P")
    # if second is not None:
    #     text += ' ' + second
    return text

def getHeadline(f):
    headline = grabContents(f, "HEADLINE")
    while headline is None:
        headline = grabContents(f, "HEADLINE")
    return headline.split(' ')

def process(dirname=".", filename="example_data", file_num=0):
    f = open(dirname + '/' + filename, 'r')
    if file_num % 20 == 0:
        h = open('/datadrive/gigaword_parsed/dev/headlines/' + filename, 'w')
        t = open('/datadrive/gigaword_parsed/dev/texts/' + filename, 'w')
    elif file_num % 20 == 1:
        h = open('/datadrive/gigaword_parsed/test/headlines/' + filename, 'w')
        t = open('/datadrive/gigaword_parsed/test/texts/' + filename, 'w')
    else:
        h = open('/datadrive/gigaword_parsed/train/headlines/' + filename, 'w')
        t = open('/datadrive/gigaword_parsed/train/texts/' + filename, 'w')
    #h = open('headlines.txt', 'w')
    #t = open('text.txt', 'w')

    count = 0
    while True:
        line = f.readline()
        if not line:
            break

        # get to the beginning of document
        if line.find('<DOC', 0, 4) == 0:
            docline = line
            try:
                # get the headline (if no headline is found, Exception is thrown)
                headline = grabContents(f, "HEADLINE")
                while headline is None:
                    headline = grabContents(f, "HEADLINE")
                text = getFirstSentence(f)
                if len(headline) <= 25 and len(text) <= 50:
                    h.write((' '.join(headline)+'\n').lower())
                    t.write((' '.join(text)+'\n').lower())
            except Exception:
                #print docline
                count += 1

    print count
    h.close()
    t.close()
    f.close()

def ostest():
    print 'ostest'
    file_num = 0
    for i in range(1,4):
        path = '/datadrive/LDC2011T07_English-Gigaword-Fifth-Edition/disc%d/gigaword_eng_5_d%d/data/' % (i,i)
        print path
        for dirname,_,filenames in os.walk(path):
            for filename in filenames:
                process(dirname, filename, file_num)
                file_num += 1

def count_words(vocab, headline, text):
    # headline = headline.translate(None, punctuation+'\n').split()
    # text = text.translate(None, punctuation+'\n').split()
    headline = headline.split()
    text = text.split()
    for word in headline:
        vocab[word] += 1
    for word in text:
        vocab[word] += 1

def write(f, d):
    arr = sorted(d.items(), key = lambda x: x[0])
    for i, elem in enumerate(arr):
        #f.write('%d,%d\t\t' % (elem[0][0], elem[0][1]))
        f.write('%s\t' % elem[0])
        f.write('%d\n' % elem[1])

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

def build_vocab():
    print 'build_vocab'
    vocab = collections.defaultdict(int)
    dist = collections.defaultdict(int)
    enc = open('/datadrive/gigaword_parsed/enc_vocab.txt', 'w')
    dec = open('/datadrive/gigaword_parsed/dec_vocab.txt', 'w')
    out = open('output.txt', 'w')
    enc.write('<pad>\n<unk>\n<s>\n<\s>\n')
    dec.write('<pad>\n<unk>\n<s>\n<\s>\n')

    directories = ('train', 'dev', 'test')
    for d in directories:
        path = '/datadrive/gigaword_parsed/%s/headlines/' % d
        print path
        for filename in os.listdir(path):
            h = open(path+filename, 'r')
            t = open('/datadrive/gigaword_parsed/%s/texts/%s' % (d, filename), 'r')

            for headline in h:
                text = t.readline()
                dist[bucketize(headline, text)] += 1
                count_words(vocab, headline, text)

            h.close()
            t.close()

    # count_words(None, vocab, dist)
    top20000 = sorted(vocab.items(), key=lambda x: x[1], reverse=True)[:20000]
    if type(top20000[0]) != str:
        top20000 = [x[0] for x in top20000]
    vecs, top10000 = get_vecs(top20000, 10000)
    with open('embeddings.txt', 'w') as embeddings_f:
        embeddings_f.write(json.dumps(vecs))
    for words in top10000:
        enc.write(entry[0] + '\n')
        dec.write(entry[0] + '\n')
    write(out, dist)
    out.close()
    enc.close()
    dec.close()

def bucketize(headline, text):
    hl_len = len(headline.split())
    t_len = len(text.split())
    if t_len < 15:
        return 'b1'
    elif t_len < 30:
        return 'b2'
    else:
        return 'b3'

def find_dist():
    dist = collections.defaultdict(int)
    out = open('output.txt', 'w')

    # train, dev, test
    directories = ('train', 'dev', 'test')
    for d in directories:
        path = '/datadrive/gigaword_parsed/%s/headlines/' % d
        print path
        for filename in os.listdir(path):
            h = open(path+filename, 'r')
            t = open('/datadrive/gigaword_parsed/%s/texts/%s' % (d, filename), 'r')

            for headline in h:
                text = t.readline()
                dist[bucketize(headline, text)] += 1

            h.close()
            t.close()
    write(out, dist)
    out.close()

    """
    for i in range(1,4):
        path = '/datadrive/LDC2011T07_English-Gigaword-Fifth-Edition/disc%d/gigaword_eng_5_d%d/data/' % (i,i)
        for _,_,filenames in os.walk(path):
            for filename in filenames:
                h = open('/datadrive/gigaword_parsed/headlines/'+filename, 'r')
                t = open('/datadrive/gigaword_parsed/texts/'+filename, 'r')

                for headline in h:
                    text = t.readline()
                    #print len(headline.split()), len(text.split())
                    dist[bucketize(headline, text)] += 1

                h.close()
                t.close()
        write(out, dist)
        out.close()
        assert False
    """

start_time = time.time()
#test()
makeDirs()
ostest()
build_vocab()
#find_dist()
#ostest()
print 'time %f' % (time.time() - start_time)
