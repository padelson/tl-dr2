# parse the gigaword files

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
    return ' '.join(contents.split(' ')[:35])

# assuming you have an open readable file...
# read the <TEXT> until </TEXT>
# throw Exception if the read is bad
def getFirstTwoSentences(f):
    text = grabContents(f, "P")
    while text is None:
        text = grabContents(f, "P")
    second = grabContents(f, "P")
    if second is not None:
        text += ' ' + second
    return ' '.join(text.split(' ')[:200])

def process(dirname=".", filename="example_data"):
    f = open(dirname + '/' + filename, 'r')
    h = open('/datadrive/gigaword_parsed/headlines/' + filename, 'w')
    t = open('/datadrive/gigaword_parsed/texts/' + filename, 'w')
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
                text = getFirstTwoSentences(f)
                h.write(headline+'\n')
                t.write(text+'\n')
            except Exception:
                #print docline
                count += 1

    print count
    h.close()
    t.close()
    f.close()

def ostest():
    for i in range(3):
        path = '/datadrive/LDC2011T07_English-Gigaword-Fifth-Edition/disc%d/gigaword_eng_5_d%d/data/' % (i,i)
        for dirname,_,filenames in os.walk(path):
            for filename in filenames:
                process(dirname, filename)

def count_words(filename, vocab, dist):
    h = open('/datadrive/gigaword_parsed/headlines/' + filename, 'r')
    t = open('/datadrive/gigaword_parsed/texts/' + filename, 'r')

    #h = open('test_data/headlines2.txt', 'r')
    #t = open('test_data/text2.txt', 'r')

    for headline in h:
        headline = headline.translate(None, punctuation+'\n').split(' ')
        text = t.readline().translate(None, punctuation+'\n').split(' ')

        dist[(len(headline)/5, len(text)/10)] += 1

        for i, word in enumerate(headline):
            vocab[word] += 1
        for i, word in enumerate(text):
            vocab[word] += 1

    h.close()
    t.close()

def write(f, d):
    arr = sorted(d.items(), key = lambda x: x[0])
    for i, elem in enumerate(arr):
        f.write('%d,%d\t\t' % ((elem[0][0]+1) * 5, (elem[0][1]+1) * 10))
        f.write('%d\n' % elem[1])

def build_vocab():
    vocab = collections.defaultdict(int)
    dist = collections.defaultdict(int)
    enc = open('/datadrive/gigaword_parsed/enc_vocab.txt', 'w')
    dec = open('/datadrive/gigaword_parsed/dec_vocab.txt', 'w')
    out = open('output.txt', 'w')
    enc.write('<pad>\n<unk>\n<s>\n<\s>\n')
    dec.write('<pad>\n<unk>\n<s>\n<\s>\n')

    for i in range(3):
        path = '/datadrive/LDC2011T07_English-Gigaword-Fifth-Edition/disc%d/gigaword_eng_5_d%d/data/' % (i,i)
        for _,_,filenames in os.walk(path):
            for filename in filenames:
                count_words(filename, vocab, dist)
    #count_words(None, vocab, dist)
    top10000 = sorted(vocab.items(), key=lambda x: x[1], reverse=True)[:10000]
    for entry in top10000:
        enc.write(entry[0] + '\n')
        dec.write(entry[0] + '\n')
    write(out, dist)
    out.close()
    enc.close()
    dec.close()

start_time = time.time()
#test()
ostest()
build_vocab()
#ostest()
print 'time %f' % (time.time() - start_time)
