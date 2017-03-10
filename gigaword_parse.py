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
    return contents

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
    return text

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

def count_words(path, vocab):
    vocab[path] += 1

def build_vocab():
    vocab = collections.defaultdict(int)
    for i in range(3):
        path = '/datadrive/LDC2011T07_English-Gigaword-Fifth-Edition/disc%d/gigaword_eng_5_d%d/data/' % (i,i)
        for dirname,_,filenames in os.walk(path):
            for filename in filenames:
                count_words(dirname+'/'+filename, vocab)

start_time = time.time()
#test()
ostest()
print 'time %f' % (time.time() - start_time)
