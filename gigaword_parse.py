# parse the gigaword files

import os
import re
import time

def make_dir(path):
    """ Create a directory if there isn't one already. """
    try:
        os.mkdir(path)
    except OSError:
        pass

parentpath = '/datadrive/LDC2011T07_English-Gigaword-Fifth-Edition/'


def grabContents(f, tag, exclusive=True):
    if (f.readline() != "<%s>\n" % tag):
        return ""
    contents = ""
    line = f.readline()
    while line != "</%s>\n" % tag:
        assert line[0] != '<' and line[-2] != '>'
        contents += line.replace('\n', ' ')
        line = f.readline()
    return contents

# assuming you have an open readable file...
# read the <TEXT> until </TEXT>
# throw Exception if the read is bad
def getFirstTwoSentences(f, line):
    while line != "<TEXT>\n":
        assert line != "</DOC>\n"
        line = f.readline()
    first = grabContents(f, 'P')
    second = grabContents(f, 'P')
    if second:
        contents = first + ' ' + second + '\n'
        line = f.readline()
        while line != "</TEXT>\n":
            assert line != "</DOC>\n"
            line = f.readline()
    else:
        contents = first
    return contents

def test():
    with open('example_data','r') as f:
        h = open('headlines.txt', 'w')
        t = open('text.txt', 'w')

        flag = False
        debug = 1
        while True:
            print debug
            line = f.readline()
            if not line:
                break

            # if a headline has been found, get corresponding text
            if flag:
                t.write(getFirstTwoSentences(f, line))
                flag = False
            else:
                headline = grabContents(f, "HEADLINE")
                if headline:
                    h.write(headline + '\n')
                    flag = True
            debug += 1

        h.close()
        t.close()

def ostest():
    for entry in os.listdir(parentpath):
        #if os.path.isfile(entry):
        print entry

start_time = time.time()
#test()
ostest()
print 'time %f' % (time.time() - start_time)
