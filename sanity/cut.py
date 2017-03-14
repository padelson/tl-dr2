import sys

if __name__ == '__main__':
    filepath = sys.argv[1]
    maxlen = int(sys.argv[2])
    with open(filepath) as f:
        newcontents = []
        for line in f:
            newcontents.append(line.split()[:maxlen])
            if len(newcontents[-1]) > maxlen:
                print 'fuck'
    with open(filepath, 'w') as f:
        for c in newcontents:
            f.write(' '.join(c)+'\n')

