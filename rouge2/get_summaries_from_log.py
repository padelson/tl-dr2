import os
import sys

summary_path = '/datadrive/gigaword_parsed/summaries/generated'


def sum_file_name(count, iter_name, sum_name):
    sys_name = sum_name + '_' + iter_name[5:]
    filename = str(count) + '_' + sys_name.replace('_', '-')
    return os.path.join(filename)


def write_summaries(results_path, iter_name, sum_name):
    count = 0
    path = os.path.join(os.path.join(results_path, iter_name), 'log')
    with open(path) as f:
        text = f.read().split('\n')[10:]
        headlines = [t for t in text if text != '']
        for h in headlines:
            if count % 10000 == 0:
                print 'writing #', count
            with open(sum_file_name(count, iter_name, sum_name), 'w') as f:
                f.write(h[20:])
            count += 1


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print 'provide path to results directory and name of summary'
    results_path = sys.argv[1]
    iter_results = os.listdir(results_path)
    for iter_name in reversed(iter_results[-2:]):
        write_summaries(results_path, iter_name, sys.argv[2])
