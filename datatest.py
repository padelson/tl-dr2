from data import *
import config
import time
from utils import *

# path = './test_data'
# b = config.BUCKETS
# train, dev, test, enc_dict, dec_dict, num_samples = split_data(path, b)
# enc_matrix, dec_matrix, decoder_masks, next_bucket = get_batch(train, 0, b, 64)
# encoder_size, decoder_size = b[0]
#
# count = 0
# print decoder_size
# for step in xrange(decoder_size):
#     print step
#     x = decoder_masks[step]

num_batches = 8
target = 7

iteration = 0
for i in range(10):
    prog = Progbar(target=target)
    while True:
        time.sleep(0.5)
        step_iter = iteration % target
        if iteration > 0 and step_iter == 0:
            step_iter = target
        prog.update(step_iter, [('train loss', i**2)])
        if iteration > 0 and iteration % target == 0:
            iteration += 1
            break
        iteration += 1
    print
