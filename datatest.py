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


prog = Progbar(target=10)
for i in range(5):
    time.sleep(2)
    prog.update(i+1, [('train loss', i**2)])
prog.update(0)
