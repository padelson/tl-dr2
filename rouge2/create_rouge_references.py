import config
import data
import os


path = '/datadrive/gigaword_parsed/'
summary_path = os.path.join(path, 'summaries/reference')
data.make_dir(os.path.join(path, 'summaries'))
data.make_dir(summary_path)
data.make_dir(os.path.join(path, 'summaries/system'))

print 'loading data'
enc_vocab = data._read_and_split_file(os.path.join(path, 'enc_vocab.txt'))
dec_vocab = data._read_and_split_file(os.path.join(path, 'dec_vocab.txt'))
enc_dict = {enc_vocab[i]: i for i in range(len(enc_vocab))}
dec_dict = {dec_vocab[i]: i for i in range(len(dec_vocab))}
# load these properly as text
dev = data.load_one_set(path, 'dev', config.BUCKETS,
                        enc_dict, dec_dict, vec=False)
test = data.load_one_set(path, 'test', config.BUCKETS,
                         enc_dict, dec_dict, vec=False)

count = 0
iteration = 0
bucket_index = 0
while True:
    bucket = config.BUCKETS[bucket_index]
    next_bucket = False
    start_i = iteration * config.BATCH_SIZE
    end_i = (iteration + 1) * config.BATCH_SIZE
    if end_i >= len(bucket['enc_input']):
        next_bucket = True
        start_i = -config.BATCH_SIZE
        end_i = None
        dec_input = bucket['dec_input'][start_i:]
    else:
        dec_input = bucket['dec_input'][start_i:end_i]
    for headline in dec_input:
        with open(os.path.join(path, str(count)+'_gigaword.txt', 'w')) as f:
            f.write(headline)
            if count % 500 == 0:
                print 'writing #', count
        count += 1
    if next_bucket:
        bucket_index += 1
        iteration = 0
    else:
        iteration += 1
    if bucket_index >= len(config.BUCKETS):
        break
