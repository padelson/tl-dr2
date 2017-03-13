#import tensorflow as tf
import numpy as np

logits = np.array([[0.1,0.2,0.3],[0.2,0.1,0.6]])
def generate_random_logits(seq_len, vocab_size):
    logits = np.array([])
    for i in range(seq_len):
        distribution = np.random.uniform(size=vocab_size)
        normalization = sum(distribution)
        np.append(logits, distribution/normalization)
    return logits
logits = generate_random_logits(5, 10)
print logits
