# import shit
# get flags and args
# initialize data
# initialize model
# train/summarize
import argparse

from model import Encoder, Decoder, Summarizer


def setup_argparse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--sum', action='store_true')
    parser.add_argument('--data')
    parser.add_argument('--input_path')
    return parser

if __name__ == '__main__':
    parser = setup_argparse()
    args = parser.parse_args()
    print args
    encoder = Encoder()
    decoder = Decoder()
    if args.train:
        summarizer = Summarizer(encoder, decoder, args.data_path, True)
        summarizer.train()
    if args.sum:
        summarizer = Summarizer(encoder, decoder, args.data_path, False)
        summarizer.summarize(args.input_path)
