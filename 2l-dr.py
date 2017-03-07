# import shit
# get flags and args
# initialize data
# initialize model
# train/summarize
import argparse

from model import Encoder, Decoder, Summarizer


def setup_argparse():
    parser = argparse.ArgumentParser()
    # TODO add arguments
    return parser

if __name__ == '__main__':
    parser = setup_argparse()
    args = parser.parse_args()
    encoder = Encoder()
    decoder = Decoder()
    if args.train:
        summarizer = Summarizer(encoder, decoder, args.data_path, True)
        summarizer.train()
    if args.sum:
        summarizer = Summarizer(encoder, decoder, args.data_path, False)
        summarizer.summarize()
