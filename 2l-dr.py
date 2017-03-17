# import shit
# get flags and args
# initialize data
# initialize model
# train/summarize
import argparse

from model import Summarizer


def setup_argparse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--sum', action='store_true')
    parser.add_argument('--data_path')
    parser.add_argument('--input_path')
    parser.add_argument('--sess_name')
    parser.add_argument('--center_conv', action='store_true')
    parser.add_argument('--pretrained', action='store_true')
    parser.add_argument('--model', choices={'rnn', 'qrnn'})
    return parser

if __name__ == '__main__':
    parser = setup_argparse()
    args = parser.parse_args()
    if args.model == 'rnn' and args.pretrained:
        print 'Pretrained GloVe vectors only available for QRNN'
        exit(0)
    if args.train:
        summarizer = Summarizer(args.data_path, True, args.sess_name,
                                args.model, args.pretrained, args.center_conv)
        summarizer.train()
    if args.sum:
        summarizer = Summarizer(args.data_path, False, args.sess_name,
                                args.model, args.pretrained)
        summarizer.summarize(args.input_path)
