'''Main function for 2l-dr: headline generation (take 2)
Executes the Summarizer class, a deep learning sequence-to-sequence model

Usage:
    --train: train the model on data at data_path
    --sum: generate a summary for the input found at input_path
    --model: (rnn or qrnn) deep learning model to use
    --data_path: path to data
    --input_path: path to summary input
    --sess_name: name of executing session
    --center_conv: use a centered convolution in qrnn instead of masked
    --pretrained: use pretrained GloVe vectors
'''
import argparse

from model import Summarizer


def setup_argparse():
    '''Set up argument parser'''
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
    if args.train:
        summarizer = Summarizer(args.data_path, True, args.sess_name,
                                args.model, args.pretrained, args.center_conv)
        summarizer.train()
    if args.sum:
        summarizer = Summarizer(args.data_path, False, args.sess_name,
                                args.model, args.pretrained)
        with open(args.input_path) as f:
            inputs = f.read()
        print summarizer.summarize(inputs)
