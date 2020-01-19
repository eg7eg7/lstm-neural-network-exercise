import keras

import keras.backend as K
from keras.datasets import imdb
from keras.layers import LSTM, Embedding, TimeDistributed, Input, Dense
from keras.models import Model
from tensorflow.python.client import device_lib

from tqdm import tqdm
import os, random

from argparse import ArgumentParser

import numpy as np

from tensorboardX import SummaryWriter

import util
def create_parser():
    ## Parse the command line options
    parser = ArgumentParser()

    parser.add_argument("-e", "--epochs",
                        dest="epochs",
                        help="Number of epochs.",
                        default=20, type=int)

    parser.add_argument("-E", "--embedding-size",
                        dest="embedding_size",
                        help="Size of the word embeddings on the input layer.",
                        default=300, type=int)

    parser.add_argument("-o", "--output-every",
                        dest="out_every",
                        help="Output every n epochs.",
                        default=1, type=int)

    parser.add_argument("-l", "--learn-rate",
                        dest="lr",
                        help="Learning rate",
                        default=0.001, type=float)

    parser.add_argument("-b", "--batch-size",
                        dest="batch",
                        help="Batch size",
                        default=128, type=int)

    parser.add_argument("-t", "--task",
                        dest="task",
                        help="Task",
                        default='wikisimple', type=str)

    parser.add_argument("-D", "--data-directory",
                        dest="data",
                        help="Data file. Should contain one sentence per line.",
                        default='./data', type=str)

    parser.add_argument("-L", "--lstm-hidden-size",
                        dest="lstm_capacity",
                        help="LSTM capacity",
                        default=256, type=int)

    parser.add_argument("-m", "--max_length",
                        dest="max_length",
                        help="Max length",
                        default=None, type=int)

    parser.add_argument("-w", "--top_words",
                        dest="top_words",
                        help="Top words",
                        default=10000, type=int)

    parser.add_argument("-I", "--limit",
                        dest="limit",
                        help="Character cap for the corpus",
                        default=None, type=int)

    parser.add_argument("-T", "--tb-directory",
                        dest="tb_dir",
                        help="Tensorboard directory",
                        default='./runs/words', type=str)

    parser.add_argument("-r", "--random-seed",
                        dest="seed",
                        help="RNG seed. Negative for random (seed is printed for reproducability).",
                        default=-1, type=int)

    parser.add_argument("-x", "--extra-layers",
                        dest="extra",
                        help="Number of extra LSTM layers.",
                        default=None, type=int)
    return parser


def go(options):
    tbw = SummaryWriter(log_dir=options.tb_dir)

    if options.seed < 0:
        seed = random.randint(0, 1000000)
        print('random seed: ', seed)
        np.random.seed(seed)
    else:
        np.random.seed(options.seed)


    split_data = util.load_words_split_types(util.DIR + '/datasets/wikisimple.txt', vocab_size=options.top_words, limit=options.limit)
    train = split_data["train"]  # [data, w2i, i2w]
    valid = split_data["validation"]  # [data, w2i, i2w]
    test = split_data["test"]  # [data, w2i, i2w]


def main():
    parser = create_parser()
    options = parser.parse_args()
    go(options)




if __name__ == "__main__":
    main()


