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
import words


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


def get_perplexity(loss):
    return 2**loss


def go(options):
    tbw = SummaryWriter(log_dir=options.tb_dir)

    if options.seed < 0:
        seed = random.randint(0, 1000000)
        print('random seed: ', seed)
        np.random.seed(seed)
    else:
        np.random.seed(options.seed)

    if options.task == 'wikisimple':
        print("wikisimple task")
        data = \
            util.load_words_split_types(util.DIR + '/datasets/wikisimple.txt', vocab_size=options.top_words,
                                        limit=options.limit)

    elif options.task == 'file':
        print("file task")
        data = \
            util.load_words_split_types(options.data_dir, vocab_size=options.top_words, limit=options.limit)

    else:
        raise Exception('Task {} not recognized.'.format(options.task))

    train = data["train"]  # [data, w2i, i2w]
    valid = data["validation"]  # [data, w2i, i2w]
    test = data["test"]  # [data, w2i, i2w]
    x_train, w2i_train, i2w_train = train[0], train[1], train[2]
    x_valid, w2i_valid, i2w_valid = valid[0], valid[1], valid[2]
    x_test, w2i_test, i2w_test = test[0], test[1], test[2]

    # Finding the length of the longest sequence
    x_max_len_train = max([len(sentence) for sentence in x_train])
    x_max_len_valid = max([len(sentence) for sentence in x_valid])
    x_max_len_test = max([len(sentence) for sentence in x_test])

    numwords_train = len(i2w_train)
    numwords_valid = len(i2w_valid)
    numwords_test = len(i2w_test)

    print('max sequence length - train', x_max_len_train)
    print(numwords_train, 'distinct words - train')

    print('max sequence length - valid', x_max_len_valid)
    print(numwords_valid, 'distinct words - valid')

    print('max sequence length - test', x_max_len_test)
    print(numwords_test, 'distinct words - test')

    x_train = util.batch_pad(x_train, options.batch, add_eos=True)
    x_valid = util.batch_pad(x_valid, options.batch, add_eos=True)
    x_test = util.batch_pad(x_test, options.batch, add_eos=True)

    def decode_train(seq):
        return ' '.join(i2w_train[id] for id in seq)

    def decode_valid(seq):
        return ' '.join(i2w_valid[id] for id in seq)

    def decode_test(seq):
        return ' '.join(i2w_test[id] for id in seq)

    print('Finished data loading. ', sum([b.shape[0] for b in x_train]), ' train sentences loaded')
    print('Finished data loading. ', sum([b.shape[0] for b in x_valid]), ' valid sentences loaded')
    print('Finished data loading. ', sum([b.shape[0] for b in x_test]), ' test sentences loaded')

    ## Define model

    input_train = Input(shape=(None,))

    embedding_train = Embedding(numwords_train, options.lstm_capacity, input_length=None)

    embedded_train = embedding_train(input_train)

    decoder_lstm_train = LSTM(options.lstm_capacity, return_sequences=True)

    h_train = decoder_lstm_train(embedded_train)

    if options.extra is not None:
        for _ in range(options.extra):
            h_train = LSTM(options.lstm_capacity, return_sequences=True)(h_train)

    fromhidden_train = Dense(numwords_train, activation='linear')

    out_train = TimeDistributed(fromhidden_train)(h_train)

    model_train = Model(input_train, out_train)

    opt = keras.optimizers.Adam(lr=options.lr)
    lss = words.sparse_loss

    model_train.compile(opt, lss)

    model_train.summary()

    ## Training

    # - Since we have a variable batch size, we make our own training loop, and train with
    #  model.train_on_batch(...). It's a little more verbose, but it gives us more control.

    epoch = 0
    instances_seen = 0

    while epoch < options.epochs:

        for batch_train in tqdm(x_train):
            n_train, l_train = batch_train.shape

            # print(f"n = {n} batch_train {batch_train}")
            batch_shifted_train = np.concatenate([np.ones((n_train, 1)), batch_train], axis=1)  # prepend start symbol
            batch_out_train = np.concatenate([batch_train, np.zeros((n_train, 1))], axis=1)  # append pad symbol

            loss_train = model_train.train_on_batch(batch_shifted_train, batch_out_train[:, :, None])

            instances_seen += n_train
            # tbw.add_scalar('lm/batch-loss', float(loss), instances_seen)
        loss_train, perplexity_train = get_loss(model_train, x_train)
        loss_valid, perplexity_valid = get_loss(model_train, x_valid)
        loss_test, perplexity_test = get_loss(model_train, x_test)

        epoch += 1

        print(f'Train - Loss = {loss_train}, Perplexity = {perplexity_train}')
        print(f'Valid - Loss = {loss_valid}, Perplexity = {perplexity_valid}')
        print(f'Test - Loss = {loss_test}, Perplexity = {perplexity_test}')

        # Show samples for some sentences from random batches
        for temp in [0.0, 0.9, 1, 1.1, 1.2]:
            print('### TEMP ', temp)
            for i in range(words.CHECK):
                b = random.choice(x_test)

                if b.shape[1] > 20:
                    seed = b[0, :20]
                else:
                    seed = b[0, :]

                seed = np.insert(seed, 0, 1)
                gen = words.generate_seq(model_train, seed, 60, temperature=temp)

                print('*** [', decode_train(seed), '] ', decode_train(gen[len(seed):]))


def get_loss(model, x):
    loss = 0.0
    n = 0
    print("Calculating loss")
    for batch_train in x:
        n_train, l_train = batch_train.shape
        batch_shifted_train = np.concatenate([np.ones((n_train, 1)), batch_train], axis=1)  # prepend start symbol
        batch_out_train = np.concatenate([batch_train, np.zeros((n_train, 1))], axis=1)  # append pad symbol
        loss_train = model.test_on_batch(batch_shifted_train, batch_out_train[:, :, None])

        loss += loss_train
        n += 1
    loss = (loss / n)
    perplexity = get_perplexity(loss)
    return loss, perplexity


def main():
    parser = create_parser()
    options = parser.parse_args()
    go(options)


if __name__ == "__main__":
    main()
