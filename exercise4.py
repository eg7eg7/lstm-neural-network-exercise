import keras

import keras.backend as K
from keras.datasets import imdb
from keras.layers import LSTM, Embedding, TimeDistributed, Input, Dense
from keras.models import Model, load_model
import logging
from tensorflow.python.client import device_lib

from tqdm import tqdm
import os, random

from argparse import ArgumentParser

import numpy as np

from tensorboardX import SummaryWriter
import math
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


class LSTM_words:
    def __init__(self, task, top_words=1000, limit=None, seed=-1, batch_size=128, data_dir=None, learning_rate=0.001,
                 lstm_capacity=256, lstm_extra=None, epochs=100, temperature=1.0):
        if seed < 0:
            seed = random.randint(0, 1000000)
            print('random seed: ', seed)
        np.random.seed(seed)
        self.input_train = None
        self.out_train = None
        self.opt = None
        self.lss = None
        self.embedding_train = None
        self.embedded_train = None
        self.decoder_lstm_train = None
        self.fromhidden_train = None
        self.h_train = None
        self.temperature = temperature
        self.top_words = top_words
        self.check = 6
        self.data = None
        self.model = None
        self.data_train = None
        self.data_valid = None
        self.data_test = None
        self.epochs = epochs
        self.lstm_capacity = lstm_capacity
        self.lstm_extra = lstm_extra
        self.batch_size = batch_size
        self.numwords_train = 0
        self.numwords_test = 0
        self.numwords_valid = 0
        self.learning_rate = learning_rate
        self.x_train, self.w2i_train, self.i2w_train = None, None, None
        self.x_valid, self.w2i_valid, self.i2w_valid = None, None, None
        self.x_test, self.w2i_test, self.i2w_test = None, None, None
        self.loss_train, self.perplexity_train = 0, 0
        self.loss_valid, self.perplexity_valid = 0, 0
        self.loss_test, self.perplexity_test = 0, 0
        self.data_dir = data_dir
        self.limit = limit
        self.task = task
        self.is_trained = False
        self.is_initialized = False

    def initialize(self):
        self.load_data()
        self.define_model()
        self.is_trained = False
        self.is_initialized = True
        return self

    def get_losses(self):
        return self.loss_train, self.loss_valid, self.loss_test

    def train(self):
        if not self.is_initialized:
            print(f'model is not initiated, model.initialize() before using train')
            return
        ## Training

        # - Since we have a variable batch size, we make our own training loop, and train with
        #  model.train_on_batch(...). It's a little more verbose, but it gives us more control.

        epoch = 0
        instances_seen = 0

        while epoch < self.epochs:

            for batch_train in self.x_train:
                n_train, l_train = batch_train.shape
                print("epoch", epoch)
                batch_shifted_train = np.concatenate([np.ones((n_train, 1)), batch_train],
                                                     axis=1)  # prepend start symbol
                batch_out_train = np.concatenate([batch_train, np.zeros((n_train, 1))], axis=1)  # append pad symbol

                self.model.train_on_batch(batch_shifted_train, batch_out_train[:, :, None])

                instances_seen += n_train
            epoch += 1
        print("Calculating loss for training set")
        self.loss_train, self.perplexity_train = LSTM_words.get_loss(self.model, self.x_train)
        print("Calculating loss for validation set")
        self.loss_valid, self.perplexity_valid = LSTM_words.get_loss(self.model, self.x_valid)
        print("Calculating loss for test set")
        self.loss_test, self.perplexity_test = LSTM_words.get_loss(self.model, self.x_test)
        self.is_trained = True

    def save_model(self, model, filepath):
        model.save(filepath + ".hdf5")

    def load_model(self, filepath):
        model = load_model(filepath)
        
    def print_info(self):
        if self.is_initialized:
            self.model.summary()
        else:
            print(f'LSTM is not initialized - initialize first')
        if self.is_trained:
            print(f'Train - Loss = {self.loss_train}, Perplexity = {self.perplexity_train}')
            print(f'Valid - Loss = {self.loss_valid}, Perplexity = {self.perplexity_valid}')
            print(f'Test - Loss = {self.loss_test}, Perplexity = {self.perplexity_test}')
        else:
            print(f'Model is not trained - no data to show')

    def get_sentence_probability(self, sentence):
        if self.is_initialized:
            return self.model.predict(sentence)
        else:
            print("No model error")
            return 0

    def generate_sentences(self, sentence_beginning="I love you", temperatures=[0.1, 1, 10]):
        sentence_to_feed, generated_sentences = [], []
        for word in sentence_beginning.split():
            sentence_to_feed.append(self.w2i_test[word])
        for temperature in temperatures:
            generated_sentences.append(words.generate_seq(self.model, sentence_to_feed, 7, temperature))

    def decode_train(self, seq):
        return ' '.join(self.i2w_train[id] for id in seq)

    def decode_valid(self, seq):
        return ' '.join(self.i2w_valid[id] for id in seq)

    def decode_test(self, seq):
        return ' '.join(self.i2w_test[id] for id in seq)

    def define_model(self):
        ## Define model

        self.input_train = Input(shape=(None,))

        self.embedding_train = Embedding(self.numwords_train, self.lstm_capacity, input_length=None)

        self.embedded_train = self.embedding_train(self.input_train)

        self.decoder_lstm_train = LSTM(self.lstm_capacity, return_sequences=True)

        self.h_train = self.decoder_lstm_train(self.embedded_train)

        if self.lstm_extra is not None:
            for _ in range(self.lstm_extra):
                self.h_train = LSTM(self.lstm_capacity, return_sequences=True)(self.h_train)

        self.fromhidden_train = Dense(self.numwords_train, activation='linear')

        self.out_train = TimeDistributed(self.fromhidden_train)(self.h_train)

        self.model = Model(self.input_train, self.out_train)
        self.opt = keras.optimizers.Adam(lr=self.learning_rate)
        self.lss = words.sparse_loss
        self.model.compile(self.opt, self.lss)
        self.model.summary()

    def load_data(self):
        if self.task == 'wikisimple':
            print("wikisimple task")
            self.data = \
                util.load_words_split_types(util.DIR + '/datasets/wikisimple.txt', vocab_size=self.top_words,
                                            limit=self.limit)

        elif self.task == 'file' and self.data_dir is not None:
            print("file task")
            self.data = \
                util.load_words_split_types(self.data_dir, vocab_size=self.top_words, limit=self.limit)

        else:
            raise Exception('Task {} not recognized.'.format(self.task))

        self.data_train, self.data_valid, self.data_test = self.data["train"], self.data["validation"], self.data["test"]
        self.x_train, self.w2i_train, self.i2w_train = self.data_train[0], self.data_train[1], self.data_train[2]
        self.x_valid, self.w2i_valid, self.i2w_valid = self.data_valid[0], self.data_valid[1], self.data_valid[2]
        self.x_test, self.w2i_test, self.i2w_test = self.data_test[0], self.data_test[1], self.data_test[2]
        # Finding the length of the longest sequence
        x_max_len_train = max([len(sentence) for sentence in self.x_train])
        x_max_len_valid = max([len(sentence) for sentence in self.x_valid])
        x_max_len_test = max([len(sentence) for sentence in self.x_test])

        numwords_train = len(self.i2w_train)
        numwords_valid = len(self.i2w_valid)
        numwords_test = len(self.i2w_test)

        print('max sequence length - train', x_max_len_train)
        print(numwords_train, 'distinct words - train')

        print('max sequence length - valid', x_max_len_valid)
        print(numwords_valid, 'distinct words - valid')

        print('max sequence length - test', x_max_len_test)
        print(numwords_test, 'distinct words - test')

        self.x_train = util.batch_pad(self.x_train, self.batch_size, add_eos=True)
        self.x_valid = util.batch_pad(self.x_valid, self.batch_size, add_eos=True)
        self.x_test = util.batch_pad(self.x_test, self.batch_size, add_eos=True)

        print('Finished data loading. ', sum([b.shape[0] for b in self.x_train]), ' training sentences loaded')
        print('Finished data loading. ', sum([b.shape[0] for b in self.x_valid]), ' validation sentences loaded')
        print('Finished data loading. ', sum([b.shape[0] for b in self.x_valid]), ' test sentences loaded')

    @staticmethod
    def get_loss(model, x):
        loss = 0.0
        n = 0

        for batch_train in tqdm(x):
            n_train, l_train = batch_train.shape
            batch_shifted_train = np.concatenate([np.ones((n_train, 1)), batch_train], axis=1)  # prepend start symbol
            batch_out_train = np.concatenate([batch_train, np.zeros((n_train, 1))], axis=1)  # append pad symbol
            loss_train = model.test_on_batch(batch_shifted_train, batch_out_train[:, :, None])

            loss += loss_train
            n += 1
        loss = (loss / n)
        perplexity = LSTM_words.get_perplexity(loss)
        return loss, perplexity

    @staticmethod
    def get_perplexity(loss):
        return math.exp(loss)

    def generate_words(self, size=60, temp=None):
        if temp is None:
            temp = self.temperature
        for i in range(self.check):
            b = random.choice(self.x_test)

            if b.shape[1] > 20:
                seed = b[0, :20]
            else:
                seed = b[0, :]

            seed = np.insert(seed, 0, 1)
            gen = words.generate_seq(self.model, seed, size, temperature=temp)

            return '*** [', self.decode_train(seed), '] ', self.decode_train(gen[len(seed):])
    
    def set_temperature(self, temp):
        self.temperature = temp

    def get_temperature(self):
        return self.temperature

    def set_num_words_to_check(self, num):
        self.check = num


def main():
    models = []
    parser = create_parser()
    options = parser.parse_args()
    model1 = LSTM_words(task=options.task, top_words=options.top_words, limit=options.limit, seed=options.seed,
                        batch_size=options.batch, learning_rate=options.lr, lstm_capacity=options.lstm_capacity,
                        lstm_extra=options.extra, epochs=options.epochs)
    model1.initialize()
    # model1.train()
    models.append(model1)
    # Show samples for some sentences from random batches
    # for temp in [0.0, 0.9, 1, 1.1, 1.2]:
    #     print('### TEMP ', temp)
    #     model1.generate_words(size=60, temp=temp)

    ## Define model
    if options.task == 'wikisimple':
        print("wikisimple task")
        data = \
            util.load_words_split_types(util.DIR + '/datasets/wikisimple.txt', vocab_size=options.top_words,
                                        limit=options.limit)

    elif options.task == 'file' and options.data_dir is not None:
        print("file task")
        data = \
            util.load_words_split_types(options.data_dir, vocab_size=options.top_words, limit=options.limit)

    else:
        raise Exception('Task {} not recognized.'.format(options.task))

    data_train, data_valid, data_test = data["train"], data["validation"], data["test"]
    x_train, w2i_train, i2w_train = data_train[0], data_train[1], data_train[2]
    x_valid, w2i_valid, i2w_valid = data_valid[0], data_valid[1], data_valid[2]
    x_test, w2i_test, i2w_test = data_test[0], data_test[1], data_test[2]
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

    input_train = Input(shape=(None,))

    embedding_train = Embedding(numwords_train, options.lstm_capacity, input_length=None)

    embedded_train = embedding_train(input_train)

    decoder_lstm_train = LSTM(options.lstm_capacity, return_sequences=True)

    h_train = decoder_lstm_train(embedded_train)

    if options.extra is not None:
        for _ in range(options.extra):
            h_train = LSTM(options.extra, return_sequences=True)(h_train)

    fromhidden_train = Dense(numwords_train, activation='linear')

    out_train = TimeDistributed(fromhidden_train)(h_train)

    model = Model(input_train, out_train)
    opt = keras.optimizers.Adam(lr=options.lr)
    lss = words.sparse_loss
    model.compile(opt, lss)
    model.summary()

    epoch = 0
    instances_seen = 0

    while epoch < options.epochs:

        for batch_train in x_train:
            n_train, l_train = batch_train.shape
            batch_shifted_train = np.concatenate([np.ones((n_train, 1)), batch_train],
                                                 axis=1)  # prepend start symbol
            batch_out_train = np.concatenate([batch_train, np.zeros((n_train, 1))], axis=1)  # append pad symbol

            model.train_on_batch(batch_shifted_train, batch_out_train[:, :, None])

            instances_seen += n_train
        epoch += 1

if __name__ == "__main__":
    main()
