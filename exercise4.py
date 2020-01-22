import keras

import keras.backend as K
from keras.datasets import imdb
from keras.layers import LSTM, Embedding, TimeDistributed, Input, Dense
from keras.models import Model, load_model
import logging
from tensorflow.python.client import device_lib

from tqdm import tqdm
import os, random
from argparse import ArgumentParser, Namespace
import numpy as np

from tensorboardX import SummaryWriter
import math
import util
import words
# TODO refactor (ctrl+alt+shift+L)

CHECK = 6
options = None

model_list = []
reverse_param_list = []
hidden_layer_param_list = []

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


def load_data(task='wikisimple', data_dir=None, limit=None, top_words=1000, batch_size=128):
    if task == 'wikisimple':
        print("wikisimple task")
        data = \
            util.load_words_split_types(util.DIR + '/datasets/wikisimple.txt', vocab_size=top_words, limit=limit)

    elif task == 'file' and data_dir is not None:
        print("file task")
        data = \
            util.load_words_split_types(data_dir, vocab_size=top_words, limit=limit)

    else:
        raise Exception('Task {} not recognized.'.format(task))

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

    x_train = util.batch_pad(x_train, batch_size, add_eos=True)
    x_valid = util.batch_pad(x_valid, batch_size, add_eos=True)
    x_test = util.batch_pad(x_test, batch_size, add_eos=True)

    print('Finished data loading. ', sum([b.shape[0] for b in x_train]), ' training sentences loaded')
    print('Finished data loading. ', sum([b.shape[0] for b in x_valid]), ' validation sentences loaded')
    print('Finished data loading. ', sum([b.shape[0] for b in x_valid]), ' test sentences loaded')
    return [x_train, w2i_train, i2w_train], [x_valid, w2i_valid, i2w_valid], [x_test, w2i_test, i2w_test]


def get_sentence_probability(model, sentence):
        return model.predict(sentence)

def generate_sentences(model, w2i, sentence_beginning="I love", sentence_length = 7,  temperatures=[0.1, 1, 10]):
    sentence_to_feed, generated_sentences = [], []
    for word in sentence_beginning.split():
        sentence_to_feed.append(w2i[word])

    for temperature in temperatures:
        print(f'Temperature = {temperature}')
        generated_sentences.append(words.generate_seq(model, sentence_to_feed, size=sentence_length, temperature=temperature))

def decode(seq, i2w):
    return ' '.join(i2w[id] for id in seq)

def get_perplexity(loss):
    return math.exp(loss)

def get_new_model(lr, i2w_train, lstm_capacity=1000, extra_layers = None , is_reverse = False):
    numwords_train = len(i2w_train)
    input_train = Input(shape=(None,))

    embedding_train = Embedding(numwords_train, lstm_capacity, input_length=None)

    embedded_train = embedding_train(input_train)

    decoder_lstm_train = LSTM(lstm_capacity, return_sequences=True, go_backwards=is_reverse)

    h_train = decoder_lstm_train(embedded_train)

    if extra_layers is not None:
        for _ in range(extra_layers):
            h_train = LSTM(extra_layers, return_sequences=True, go_backwards=is_reverse)(h_train)

    fromhidden_train = Dense(numwords_train, activation='linear')

    out_train = TimeDistributed(fromhidden_train)(h_train)

    model = Model(input_train, out_train)
    opt = keras.optimizers.Adam(lr=lr)
    lss = words.sparse_loss
    model.compile(opt, lss)
    model.summary()
    return model


def create_models(train_list):
    models = []
    for num_hidden in [1, 2]:
        for is_reverse in [False, True]:
            model = get_new_model(lr=options.lr, i2w_train = train_list[2], lstm_capacity=options.lstm_capacity, extra_layers=num_hidden,is_reverse=is_reverse)
            models.append(model)
            reverse_param_list.append(is_reverse)
            hidden_layer_param_list.append(num_hidden)
    return models


def train_model(train_list, model, epochs):
    epoch = 0
    instances_seen = 0
    while epoch < epochs:
        print(f'epoch{epoch}')
        for batch_train in train_list[0]:
            n_train, l_train = batch_train.shape
            batch_shifted_train = np.concatenate([np.ones((n_train, 1)), batch_train],
                                                 axis=1)  # prepend start symbol
            batch_out_train = np.concatenate([batch_train, np.zeros((n_train, 1))], axis=1)  # append pad symbol

            model.train_on_batch(batch_shifted_train, batch_out_train[:, :, None])

            instances_seen += n_train
        epoch += 1
    return model


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
    perplexity = get_perplexity(loss)
    return loss, perplexity


def train_all_models_and_print_loss_perplexity(models, train_list, valid_list, test_list):
    trained_models = []
    for index, model in tqdm(enumerate(models), total=len(models)):
        print(f'')
        model = train_model(train_list, model, options.epochs)
        trained_models.append(model)
        loss_train, perplexity_train = get_loss(model, train_list[0])
        loss_valid, perplexity_valid = get_loss(model, valid_list[0])
        loss_test, perplexity_test = get_loss(model, test_list[0])
        print(f'number epochs = {options.epochs}, reverse_lstm = {reverse_param_list[index]}, hidden_layers = {hidden_layer_param_list[index]}')
        print(f'Train - Loss = {loss_train}, Perplexity = {perplexity_train}')
        print(f'Valid - Loss = {loss_valid}, Perplexity = {perplexity_valid}')
        print(f'Test - Loss = {loss_test}, Perplexity = {perplexity_test}')
    return trained_models

# TODO global other params

def main():
    global options
    parser = create_parser()
    options = parser.parse_args()
    logging.debug("options parsing performed")
    print(options)
    train_list, validation_list, test_list = load_data(task=options.task, data_dir=options.data,limit=options.limit, top_words=options.top_words, batch_size=options.batch)
    logging.debug("load_data performed")
    # train_list = [x_train, w2i_train, i2w_train], validation_list = [x_valid, w2i_valid, i2w_valid] test_list = [x_test, w2i_test, i2w_test]
    models = create_models(train_list)
    logging.debug("create_models performed")
    models = train_all_models_and_print_loss_perplexity(models, train_list, validation_list, test_list)
    generate_sentences(models[0], w2i=train_list[1])
    # def generate_sentences(model, w2i, sentence_beginning="I love", sentence_length=7, temperatures=[0.1, 1, 10]):



if __name__ == "__main__":
    main()