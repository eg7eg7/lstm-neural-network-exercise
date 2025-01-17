import math
import os
from argparse import ArgumentParser

import coloredlogs
import keras
import logging
import numpy as np
from keras.layers import LSTM, Embedding, TimeDistributed, Input, Dense
from keras.models import Model
from scipy.special import logsumexp
from tqdm import tqdm

import util
import words

#  Eden Dupont 204808596
#  Daniil Rolnik 334018009

'''
All none GUI requirements are met here
All needed imports may be found in env.txt file
'''

CHECK = 6
options = None

model_list = []
reverse_param_list = []
hidden_layer_param_list = []
window = None
coloredlogs.install()
log = logging.getLogger(__name__)
logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))
log.setLevel(logging.INFO)


def create_parser():
    # Parse the command line options
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
                        default=100000, type=int)

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
    dataset_dict = {'wikisimple': '/datasets/wikisimple.txt',
                    'europarl.100': '/datasets/europarl.100.txt',
                    'coco': '/datasets/coco.valannotations.txt',
                    'europarl.50': '/datasets/europarl.50.txt',
                    'europarl.40': '/datasets/europarl.40.txt',
                    'shakespeare': '/datasets/shakespeare.txt',
                    'alice': '/datasets/alice.txt'
                    }
    directory = dataset_dict[task]
    if dir is not None:
        data = util.load_words_split_types(util.DIR + directory, vocab_size=top_words, limit=limit)
    else:
        raise Exception('Task {} not recognized.'.format(task))

    data_train, data_valid, data_test = data["train"], data["validation"], data["test"]
    x_train, w2i_train, i2w_train = data_train[0], data_train[1], data_train[2]
    x_valid, w2i_valid, i2w_valid = data_valid[0], data_valid[1], data_valid[2]
    x_test, w2i_test, i2w_test = data_test[0], data_test[1], data_test[2]

    x_train = util.batch_pad(x_train, batch_size, add_eos=True)
    x_valid = util.batch_pad(x_valid, batch_size, add_eos=True)
    x_test = util.batch_pad(x_test, batch_size, add_eos=True)

    print('Finished data loading. ', sum([b.shape[0] for b in x_train]), ' training sentences loaded')
    print('Finished data loading. ', sum([b.shape[0] for b in x_valid]), ' validation sentences loaded')
    print('Finished data loading. ', sum([b.shape[0] for b in x_valid]), ' test sentences loaded')
    return [x_train, w2i_train, i2w_train], [x_valid, w2i_valid, i2w_valid], [x_test, w2i_test, i2w_test]


def get_probability(model, seed, temp):
    exp_total_prob = 1
    ls = seed.shape[0]
    probs = model.predict(seed[None, :])
    for i in range(1, ls):
        l_probs = probs[0, i - 1, :]
        l_probs = l_probs / temp
        l_probs = l_probs - logsumexp(l_probs)
        l_probs = np.exp(l_probs)
        cur_prob = l_probs[seed[i]]
        exp_total_prob *= cur_prob
    total_prob_str = format(exp_total_prob, '.20f')
    float_total_prob = total_prob_str.replace(',', '.')
    return float_total_prob, exp_total_prob


def generate_sentence(model, sentence, w2i, i2w, size, temperature):
    encoded_seq = encode(sentence, w2i)
    generated_tokens = words.generate_seq(model=model, seed=np.array(encoded_seq), size=size, temperature=temperature)
    generated_seq = decode(generated_tokens, i2w)
    return generated_seq


def decode(seq, i2w):
    return ' '.join(i2w[word_id] for word_id in seq)


def encode(seq, w2i):
    encoded_seq = []
    seq = seq.split()
    for word in seq:
        if word in w2i:
            encoded_seq.append(w2i[word])
        else:
            encoded_seq.append(w2i[util.EXTRA_SYMBOLS[2]])  # util.EXTRA_SYMBOLS[2] = <UNK> Unknown Tag
    return encoded_seq


def get_perplexity(loss):
    return math.exp(loss)


def get_new_model(lr, i2w_train, lstm_capacity=1000, extra_layers=None, is_reverse=False):
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


#### create 4 LSTM neural networks with 1 or 2 hidden layers and reverse=True or reverse = False
# requirement #8
def create_models(train_list):
    models = []
    for num_hidden in [1, 2]:
        for is_reverse in [False, True]:
            model = get_new_model(lr=options.lr, i2w_train=train_list[2], lstm_capacity=options.lstm_capacity,
                                  extra_layers=num_hidden, is_reverse=is_reverse)
            models.append(model)
            reverse_param_list.append(is_reverse)
            hidden_layer_param_list.append(num_hidden)
    return models


def train_model(train_list, model, epochs):
    epoch = 0
    instances_seen = 0
    while epoch < epochs:
        log.info(f'epoch #{epoch} out of {epochs}')
        log.info(f'Training model in batches..')
        for batch_train in tqdm(train_list[0], position=0):
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

    for batch_train in tqdm(x, position=0):
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
    num_epochs = options.epochs
    for index, model in tqdm(enumerate(models), total=len(models), position=0):
        model = train_model(train_list, model, epochs=num_epochs)
        trained_models.append(model)
        log.info(f'\nmodel#{index}:')
        log.info(
            f'number epochs = {num_epochs}, reverse_lstm = {reverse_param_list[index]}, hidden_layers = {hidden_layer_param_list[index]}')

        for data, title in zip([train_list[0], valid_list[0], test_list[0]], ['Train', 'Validation', 'Test']):
            log.info(f'Calculating loss and perplexity for {title}')
            loss, perplexity = get_loss(model, data)
            log.info(f'{title} - Loss = {loss}, Perplexity = {perplexity}')
    return trained_models


def main():
    global options
    parser = create_parser()
    options = parser.parse_args()
    log.info("options parsing performed")
    log.info(options)
    train_list, validation_list, test_list = load_data(task=options.task, data_dir=options.data, limit=options.limit,
                                                       top_words=options.top_words, batch_size=options.batch)
    log.info("load_data performed")
    #  train_list = [data, w2i, i2w]

    #  requirement #4
    models = create_models(train_list)
    log.info("create_models performed")

    # requirement #5, 6, 9
    sentence1 = util.EXTRA_SYMBOLS[1] + " I love"
    sentence2 = util.EXTRA_SYMBOLS[1] + " I love cupcakes"
    probability_sentences = [sentence1, sentence2]
    models = train_all_models_and_print_loss_perplexity(models, train_list, validation_list, test_list)
    for index, model in enumerate(models):
        for temperature in [0.1, 1, 10]:
            # requirement #5 + #9
            for probability_sentence in probability_sentences:
                encoded_seq = encode(probability_sentence, train_list[1])
                float_probability, exp_probability = get_probability(model, np.array(encoded_seq), temperature)
                log.info(
                    f'For sentence {probability_sentence} float probability = {float_probability}, exponential probability = {exp_probability}')

            # requirement  # 6
            size = 7
            w2i, i2w = train_list[1], train_list[2]
            generated_sentence = generate_sentence(model=model, sentence=sentence1, w2i=w2i, i2w=i2w, size=size,
                                                   temperature=temperature)
            log.info(
                f'generated sentence for model #{index} with next parameters,beginning sentence = {sentence1}, size = {size}, temperature = {temperature}')
            print('*** [', sentence1, '] ', generated_sentence)


if __name__ == "__main__":
    main()
