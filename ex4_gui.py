import logging
import os

import PySimpleGUI as sg
import coloredlogs
import numpy as np

from ex4 import generate_sentence, get_probability, encode, load_data, train_model, get_new_model, create_parser, \
    get_loss

'''
All GUI requirements are met here
All needed imports may be found in env.txt file
'''

options = None
model = None
dataset = None  # [data, w2i, i2w]
train_list = None
valid_list = None
test_list = None
coloredlogs.install()
log = logging.getLogger(__name__)
logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))
log.setLevel(logging.INFO)


def create_new_model_with_parameters(is_reverse, num_hidden_layers, dataset_name, num_epochs):
    global model, dataset, train_list, valid_list, test_list
    train_list, valid_list, test_list = load_data(task=dataset_name, data_dir=options.data, limit=options.limit,
                                                  top_words=options.top_words, batch_size=options.batch)
    dataset = train_list
    log.info('Creating new model')
    model = get_new_model(lr=options.lr, i2w_train=dataset[2], lstm_capacity=options.lstm_capacity,
                          extra_layers=num_hidden_layers, is_reverse=is_reverse)
    log.info('Creating model finished.')
    log.info('Training model..')
    model = train_model(train_list, model, num_epochs)
    log.info('Model is ready')


def get_generated_sentence_and_probability(sentence_size, seed_sentence, temperature):
    w2i, i2w = dataset[1], dataset[2]
    probability_sentence = "famous conductors"
    encoded_seq = encode(probability_sentence, w2i)
    float_probability, exp_probability = get_probability(model, np.array(encoded_seq), temperature)
    log.info(
        f'For sentence {probability_sentence} float probability = {float_probability}, exponential probability = {exp_probability}')

    generated_sentence = generate_sentence(model=model, sentence=seed_sentence, w2i=w2i, i2w=i2w, size=sentence_size,
                                           temperature=temperature)
    log.info(
        f'generating sentence for model with next parameters,beginning sentence = {seed_sentence}, size = {sentence_size}, temperature = {temperature}')
    print('*** [', seed_sentence, '] ', generated_sentence)
    return generated_sentence, float_probability


# requirement #7
def create_gui():
    global model
    generate_button_text = 'Generate sentence'
    exit_button_text = 'Exit'
    create_model_text = 'Create Model'
    sg.theme('DarkAmber')  # Add a touch of color
    # All the stuff inside your window.
    layout = [[sg.Text('Choose dataset:'),
               sg.InputCombo(
                   ('wikisimple', 'shakespeare', 'alice', 'coco', 'europarl.40', 'europarl.50', 'europarl.100'),
                   size=(20, 10), default_value='wikisimple', key='dataset_key')],
              [sg.Text('Choose model parameters:')],
              [sg.Text('Is Reversed?:'), sg.InputCombo((False, True), size=(8, 2),
                                                       tooltip='Train on reverse or not', key='reverse_key')],
              [sg.Text('Number of layers:'),
               sg.InputCombo(([1, 2, 3, 4]), size=(10, 10), default_value=1, tooltip='Number of layers',
                             key='num_layers_key')],
              [sg.Text('Number epochs: '),
               sg.Slider(range=(1, 20), orientation='h', size=(34, 20), default_value=1, key='epoch_slider',
                         visible=True)],
              [sg.Button(create_model_text, visible=True)],
              [sg.Multiline(default_text='Enter seed words here', size=(60, 5), visible=False, key='INPUT')],
              [sg.Multiline(default_text='Press "Generate" to view the sentence', size=(60, 5), key='OUTPUT',
                            visible=False)],
              [sg.Text('Temperature:', key='temp_slider_text', visible=False)],
              [sg.Slider(range=(0.1, 20), orientation='h', size=(34, 20), default_value=1.0, resolution=.1,
                         key='temp_slider', visible=False)],
              [sg.Text('sentence size to generate:', visible=False, key='size_choose_key')],
              [sg.InputText(default_text='10', key='-IN-SIZE-', visible=False)],
              [sg.Text('', visible=False, key='probability_key', size=(40, 2))],
              [sg.Text('', visible=False, key='loss_perp_key', size=(70, 6))],
              [sg.Button(generate_button_text, visible=False, key='generate_key')]]
    generation_group = ['generate_key', '-IN-SIZE-', 'size_choose_key', 'temp_slider', 'temp_slider_text', 'OUTPUT',
                        'INPUT', 'probability_key', 'loss_perp_key']
    # Create the Window
    window = sg.Window('LSTM Exercise 4 - Eden Dupont & Daniel Rolnik', layout)
    # Event Loop to process "events" and get the "values" of the inputs
    while True:
        event, values = window.read()
        if event in (None, 'exit_btn_key'):  # if user closes window or clicks cancel
            break
        if event in (None, 'generate_key'):
            if model is not None:
                print("generating sentence...")
                size = int(values['-IN-SIZE-'])
                seed = values['INPUT']
                temp = float(values['temp_slider'])
                generated_sentence, probability = get_generated_sentence_and_probability(sentence_size=size,
                                                                                         seed_sentence=seed,
                                                                                         temperature=temp)

                window['OUTPUT'].update(generated_sentence, visible=True)
                text = "Probability for input sentence is " + str(probability)
                window['probability_key'].update(text)
        if event in (None, create_model_text):
            print("creating model...")

            rev = values['reverse_key']
            num_layer = values['num_layers_key']
            dataset_name = values['dataset_key']
            epochs = values['epoch_slider']
            for x in generation_group:
                window[x].update(visible=True)
            window['INPUT'].update('Please wait...creating model', visible=True)
            window.read()
            create_new_model_with_parameters(is_reverse=rev, num_hidden_layers=num_layer, dataset_name=dataset_name,
                                             num_epochs=epochs)
            text = ""
            for x, name in zip([train_list[0], valid_list[0], test_list[0]], ["Training", "Validation", "Testing"]):
                loss, perp = get_loss(model, x)
                text += f'{name} dataset loss = {loss}, perplexity = {perp}\n'

            window['loss_perp_key'].update(text)
            window['INPUT'].update('Enter seed words here')
            window['OUTPUT'].update('Press "Generate" to view the sentence')

    window.close()


def main():
    global options
    parser = create_parser()
    options = parser.parse_args()
    create_gui()


if __name__ == "__main__":
    main()
