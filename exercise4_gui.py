import exercise4
import PySimpleGUI as sg

options = None
#  TODO add more datasets to options in exercise4
#requirement #7
def create_gui():
    generate_button_text = 'Generate'
    exit_button_text = 'Exit'
    sg.theme('DarkAmber')  # Add a touch of color
    # All the stuff inside your window.
    layout = [[sg.Text('Choose dataset:')],
              [sg.InputCombo(('wikisimple', 'shakespeare', 'alice', 'coco.valannotations', 'europarl.40', 'europarl.50', 'europarl.100'), size=(20, 1))],
              [sg.Multiline(default_text='Enter seed words here', size=(60, 5))],
              [sg.Multiline(default_text='Press "Generate" to view the sentence', size=(60, 5), key='OUTPUT')],
              [sg.Text('Temperature:')],
              [sg.Slider(range=(0, 20), orientation='h', size=(34, 20), default_value=1, resolution=.1)],
              [sg.Text('sentence size to generate:')],
              [sg.InputText(default_text='10', key='-IN-SIZE-')],
              [sg.Button(generate_button_text), sg.Button(exit_button_text)]]

    # Create the Window
    window = sg.Window('LSTM Exercise 4 - Eden Dupont & Daniel Rolnik', layout)
    # Event Loop to process "events" and get the "values" of the inputs
    while True:
        event, values = window.read()
        if event in (None, exit_button_text):  # if user closes window or clicks cancel
            break
        if event in (None, generate_button_text):
            print("generating sentence...")
            generated_sentence = 'vrr'  # TODO add proper value
            window['OUTPUT'].update(generated_sentence)
            # print('You entered ', values[0])
    window.close()


def main():
    global options
    parser = exercise4.create_parser()
    options = parser.parse_args()
    create_gui()


if __name__ == "__main__":
    main()
