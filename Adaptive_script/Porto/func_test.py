import PySimpleGUI as sg

sg.theme('Dark Amber')  # Let's set our own color theme

# STEP 1 define the layout
layout = [
            [sg.Text('This is a very basic PySimpleGUI layout')],
            [sg.Input()],
            [sg.Button('Button'), sg.Button('Exit')]
         ]

#STEP 2 - create the window
window = sg.Window('My new window', layout)

# STEP3 - the event loop
while True:
    event, values = window.read()   # Read the event that happened and the values dictionary
    print(event, values)
    if event == sg.WIN_CLOSED or event == 'Exit':     # If user closed window with X or if user clicked "Exit" button then exit
        break
    if event == 'Button':
      print('You pressed the button')
window.close()

#%%

import PySimpleGUI as sg

sg.theme('DarkAmber')    # Add a little colr

"""
  DESIGN PATTERN 1 - Single-shot window. Input field has a key.
"""

# 1- the layout

layout = [[sg.Text('My one-shot window.')],
          [sg.InputText(key='-IN-')],
          [sg.Submit(), sg.Cancel()]]

# 2 - the window

window = sg.Window('Window Title', layout)

# 3 - the read
event, values = window.read()

# 4 - the close
window.close()

# finally show the input value in a popup window
sg.popup('You entered', values['-IN-'])

#%%
import PySimpleGUI as sg

# STEP 1 - create the window, read the window, close the window.
event, values = sg.Window('My single-line GUI!',
                    [[sg.Text('My one-shot window.')],
                     [sg.InputText(key='-IN-')],
                     [sg.Submit(), sg.Cancel()]]).read(close=True)

# finally show the input value in a popup window
sg.popup('You entered', values['-IN-'])

#%%

import PySimpleGUI as sg

"""
  DESIGN PATTERN 2 - Multi-read window. Reads and updates fields in a window
"""

sg.theme('Dark Amber')    # Add some color for fun

# 1- the layout
layout = [[sg.Text('Your typed chars appear here:'), sg.Text(size=(15,1), key='-OUTPUT-')],
          [sg.Input(key='-IN-')],
          [sg.Button('Show'), sg.Button('Exit')]]

# 2 - the window
window = sg.Window('Pattern 2', layout)

# 3 - the event loop
while True:
    event, values = window.read()
    print(event, values)
    if event == sg.WIN_CLOSED or event == 'Exit':
        break
    if event == 'Show':
        # Update the "output" text element to be the value of "input" element
        window['-OUTPUT-'].update(values['-IN-'])

        # In older code you'll find it written using FindElement or Element
        # window.FindElement('-OUTPUT-').Update(values['-IN-'])
        # A shortened version of this update can be written without the ".Update"
        # window['-OUTPUT-'](values['-IN-'])

# 4 - the close
window.close()

#%%
import PySimpleGUI as sg

"""
  Simple Form
  Use this design pattern to show a form one time to a user that is "submitted"
"""

layout = [[sg.Text('Please enter your Name, Address, Phone')],
          [sg.Text('Name', size=(10, 1)), sg.InputText(key='-NAME-')],
          [sg.Text('Address', size=(10, 1)), sg.InputText(key='-ADDRESS-')],
          [sg.Text('Phone', size=(10, 1)), sg.InputText(key='-PHONE-')],
          [sg.Button('Submit'), sg.Button('Cancel')]]

window = sg.Window('Simple Data Entry Window', layout)
event, values = window.read(close=True)

if event == 'Submit':
    print('The events was ', event, 'You input', values['-NAME-'], values['-ADDRESS-'], values['-PHONE-'])
else:
    print('User cancelled')

#%%
#!/usr/bin/env python
import PySimpleGUI as sg
"""
    Demonstration of MENUS!
    How do menus work?  Like buttons is how.
    Check out the variable menu_def for a hint on how to 
    define menus
"""
def second_window():

    layout = [[sg.Text('The second form is small \nHere to show that opening a window using a window works')],
              [sg.OK()]]

    window = sg.Window('Second Form', layout)
    event, values = window.read()
    window.close()

def test_menus():

    sg.theme('LightGreen')
    sg.set_options(element_padding=(0, 0))

    # ------ Menu Definition ------ #
    menu_def = [['&File', ['&Open', '&Save', '&Properties', 'E&xit' ]],
                ['&Edit', ['&Paste', ['Special', 'Normal',], 'Undo'],],
                ['&Toolbar', ['---', 'Command &1', 'Command &2', '---', 'Command &3', 'Command &4']],
                ['&Help', '&About...'],]

    right_click_menu = ['Unused', ['Right', '!&Click', '&Menu', 'E&xit', 'Properties']]

    # ------ GUI Defintion ------ #
    layout = [
              [sg.MenubarCustom(menu_def, tearoff=False)],
              [sg.Text('Click right on me to see right click menu')],
              [sg.Output(size=(60,20))],
              [sg.ButtonMenu('ButtonMenu',key='-BMENU-', menu_def=menu_def[0])],
              ]

    window = sg.Window("Windows-like program",
                       layout,
                       default_element_size=(12, 1),
                       grab_anywhere=True,
                       right_click_menu=right_click_menu,
                       default_button_element_size=(12, 1))

    # ------ Loop & Process button menu choices ------ #
    while True:
        event, values = window.read()
        if event is None or event == 'Exit':
            return
        print('Event = ', event)
        # ------ Process menu choices ------ #
        if event == 'About...':
            window.disappear()
            sg.popup('About this program','Version 1.0', 'PySimpleGUI rocks...', grab_anywhere=True)
            window.reappear()
        elif event == 'Open':
            filename = sg.popup_get_file('file to open', no_window=True)
            print(filename)
        elif event == 'Properties':
            second_window()
        elif event == '-BMENU-':
            print('You selected from the button menu:', values['-BMENU-'])

test_menus()

#%%

import PySimpleGUI as sg

# default settings
bw = {'size': (7, 2), 'font': ('Franklin Gothic Book', 24), 'button_color': ("black", "#F8F8F8")}
bt = {'size': (7, 2), 'font': ('Franklin Gothic Book', 24), 'button_color': ("black", "#F1EABC")}
bo = {'size': (15, 2), 'font': ('Franklin Gothic Book', 24), 'button_color': ("black", "#ECA527"), 'focus': True}

layout = [
    [sg.Text('PyDataMath-II', size=(50, 1), justification='right', background_color="#272533",
             text_color='white', font=('Franklin Gothic Book', 14, 'bold'))],
    [sg.Text('0.0000', size=(18, 1), justification='right', background_color='black', text_color='red',
             font=('Digital-7', 48), relief='sunken', key="_DISPLAY_")],
    [sg.Button('C', **bt), sg.Button('CE', **bt), sg.Button('%', **bt), sg.Button("/", **bt)],
    [sg.Button('7', **bw), sg.Button('8', **bw), sg.Button('9', **bw), sg.Button("*", **bt)],
    [sg.Button('4', **bw), sg.Button('5', **bw), sg.Button('6', **bw), sg.Button("-", **bt)],
    [sg.Button('1', **bw), sg.Button('2', **bw), sg.Button('3', **bw), sg.Button("+", **bt)],
    [sg.Button('0', **bw), sg.Button('.', **bw), sg.Button('=', **bo, bind_return_key=True)],
]
window = sg.Window('PyDataMath-II', layout=layout, background_color="#272533", return_keyboard_events=True)

''' calculator functions '''
var = {'front': [], 'back': [], 'decimal': False, 'x_val': 0.0, 'y_val': 0.0, 'result': 0.0, 'operator': ''}


# helper functions
def format_number():
    ''' create a consolidated string of numbers from front and back lists '''
    return float(''.join(var['front']) + '.' + ''.join(var['back']))


def update_display(display_value):
    ''' update the calculator display after an event click '''
    try:
        window['_DISPLAY_'].update(value='{:,.4f}'.format(display_value))
    except:
        window['_DISPLAY_'].update(value=display_value)


# click events
def number_click(event):
    ''' number button button click event '''
    global var
    if var['decimal']:
        var['back'].append(event)
    else:
        var['front'].append(event)
    update_display(format_number())


def clear_click():
    ''' CE or C button click event '''
    global var
    var['front'].clear()
    var['back'].clear()
    var['decimal'] = False


def operator_click(event):
    ''' + - / * button click event '''
    global var
    var['operator'] = event
    try:
        var['x_val'] = format_number()
    except:
        var['x_val'] = var['result']
    clear_click()


def calculate_click():
    ''' equals button click event '''
    global var
    var['y_val'] = format_number()
    try:
        var['result'] = eval(str(var['x_val']) + var['operator'] + str(var['y_val']))
        update_display(var['result'])
        clear_click()
    except:
        update_display("ERROR! DIV/0")
        clear_click()


while True:
    event, values = window.read()
    print(event)
    if event is None:
        break
    if event in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']:
        number_click(event)
    if event in ['Escape:27', 'C', 'CE']:  # 'Escape:27 for keyboard control
        clear_click()
        update_display(0.0)
        var['result'] = 0.0
    if event in ['+', '-', '*', '/']:
        operator_click(event)
    if event == '=':
        calculate_click()
    if event == '.':
        var['decimal'] = True
    if event == '%':
        update_display(var['result'] / 100.0)

#%%
import PySimpleGUI as sg

class Usarios:
    def __init__(self, isusario, nome, telefone, email, usario, senha):
        pass

    def insertUser(self):
        return 'User inserted'

layout = [  [sg.Text('Informe os dados:', size=(25,1), justification='center', font=("Verdana", "10", "bold"))],
            [sg.Text('idUsario:', size=(10,1), justification='right'), sg.Input(size=(14,1), key='-USARIO-ID-', do_not_clear=False), sg.Button('Buscar')],
            [sg.T('Nome:', size=(10,1), justification='right'), sg.I(key='-NOME-', do_not_clear=False)],
            [sg.T('Telefone:', size=(10,1), justification='right'), sg.I(key='-TELE-', do_not_clear=False)],
            [sg.T('E-mail:', size=(10,1), justification='right'), sg.I(key='-EMAIL-', do_not_clear=False)],
            [sg.T('Usario:', size=(10,1), justification='right'), sg.I(key='-USARIO-', do_not_clear=False)],
            [sg.T('Senha:', size=(10,1), justification='right'), sg.I(key='-SENHA-', do_not_clear=False)],
            [sg.T(' '*8), sg.Button('Inserir'), sg.Button('Alterar'), sg.Button('Excluir')],
            [sg.T(key='-MESSAGE-', size=(30,1), font=("Verdana", "9", "italic"))],
            [sg.Text('Busca realizda com sucesso!',size=(30,1), justification='center', font=("Verdana", "10", "italic"))]]

window = sg.Window('Informe Os Dados', layout, font='Calibri 10', default_element_size=(25,1))

while True:             # Event Loop
    event, values = window.read()
    print(event, values)
    if event is None:
        break
    if event == 'Inserir':
        user = Usarios(isusario=values['-USARIO-ID-'],
                       nome=values['-NOME-'],
                       telefone=values['-TELE-'],
                       email=values['-EMAIL-'],
                       usario=values['-USARIO-'],
                       senha=values['-SENHA-'])
        window['-MESSAGE-'].Update(user.insertUser())
    else:
        window['-MESSAGE-'].Update('Not yet implemlented')




