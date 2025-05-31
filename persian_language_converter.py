

'''
Description;

In this file, we will try to create a function helping us to solve Persian Language writing problems.
The aim of it is to stick Farsi letters and the way of showing that should be from right to left.

To solve the problem of showing, we will use arabic-reshaper and python-bidi libraries, 
that before of executing code must be installed.

Versions and libraries;
Python 3.12.9
python-bidi 0.6.6

'''

# Importing necessary libraries

import arabic_reshaper
from bidi.algorithm import get_display

def persian_lang_converter(text):
    '''
    persian_lang_converter is a function that helps us to solve the problems of the Persian language, like sticking letters and writing from right to left
    It takes one input as text and returns the converted input text.
    Duty of arabic_reshaper is sticking letters, and get_display alters writing from right to left. 
    '''
    stiker = arabic_reshaper.reshape(text)
    RtoL_converter = get_display(stiker)
    return RtoL_converter


