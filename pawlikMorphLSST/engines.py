"""
This module contains the engines to help with the parallel processing of the
image via various python libraries.
"""

__all__ = ["multiprocEngine"]


class multiprocEngine(object):
    '''Class existences so that multiprocessing Pool method can be used on _analyseImage.
       Basically a way to pass the function arguments that are he same with
       one variable argument, i.e the file name'''

    def __init__(self, callFunc, parameters):
        '''This sets the arguments for the function passed to pool via
           engine'''
        self.callFunc = callFunc
        self.parameters = parameters

    def __call__(self, filename):
        '''This calls the function when engine is called on pool'''
        return self.callFunc(filename, *self.parameters)
