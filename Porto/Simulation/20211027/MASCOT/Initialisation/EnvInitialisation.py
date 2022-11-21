#! /usr/bin/env python3
__author__ = "Yaolin Ge"
__copyright__ = "Copyright 2021, The MASCOT Project, NTNU (https://wiki.math.ntnu.no/mascot)"
__credits__ = ["Yaolin Ge"]
__license__ = "MIT"
__version__ = "1.0.1"
__maintainer__ = "Yaolin Ge"
__email__ = "yaolin.ge@ntnu.no"
__status__ = "UnderDevelopment"

import os

class Initialiser:
    path_global = None

    def __init__(self):
        print("Now setting up the global environment")
        self.get_path_global()

    def get_path_global(self):
        print("The global environment is defined as: ")
        print(os.getcwd())
        self.path_global = os.getcwd()
        f_pre = open("path_global.txt", 'w')
        f_pre.write(self.path_global)
        f_pre.close()
        print("The global path is set up successfully!")

if __name__ == "__main__":
    a = Initialiser()
