#! /usr/bin/env python3
__author__ = "Yaolin Ge"
__copyright__ = "Copyright 2021, The MASCOT Project, NTNU (https://wiki.math.ntnu.no/mascot)"
__credits__ = ["Yaolin Ge"]
__license__ = "MIT"
__version__ = "1.0.1"
__maintainer__ = "Yaolin Ge"
__email__ = "yaolin.ge@ntnu.no"
__status__ = "UnderDevelopment"


'''
Objective: Processing the data collected from the pre_survey and conduct the preprocessing before the adaptive mission
'''


from PostProcessor import PostProcessor
from PolygonHandler import PolygonCircle
from GridHandler import GridPoly
from PreProcessor import PreProcessor

if __name__ == "__main__":
    a = PostProcessor()
    b = PolygonCircle()
    c = GridPoly()
    d = PreProcessor()