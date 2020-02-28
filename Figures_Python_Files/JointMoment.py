import collections
import copy
import os
import re
import csv
import enum
import glob
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pylab as pl
from scipy.signal import butter, filtfilt
import importlib
from tabulate import tabulate
from numpy import nanmean, nanstd
from perimysium import postprocessing as pp
from perimysium import dataman
import pathlib
#####################################################################################
#####################################################################################
# Reading CSV files into a dictionary
directory = 'D:\Ali.K.M.Bonab\Walking_Mass_Inertia_Effect\Data\Data\RRA\*_torque.csv'
files = enumerate(glob.iglob(directory), 1)
dataset = {pathlib.PurePath(f[1]).stem: np.loadtxt(f[1], delimiter=',') for f in files}
#####################################################################################
# Processing Data

