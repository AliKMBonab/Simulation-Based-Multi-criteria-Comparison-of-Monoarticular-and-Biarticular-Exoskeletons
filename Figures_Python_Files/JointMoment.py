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
import Utils as utils
#####################################################################################
#####################################################################################
# Reading CSV files into a dictionary
directory = 'D:\Ali.K.M.Bonab\Walking_Mass_Inertia_Effect\Data\Data\RRA\*_torque.csv'
files = enumerate(glob.iglob(directory), 1)
dataset = {pathlib.PurePath(f[1]).stem: np.loadtxt(f[1], delimiter=',') for f in files}
#####################################################################################
# Processing Data
subjects = ['05','07','09','10','11','12','14']
trials_num = ['01','02','03']
gl_noload = {'noload_subject{}_trial{}'.format(i,j): utils.construct_gl_mass_side(subjectno=i,trialno=j,loadcond='noload') for i in subjects for j in trials_num}
gl_loaded = {'loaded_subject{}_trial{}'.format(i,j): utils.construct_gl_mass_side(subjectno=i,trialno=j,loadcond='loaded') for i in subjects for j in trials_num}
noload_mean_toe_off,_,loaded_mean_toe_off,_ = utils.toe_off_avg_std(gl_noload,gl_loaded)

        


