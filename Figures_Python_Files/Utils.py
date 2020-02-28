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
#######################################################################
#######################################################################
def toeoff_pgc(gl, side):
    if side == 'right':
        toeoff = gl.right_toeoff
        strike = gl.right_strike
    elif side == 'left':
        toeoff = gl.left_toeoff
        strike = gl.left_strike
    cycle_duration = (gl.cycle_end - gl.cycle_start)
    while toeoff < strike:
        toeoff += cycle_duration
    while toeoff > strike + cycle_duration:
        toeoff -= cycle_duration
    return pp.percent_duration_single(toeoff,strike,strike + cycle_duration)

def construct_gl_mass_side(subjectno,trialno,loadcond):
    """This function has been designed to construct gl from the dataset. It also returns subject mass
       and trial number to be used on other functions"""
    import Subjects_Dataset as sd
    if loadcond == 'noload':
        data = sd.noload_dataset["subject{}_noload_trial{}".format(subjectno,trialno)]
    elif loadcond == 'load' or loadcond == 'loaded':
        data = sd.loaded_dataset["subject{}_loaded_trial{}".format(subjectno,trialno)]
    else:
        raise Exception("load condition is wrong.")
    mass = data["mass"]
    side = data["primary_legs"]
    gl = dataman.GaitLandmarks( primary_leg = data['primary_legs'],
                                cycle_start = data['subjects_cycle_start_time'],
                                cycle_end   = data['subjects_cycle_end_time'],
                                left_strike = data['footstrike_left_leg'],
                                left_toeoff = data['toeoff_time_left_leg'],
                                right_strike= data['footstrike_right_leg'],
                                right_toeoff= data['toeoff_time_right_leg'])
    return gl,mass,side

def mean_std_over_subjects(data):
    mean = np.nanmean(data,axis=1)
    std = np.nanstd(data,axis=1)

def toe_off_avg_std(gl_noload,gl_loaded):
    '''This function returns the mean toe off percentage for loaded and noloaded subjects
        parameters:
            gl_noload: a dictionary of noload subjects gait landmark
            gl_loaded: a dictionary of loaded subjects gait landmark
        output:
            np.mean(noload_toe_off),np.std(noload_toe_off),np.mean(loaded_toe_off),np.std(loaded_toe_off)
        '''
    noload_toe_off= np.zeros(21)
    loaded_toe_off= np.zeros(21)
    c0 = 0
    c1 = 0
    for key in gl_noload.keys():
        noload_toe_off[c0] =toeoff_pgc(gl=gl_noload[key][0],side= gl_noload[key][2])
        c0+=1
    for key in gl_loaded.keys():
        loaded_toe_off[c1] =toeoff_pgc(gl=gl_loaded[key][0],side= gl_loaded[key][2])
        c1+=1
    return np.mean(noload_toe_off),np.std(noload_toe_off),np.mean(loaded_toe_off),np.std(loaded_toe_off)