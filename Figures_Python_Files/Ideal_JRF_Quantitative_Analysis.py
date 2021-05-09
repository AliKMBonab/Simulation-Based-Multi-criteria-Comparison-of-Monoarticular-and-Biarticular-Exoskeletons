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
import seaborn as sns
import Utils as utils
from Colors import colors as mycolors
#####################################################################################
subjects = ['05','07','09','10','11','12','14']
trials_num = ['01','02','03']
gait_cycle = np.linspace(0,100,1000)
plt.rcParams.update({'font.size': 12})
#####################################################################################
# assisted reaction moments
directory = r'.\Data\Ideal\*_reaction_moments.csv'
files = enumerate(glob.iglob(directory), 1)
jointreaction_dataset = {pathlib.PurePath(f[1]).stem: np.loadtxt(f[1], delimiter=',') for f in files}
bi_noload_RM_dictionary = utils.clasify_data(jointreaction_dataset['biarticular_ideal_noload_reaction_moments'],loadcondition='noload')
bi_loaded_RM_dictionary = utils.clasify_data(jointreaction_dataset['biarticular_ideal_loaded_reaction_moments'],loadcondition='loaded')
mono_noload_RM_dictionary = utils.clasify_data(jointreaction_dataset['monoarticular_ideal_noload_reaction_moments'],loadcondition='noload')
mono_loaded_RM_dictionary = utils.clasify_data(jointreaction_dataset['monoarticular_ideal_loaded_reaction_moments'],loadcondition='loaded')
# assisted reaction forces
directory = r'.\Data\Ideal\*_reaction_forces.csv'
files = enumerate(glob.iglob(directory), 1)
jointreaction_dataset = {pathlib.PurePath(f[1]).stem: np.loadtxt(f[1], delimiter=',') for f in files}
bi_noload_RF_dictionary = utils.clasify_data(jointreaction_dataset['biarticular_ideal_noload_reaction_forces'],loadcondition='noload',forces_name=['Fx','Fy','Fz'])
bi_loaded_RF_dictionary = utils.clasify_data(jointreaction_dataset['biarticular_ideal_loaded_reaction_forces'],loadcondition='loaded',forces_name=['Fx','Fy','Fz'])
mono_noload_RF_dictionary = utils.clasify_data(jointreaction_dataset['monoarticular_ideal_noload_reaction_forces'],loadcondition='noload',forces_name=['Fx','Fy','Fz'])
mono_loaded_RF_dictionary = utils.clasify_data(jointreaction_dataset['monoarticular_ideal_loaded_reaction_forces'],loadcondition='loaded',forces_name=['Fx','Fy','Fz'])
#####################################################################################
# Reading CSV files into a dictionary and constructing gls
# joint reaction moment dataset
directory = r'.\Data\Unassist\*_reaction_moments.csv'
files = enumerate(glob.iglob(directory), 1)
jointreaction_dataset = {pathlib.PurePath(f[1]).stem: np.loadtxt(f[1], delimiter=',') for f in files}
noload_RM_dictionary = utils.clasify_data(jointreaction_dataset['unassist_noload_reaction_moments'],loadcondition='noload')
loaded_RM_dictionary = utils.clasify_data(jointreaction_dataset['unassist_loaded_reaction_moments'],loadcondition='loaded')
# joint reaction force dataset
directory = r'.\Data\Unassist\*_reaction_forces.csv'
files = enumerate(glob.iglob(directory), 1)
jointreaction_dataset = {pathlib.PurePath(f[1]).stem: np.loadtxt(f[1], delimiter=',') for f in files}
noload_RF_dictionary = utils.clasify_data(jointreaction_dataset['unassist_noload_reaction_forces'],loadcondition='noload',forces_name=['Fx','Fy','Fz'])
loaded_RF_dictionary = utils.clasify_data(jointreaction_dataset['unassist_loaded_reaction_forces'],loadcondition='loaded',forces_name=['Fx','Fy','Fz'])
#####################################################################################
# gls
gl_noload = {'noload_subject{}_trial{}'.format(i,j): utils.construct_gl_mass_side(subjectno=i,trialno=j,loadcond='noload') for i in subjects for j in trials_num}
gl_loaded = {'loaded_subject{}_trial{}'.format(i,j): utils.construct_gl_mass_side(subjectno=i,trialno=j,loadcond='loaded') for i in subjects for j in trials_num}
#####################################################################################
# Processing Data
# toe-off
_,_,_,_,noload_subjects_toe_off,loaded_subjects_toe_off = utils.toe_off_avg_std(gl_noload,gl_loaded,subjects=True)


    # loaded walking reaction forces analysis

mean_max_jrf_change_dict,std_max_jrf_change_dict = utils.quantify_max_jrf_change(noload_subjects_toe_off,noload_RF_dictionary,bi_noload_RF_dictionary)
