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
#####################################################################################
# Reading CSV files into a dictionary and constructing gls
rra_dataset = utils.csv2numpy('D:/Ali.K.M.Bonab/Walking_Mass_Inertia_Effect/Data/Data/RRA/rra_final_data.csv') 
unassist_dataset = utils.csv2numpy('D:/Ali.K.M.Bonab/Walking_Mass_Inertia_Effect/Data/Data/Unassist/unassist_final_data.csv') 
# pareto exo torque dataset
directory = 'D:/Ali.K.M.Bonab/Walking_Mass_Inertia_Effect/Data/Data/Pareto/*_torque.csv'
files = enumerate(glob.iglob(directory), 1)
exo_torque_dataset = {pathlib.PurePath(f[1]).stem: np.loadtxt(f[1], delimiter=',') for f in files}
# pareto exo power dataset
directory = 'D:/Ali.K.M.Bonab/Walking_Mass_Inertia_Effect/Data/Data/Pareto/*_power.csv'
files = enumerate(glob.iglob(directory), 1)
exo_power_dataset = {pathlib.PurePath(f[1]).stem: np.loadtxt(f[1], delimiter=',') for f in files}
# gls
gl_noload = {'noload_subject{}_trial{}'.format(i,j): utils.construct_gl_mass_side(subjectno=i,trialno=j,loadcond='noload') for i in subjects for j in trials_num}
gl_loaded = {'loaded_subject{}_trial{}'.format(i,j): utils.construct_gl_mass_side(subjectno=i,trialno=j,loadcond='loaded') for i in subjects for j in trials_num}
#####################################################################################
# Processing Data
# toe-off
_,_,_,_,noload_toe_off,loaded_toe_off = utils.toe_off_avg_std(gl_noload,gl_loaded,subjects=True)

# actuators power profiles
bi_loaded_hip_power = exo_power_dataset['biarticular_pareto_load_hipactuator_power']
bi_loaded_knee_power = exo_power_dataset['biarticular_pareto_load_kneeactuator_power']
bi_noload_hip_power = exo_power_dataset['biarticular_pareto_noload_hipactuator_power']
bi_noload_knee_power = exo_power_dataset['biarticular_pareto_noload_kneeactuator_power']
mono_loaded_hip_power = exo_power_dataset['monoarticular_pareto_load_hipactuator_power']
mono_loaded_knee_power = exo_power_dataset['monoarticular_pareto_load_kneeactuator_power']
mono_noload_hip_power = exo_power_dataset['monoarticular_pareto_noload_hipactuator_power']
mono_noload_knee_power = exo_power_dataset['monoarticular_pareto_noload_kneeactuator_power']
# rmse of power
mean_bi_hip_loadedvsnoload_power,std_bi_hip_loadedvsnoload_power = utils.profiles_all_phases_rmse(bi_loaded_hip_power,bi_noload_hip_power,loaded_toe_off,noload_toe_off)
mean_bi_knee_loadedvsnoload_power,std_bi_knee_loadedvsnoload_power = utils.profiles_all_phases_rmse(bi_loaded_knee_power,bi_noload_knee_power,loaded_toe_off,noload_toe_off)

# actuators torque profiles
bi_loaded_hip_torque = exo_torque_dataset['biarticular_pareto_load_hipactuator_torque']
bi_loaded_knee_torque = exo_torque_dataset['biarticular_pareto_load_kneeactuator_torque']
bi_noload_hip_torque = exo_torque_dataset['biarticular_pareto_noload_hipactuator_torque']
bi_noload_knee_torque = exo_torque_dataset['biarticular_pareto_noload_kneeactuator_torque']
mono_loaded_hip_torque = exo_torque_dataset['monoarticular_pareto_load_hipactuator_torque']
mono_loaded_knee_torque = exo_torque_dataset['monoarticular_pareto_load_kneeactuator_torque']
mono_noload_hip_torque = exo_torque_dataset['monoarticular_pareto_noload_hipactuator_torque']
mono_noload_knee_torque = exo_torque_dataset['monoarticular_pareto_noload_kneeactuator_torque']
