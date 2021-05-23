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
# ideal exo torque dataset
directory = './Data/Ideal/*_torque.csv'
files = enumerate(glob.iglob(directory), 1)
ideal_exo_torque_dataset = {pathlib.PurePath(f[1]).stem: np.loadtxt(f[1], delimiter=',') for f in files}
# ideal exo power dataset
directory = './Data/Ideal/*_power.csv'
files = enumerate(glob.iglob(directory), 1)
ideal_exo_power_dataset = {pathlib.PurePath(f[1]).stem: np.loadtxt(f[1], delimiter=',') for f in files}
#####################################################################################
# pareto exo torque dataset
directory = './Data/Pareto/*_torque.csv'
files = enumerate(glob.iglob(directory), 1)
exo_torque_dataset = {pathlib.PurePath(f[1]).stem: np.loadtxt(f[1], delimiter=',') for f in files}
# pareto exo power dataset
directory = './Data/Pareto/*_power.csv'
files = enumerate(glob.iglob(directory), 1)
exo_power_dataset = {pathlib.PurePath(f[1]).stem: np.loadtxt(f[1], delimiter=',') for f in files}
# gls
gl_noload = {'noload_subject{}_trial{}'.format(i,j): utils.construct_gl_mass_side(subjectno=i,trialno=j,loadcond='noload') for i in subjects for j in trials_num}
gl_loaded = {'loaded_subject{}_trial{}'.format(i,j): utils.construct_gl_mass_side(subjectno=i,trialno=j,loadcond='loaded') for i in subjects for j in trials_num}
#####################################################################################
subjects = ['05','07','09','10','11','12','14']
trials_num = ['01','02','03']
gait_cycle = np.linspace(0,100,1000)
# toe-off
_,_,_,_,noload_subjects_toe_off,loaded_subjects_toe_off = utils.toe_off_avg_std(gl_noload,gl_loaded,subjects=True)
#####################################################################################
# exoskeleton torque profiles
# biarticular
# hip
ideal_bi_loaded_hip_torque = utils.normalize_direction_data(ideal_exo_torque_dataset['biarticular_ideal_loaded_hipactuator_torque'],gl_noload,direction=False)
ideal_bi_noload_hip_torque = utils.normalize_direction_data(ideal_exo_torque_dataset['biarticular_ideal_noload_hipactuator_torque'],gl_noload,direction=False)
# knee
ideal_bi_loaded_knee_torque = utils.normalize_direction_data(ideal_exo_torque_dataset['biarticular_ideal_loaded_kneeactuator_torque'],gl_noload,direction=True)
ideal_bi_noload_knee_torque = utils.normalize_direction_data(ideal_exo_torque_dataset['biarticular_ideal_noload_kneeactuator_torque'],gl_noload,direction=True)
# monoarticular
# hip
ideal_mono_loaded_hip_torque = utils.normalize_direction_data(ideal_exo_torque_dataset['monoarticular_ideal_loaded_hipactuator_torque'],gl_noload,direction=False)
ideal_mono_noload_hip_torque = utils.normalize_direction_data(ideal_exo_torque_dataset['monoarticular_ideal_noload_hipactuator_torque'],gl_noload,direction=False)
# knee
ideal_mono_loaded_knee_torque = utils.normalize_direction_data(ideal_exo_torque_dataset['monoarticular_ideal_loaded_kneeactuator_torque'],gl_noload,direction=True)
ideal_mono_noload_knee_torque = utils.normalize_direction_data(ideal_exo_torque_dataset['monoarticular_ideal_noload_kneeactuator_torque'],gl_noload,direction=True)
#******************************
# exoskeleton power profiles
# biarticular
# hip
ideal_bi_loaded_hip_power = utils.normalize_direction_data(ideal_exo_power_dataset['biarticular_ideal_loaded_hipactuator_power'],gl_noload,direction=False)
ideal_bi_noload_hip_power = utils.normalize_direction_data(ideal_exo_power_dataset['biarticular_ideal_noload_hipactuator_power'],gl_noload,direction=False)
# knee
ideal_bi_loaded_knee_power = utils.normalize_direction_data(ideal_exo_power_dataset['biarticular_ideal_loaded_kneeactuator_power'],gl_noload,direction=False)
ideal_bi_noload_knee_power = utils.normalize_direction_data(ideal_exo_power_dataset['biarticular_ideal_noload_kneeactuator_power'],gl_noload,direction=False)
# monoarticular
# hip
ideal_mono_loaded_hip_power = utils.normalize_direction_data(ideal_exo_power_dataset['monoarticular_ideal_loaded_hipactuator_power'],gl_noload,direction=False)
ideal_mono_noload_hip_power = utils.normalize_direction_data(ideal_exo_power_dataset['monoarticular_ideal_noload_hipactuator_power'],gl_noload,direction=False)
# knee
ideal_mono_loaded_knee_power = utils.normalize_direction_data(ideal_exo_power_dataset['monoarticular_ideal_loaded_kneeactuator_power'],gl_noload,direction=False)
ideal_mono_noload_knee_power = utils.normalize_direction_data(ideal_exo_power_dataset['monoarticular_ideal_noload_kneeactuator_power'],gl_noload,direction=False)
#####################################################################################
# Processing Data
# toe-off
noload_mean_toe_off,_,loaded_mean_toe_off,_= utils.toe_off_avg_std(gl_noload,gl_loaded)
# actuators power profiles
bi_loaded_hip_power = utils.pareto_profiles_avg_std(exo_power_dataset['biarticular_pareto_load_hipactuator_power'],gl_noload,change_direction=False,mean_std=False)
bi_loaded_knee_power = utils.pareto_profiles_avg_std(exo_power_dataset['biarticular_pareto_load_kneeactuator_power'],gl_noload,change_direction=False,mean_std=False)
bi_noload_hip_power = utils.pareto_profiles_avg_std(exo_power_dataset['biarticular_pareto_noload_hipactuator_power'],gl_noload,change_direction=False,mean_std=False)
bi_noload_knee_power = utils.pareto_profiles_avg_std(exo_power_dataset['biarticular_pareto_noload_kneeactuator_power'],gl_noload,change_direction=False,mean_std=False)
mono_loaded_hip_power = utils.pareto_profiles_avg_std(exo_power_dataset['monoarticular_pareto_load_hipactuator_power'],gl_noload,change_direction=False,mean_std=False)
mono_loaded_knee_power = utils.pareto_profiles_avg_std(exo_power_dataset['monoarticular_pareto_load_kneeactuator_power'],gl_noload,change_direction=False,mean_std=False)
mono_noload_hip_power = utils.pareto_profiles_avg_std(exo_power_dataset['monoarticular_pareto_noload_hipactuator_power'],gl_noload,change_direction=False,mean_std=False)
mono_noload_knee_power = utils.pareto_profiles_avg_std(exo_power_dataset['monoarticular_pareto_noload_kneeactuator_power'],gl_noload,change_direction=False,mean_std=False)

# actuators torque profiles
bi_loaded_hip_torque = utils.pareto_profiles_avg_std(exo_torque_dataset['biarticular_pareto_load_hipactuator_torque'],gl_noload,change_direction=False,mean_std=False)
bi_loaded_knee_torque = utils.pareto_profiles_avg_std(exo_torque_dataset['biarticular_pareto_load_kneeactuator_torque'],gl_noload,change_direction=True,mean_std=False)
bi_noload_hip_torque = utils.pareto_profiles_avg_std(exo_torque_dataset['biarticular_pareto_noload_hipactuator_torque'],gl_noload,change_direction=False,mean_std=False)
bi_noload_knee_torque = utils.pareto_profiles_avg_std(exo_torque_dataset['biarticular_pareto_noload_kneeactuator_torque'],gl_noload,change_direction=True,mean_std=False)
mono_loaded_hip_torque = utils.pareto_profiles_avg_std(exo_torque_dataset['monoarticular_pareto_load_hipactuator_torque'],gl_noload,change_direction=False,mean_std=False)
mono_loaded_knee_torque = utils.pareto_profiles_avg_std(exo_torque_dataset['monoarticular_pareto_load_kneeactuator_torque'],gl_noload,change_direction=True,mean_std=False)
mono_noload_hip_torque = utils.pareto_profiles_avg_std(exo_torque_dataset['monoarticular_pareto_noload_hipactuator_torque'],gl_noload,change_direction=False,mean_std=False)
mono_noload_knee_torque = utils.pareto_profiles_avg_std(exo_torque_dataset['monoarticular_pareto_noload_kneeactuator_torque'],gl_noload,change_direction=True,mean_std=False)
#####################################################################################
# Paretofront
# indices
bi_noload_indices = [25,24,23,22,21,19,18,17,13,12,11,1]
bi_loaded_indices = [25,24,23,22,21,17,16,13,12,11,6,1]
mono_noload_indices = [25,20,15,14,13,8,7,6,2,1]
mono_loaded_indices = [25,20,15,10,5,4,3,2,1]
# biarticular power profiles
paretofront_bi_noload_hip_power_max_change_phases = utils.paretofront_profiles_change_phases(bi_noload_hip_power,ideal_bi_noload_hip_power,noload_subjects_toe_off,bi_noload_indices)
paretofront_bi_loaded_hip_power_max_change_phases = utils.paretofront_profiles_change_phases(bi_loaded_hip_power,ideal_bi_loaded_hip_power,loaded_subjects_toe_off,bi_loaded_indices)
paretofront_bi_noload_knee_power_max_change_phases = utils.paretofront_profiles_change_phases(bi_noload_knee_power,ideal_bi_noload_knee_power,noload_subjects_toe_off,bi_noload_indices)
paretofront_bi_loaded_knee_power_max_change_phases = utils.paretofront_profiles_change_phases(bi_loaded_knee_power,ideal_bi_loaded_knee_power,loaded_subjects_toe_off,bi_loaded_indices)
biarticular_power_change_dict = {'loaded hip power': paretofront_bi_loaded_hip_power_max_change_phases,'loaded knee power': paretofront_bi_loaded_knee_power_max_change_phases,\
                                 'noload hip power': paretofront_bi_noload_hip_power_max_change_phases,'noload knee power': paretofront_bi_noload_knee_power_max_change_phases}
#  monoarticular power profiles
paretofront_mono_noload_hip_power_max_change_phases = utils.paretofront_profiles_change_phases(mono_noload_hip_power,ideal_mono_noload_hip_power,noload_subjects_toe_off,mono_noload_indices)
paretofront_mono_loaded_hip_power_max_change_phases = utils.paretofront_profiles_change_phases(mono_loaded_hip_power,ideal_mono_loaded_hip_power,loaded_subjects_toe_off,mono_loaded_indices)
paretofront_mono_noload_knee_power_max_change_phases = utils.paretofront_profiles_change_phases(mono_noload_knee_power,ideal_mono_noload_knee_power,noload_subjects_toe_off,mono_noload_indices)
paretofront_mono_loaded_knee_power_max_change_phases = utils.paretofront_profiles_change_phases(mono_loaded_knee_power,ideal_mono_loaded_knee_power,loaded_subjects_toe_off,mono_loaded_indices)
monoarticular_power_change_dict = {'loaded hip power': paretofront_mono_loaded_hip_power_max_change_phases,'loaded knee power': paretofront_mono_loaded_knee_power_max_change_phases,\
                                 'noload hip power': paretofront_mono_noload_hip_power_max_change_phases,'noload knee power': paretofront_mono_noload_knee_power_max_change_phases}
# biarticular torque profiles
paretofront_bi_noload_hip_torque_max_change_phases = utils.paretofront_profiles_change_phases(bi_noload_hip_torque,ideal_bi_noload_hip_torque,noload_subjects_toe_off,bi_noload_indices)
paretofront_bi_loaded_hip_torque_max_change_phases = utils.paretofront_profiles_change_phases(bi_loaded_hip_torque,ideal_bi_loaded_hip_torque,loaded_subjects_toe_off,bi_loaded_indices)
paretofront_bi_noload_knee_torque_max_change_phases = utils.paretofront_profiles_change_phases(bi_noload_knee_torque,ideal_bi_noload_knee_torque,noload_subjects_toe_off,bi_noload_indices)
paretofront_bi_loaded_knee_torque_max_change_phases = utils.paretofront_profiles_change_phases(bi_loaded_knee_torque,ideal_bi_loaded_knee_torque,loaded_subjects_toe_off,bi_loaded_indices)
biarticular_torque_change_dict = {'loaded hip torque': paretofront_bi_loaded_hip_torque_max_change_phases,'loaded knee torque': paretofront_bi_loaded_knee_torque_max_change_phases,\
                                 'noload hip torque': paretofront_bi_noload_hip_torque_max_change_phases,'noload knee torque': paretofront_bi_noload_knee_torque_max_change_phases}
# monoarticular torque profiles
paretofront_mono_noload_hip_torque_max_change_phases = utils.paretofront_profiles_change_phases(mono_noload_hip_torque,ideal_mono_noload_hip_torque,noload_subjects_toe_off,mono_noload_indices)
paretofront_mono_loaded_hip_torque_max_change_phases = utils.paretofront_profiles_change_phases(mono_loaded_hip_torque,ideal_mono_loaded_hip_torque,loaded_subjects_toe_off,mono_loaded_indices)
paretofront_mono_noload_knee_torque_max_change_phases = utils.paretofront_profiles_change_phases(mono_noload_knee_torque,ideal_mono_noload_knee_torque,noload_subjects_toe_off,mono_noload_indices)
paretofront_mono_loaded_knee_torque_max_change_phases = utils.paretofront_profiles_change_phases(mono_loaded_knee_torque,ideal_mono_loaded_knee_torque,loaded_subjects_toe_off,mono_loaded_indices)
monoarticular_torque_change_dict = {'loaded hip torque': paretofront_mono_loaded_hip_torque_max_change_phases,'loaded knee torque': paretofront_mono_loaded_knee_torque_max_change_phases,\
                                    'noload hip torque': paretofront_mono_noload_hip_torque_max_change_phases,'noload knee torque': paretofront_mono_noload_knee_torque_max_change_phases}

#####################################################################################
# Noload biarticular profile
gait_phase_row = ['loading response','loading response','mid stance','mid stance','terminal stance','terminal stance',\
                  'pre swing','pre swing','initial swing','initial swing','mid swing','mid swing','terminal swing','terminal swing']*len(bi_noload_indices)
                  
config_row =  [item for item in bi_noload_indices for i in range(2*len(['loading response','mid stance','terminal stance','pre swing',\
               'initial swing','mid swing','terminal swing']))]
mean_std_row = ['mean','std']*int(len(config_row)/2)
#------------------------------------------------------------------------------------
rows = np.row_stack((gait_phase_row,config_row))
rows = np.row_stack((mean_std_row,rows))
rows = np.column_stack((['-','-','-'],rows))
rows = np.column_stack((['-','-','-'],rows))
rows = np.column_stack((['-','-','-'],rows))
#------------------------------------------------------------------------------------
# power profiles
paretofront_bi_noload_hip_power_max_change_phases_array = np.array(list(paretofront_bi_noload_hip_power_max_change_phases.items()))[:,1]
noload_bi_hip_power_dataset = ['noload','biarticular','hip power'] + paretofront_bi_noload_hip_power_max_change_phases_array.tolist()
paretofront_bi_noload_knee_power_max_change_phases_array =np.array(list(paretofront_bi_noload_knee_power_max_change_phases.items()))[:,1]
noload_bi_knee_power_dataset = ['noload','biarticular','knee power'] + paretofront_bi_noload_knee_power_max_change_phases_array.tolist()
# torque profiles
paretofront_bi_noload_hip_torque_max_change_phases_array = np.array(list(paretofront_bi_noload_hip_torque_max_change_phases.items()))[:,1]
noload_bi_hip_torque_dataset = ['noload','biarticular','hip torque'] + paretofront_bi_noload_hip_torque_max_change_phases_array.tolist()
paretofront_bi_noload_knee_torque_max_change_phases_array =np.array(list(paretofront_bi_noload_knee_torque_max_change_phases.items()))[:,1]
noload_bi_knee_torque_dataset = ['noload','biarticular','knee torque'] + paretofront_bi_noload_knee_torque_max_change_phases_array.tolist()
#------------------------------------------------------------------------------------
# establish final dataset
bi_noload_dataset = np.row_stack((rows,noload_bi_hip_power_dataset))
bi_noload_dataset = np.row_stack((bi_noload_dataset,noload_bi_knee_power_dataset))
bi_noload_dataset = np.row_stack((bi_noload_dataset,noload_bi_hip_torque_dataset))
bi_noload_dataset = np.row_stack((bi_noload_dataset,noload_bi_knee_torque_dataset))
#------------------------------------------------------------------------------------
# write to csv
directory = r'.\Data\Pareto\Biarticular_Noload_Profiles_Quantitatives.csv'
np.savetxt(directory, bi_noload_dataset, fmt="%s", delimiter=",")
#####################################################################################
# Loaded biarticular profile
gait_phase_row = ['loading response','loading response','mid stance','mid stance','terminal stance','terminal stance',\
                  'pre swing','pre swing','initial swing','initial swing','mid swing','mid swing','terminal swing','terminal swing']*len(bi_loaded_indices)
                  
config_row =  [item for item in bi_loaded_indices for i in range(2*len(['loading response','mid stance','terminal stance','pre swing',\
               'initial swing','mid swing','terminal swing']))]
mean_std_row = ['mean','std']*int(len(config_row)/2)
#------------------------------------------------------------------------------------
rows = np.row_stack((gait_phase_row,config_row))
rows = np.row_stack((mean_std_row,rows))
rows = np.column_stack((['-','-','-'],rows))
rows = np.column_stack((['-','-','-'],rows))
rows = np.column_stack((['-','-','-'],rows))
#------------------------------------------------------------------------------------
# power profiles
paretofront_bi_loaded_hip_power_max_change_phases_array = np.array(list(paretofront_bi_loaded_hip_power_max_change_phases.items()))[:,1]
loaded_bi_hip_power_dataset = ['loaded','biarticular','hip power'] + paretofront_bi_loaded_hip_power_max_change_phases_array.tolist()
paretofront_bi_loaded_knee_power_max_change_phases_array =np.array(list(paretofront_bi_loaded_knee_power_max_change_phases.items()))[:,1]
loaded_bi_knee_power_dataset = ['loaded','biarticular','knee power'] + paretofront_bi_loaded_knee_power_max_change_phases_array.tolist()
# torque profiles
paretofront_bi_loaded_hip_torque_max_change_phases_array = np.array(list(paretofront_bi_loaded_hip_torque_max_change_phases.items()))[:,1]
loaded_bi_hip_torque_dataset = ['loaded','biarticular','hip torque'] + paretofront_bi_loaded_hip_torque_max_change_phases_array.tolist()
paretofront_bi_loaded_knee_torque_max_change_phases_array =np.array(list(paretofront_bi_loaded_knee_torque_max_change_phases.items()))[:,1]
loaded_bi_knee_torque_dataset = ['loaded','biarticular','knee torque'] + paretofront_bi_loaded_knee_torque_max_change_phases_array.tolist()
#------------------------------------------------------------------------------------
# establish final dataset
bi_loaded_dataset = np.row_stack((rows,loaded_bi_hip_power_dataset))
bi_loaded_dataset = np.row_stack((bi_loaded_dataset,loaded_bi_knee_power_dataset))
bi_loaded_dataset = np.row_stack((bi_loaded_dataset,loaded_bi_hip_torque_dataset))
bi_loaded_dataset = np.row_stack((bi_loaded_dataset,loaded_bi_knee_torque_dataset))
#------------------------------------------------------------------------------------
# write to csv
directory = r'.\Data\Pareto\Biarticular_Loaded_Profiles_Quantitatives.csv'
np.savetxt(directory, bi_loaded_dataset, fmt="%s", delimiter=",")
#####################################################################################
# Noload monoarticular profile
gait_phase_row = ['loading response','loading response','mid stance','mid stance','terminal stance','terminal stance',\
                  'pre swing','pre swing','initial swing','initial swing','mid swing','mid swing','terminal swing','terminal swing']*len(mono_noload_indices)
                  
config_row =  [item for item in mono_noload_indices for i in range(2*len(['loading response','mid stance','terminal stance','pre swing',\
               'initial swing','mid swing','terminal swing']))]
mean_std_row = ['mean','std']*int(len(config_row)/2)
#------------------------------------------------------------------------------------
rows = np.row_stack((gait_phase_row,config_row))
rows = np.row_stack((mean_std_row,rows))
rows = np.column_stack((['-','-','-'],rows))
rows = np.column_stack((['-','-','-'],rows))
rows = np.column_stack((['-','-','-'],rows))
#------------------------------------------------------------------------------------
# power profiles
paretofront_mono_noload_hip_power_max_change_phases_array = np.array(list(paretofront_mono_noload_hip_power_max_change_phases.items()))[:,1]
noload_mono_hip_power_dataset = ['noload','monoarticular','hip power'] + paretofront_mono_noload_hip_power_max_change_phases_array.tolist()
paretofront_mono_noload_knee_power_max_change_phases_array =np.array(list(paretofront_mono_noload_knee_power_max_change_phases.items()))[:,1]
noload_mono_knee_power_dataset = ['noload','monoarticular','knee power'] + paretofront_mono_noload_knee_power_max_change_phases_array.tolist()
# torque profiles
paretofront_mono_noload_hip_torque_max_change_phases_array = np.array(list(paretofront_mono_noload_hip_torque_max_change_phases.items()))[:,1]
noload_mono_hip_torque_dataset = ['noload','monoarticular','hip torque'] + paretofront_mono_noload_hip_torque_max_change_phases_array.tolist()
paretofront_mono_noload_knee_torque_max_change_phases_array =np.array(list(paretofront_mono_noload_knee_torque_max_change_phases.items()))[:,1]
noload_mono_knee_torque_dataset = ['noload','monoarticular','knee torque'] + paretofront_mono_noload_knee_torque_max_change_phases_array.tolist()
#------------------------------------------------------------------------------------
# establish final dataset
mono_noload_dataset = np.row_stack((rows,noload_mono_hip_power_dataset))
mono_noload_dataset = np.row_stack((mono_noload_dataset,noload_mono_knee_power_dataset))
mono_noload_dataset = np.row_stack((mono_noload_dataset,noload_mono_hip_torque_dataset))
mono_noload_dataset = np.row_stack((mono_noload_dataset,noload_mono_knee_torque_dataset))
#------------------------------------------------------------------------------------
# write to csv
directory = r'.\Data\Pareto\Monoarticular_Noload_Profiles_Quantitatives.csv'
np.savetxt(directory, mono_noload_dataset, fmt="%s", delimiter=",")
#####################################################################################
# Loaded monoarticular profile
gait_phase_row = ['loading response','loading response','mid stance','mid stance','terminal stance','terminal stance',\
                  'pre swing','pre swing','initial swing','initial swing','mid swing','mid swing','terminal swing','terminal swing']*len(mono_loaded_indices)
                  
config_row =  [item for item in mono_loaded_indices for i in range(2*len(['loading response','mid stance','terminal stance','pre swing',\
               'initial swing','mid swing','terminal swing']))]
mean_std_row = ['mean','std']*int(len(config_row)/2)
#------------------------------------------------------------------------------------
rows = np.row_stack((gait_phase_row,config_row))
rows = np.row_stack((mean_std_row,rows))
rows = np.column_stack((['-','-','-'],rows))
rows = np.column_stack((['-','-','-'],rows))
rows = np.column_stack((['-','-','-'],rows))
#------------------------------------------------------------------------------------
# power profiles
paretofront_mono_loaded_hip_power_max_change_phases_array = np.array(list(paretofront_mono_loaded_hip_power_max_change_phases.items()))[:,1]
loaded_mono_hip_power_dataset = ['loaded','monoarticular','hip power'] + paretofront_mono_loaded_hip_power_max_change_phases_array.tolist()
paretofront_mono_loaded_knee_power_max_change_phases_array =np.array(list(paretofront_mono_loaded_knee_power_max_change_phases.items()))[:,1]
loaded_mono_knee_power_dataset = ['loaded','monoarticular','knee power'] + paretofront_mono_loaded_knee_power_max_change_phases_array.tolist()
# torque profiles
paretofront_mono_loaded_hip_torque_max_change_phases_array = np.array(list(paretofront_mono_loaded_hip_torque_max_change_phases.items()))[:,1]
loaded_mono_hip_torque_dataset = ['loaded','monoarticular','hip torque'] + paretofront_mono_loaded_hip_torque_max_change_phases_array.tolist()
paretofront_mono_loaded_knee_torque_max_change_phases_array =np.array(list(paretofront_mono_loaded_knee_torque_max_change_phases.items()))[:,1]
loaded_mono_knee_torque_dataset = ['loaded','monoarticular','knee torque'] + paretofront_mono_loaded_knee_torque_max_change_phases_array.tolist()
#------------------------------------------------------------------------------------
# establish final dataset
mono_loaded_dataset = np.row_stack((rows,loaded_mono_hip_power_dataset))
mono_loaded_dataset = np.row_stack((mono_loaded_dataset,loaded_mono_knee_power_dataset))
mono_loaded_dataset = np.row_stack((mono_loaded_dataset,loaded_mono_hip_torque_dataset))
mono_loaded_dataset = np.row_stack((mono_loaded_dataset,loaded_mono_knee_torque_dataset))
#------------------------------------------------------------------------------------
# write to csv
directory = r'.\Data\Pareto\Monoarticular_Loaded_Profiles_Quantitatives.csv'
np.savetxt(directory, mono_loaded_dataset, fmt="%s", delimiter=",")
