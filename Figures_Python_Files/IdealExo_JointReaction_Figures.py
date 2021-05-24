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
# Reading CSV files into a dictionary and constructing gls
jrf_dataset = utils.csv2numpy('.\Data\RRA\jrf_final_data.csv') 
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

# gls
gl_noload = {'noload_subject{}_trial{}'.format(i,j): utils.construct_gl_mass_side(subjectno=i,trialno=j,loadcond='noload') for i in subjects for j in trials_num}
gl_loaded = {'loaded_subject{}_trial{}'.format(i,j): utils.construct_gl_mass_side(subjectno=i,trialno=j,loadcond='loaded') for i in subjects for j in trials_num}
#####################################################################################
# Processing Data
# toe-off
noload_mean_toe_off,_,loaded_mean_toe_off,_ = utils.toe_off_avg_std(gl_noload,gl_loaded)
# reaction moment of assisted subjects 
# back joint
# biarticular
bi_loaded_back_RMz = utils.normalize_direction_data(bi_loaded_RM_dictionary['back_joint_Mz'],gl_noload,direction=False)
bi_noload_back_RMz = utils.normalize_direction_data(bi_noload_RM_dictionary['back_joint_Mz'],gl_noload,direction=False)
mean_bi_loaded_back_RMz,std_bi_loaded_back_RMz = utils.mean_std_over_subjects(bi_loaded_back_RMz,avg_trials=False)
mean_bi_noload_back_RMz,std_bi_noload_back_RMz = utils.mean_std_over_subjects(bi_noload_back_RMz,avg_trials=False)
# monoarticular
mono_loaded_back_RMz = utils.normalize_direction_data(mono_loaded_RM_dictionary['back_joint_Mz'],gl_noload,direction=False)
mono_noload_back_RMz = utils.normalize_direction_data(mono_noload_RM_dictionary['back_joint_Mz'],gl_noload,direction=False)
mean_mono_loaded_back_RMz,std_mono_loaded_back_RMz = utils.mean_std_over_subjects(mono_loaded_back_RMz,avg_trials=False)
mean_mono_noload_back_RMz,std_mono_noload_back_RMz = utils.mean_std_over_subjects(mono_noload_back_RMz,avg_trials=False)
# duct tape joint
# biarticular
bi_loaded_duct_tape_RMz = utils.normalize_direction_data(bi_loaded_RM_dictionary['duct_tape_joint_Mz'],gl_noload,direction=False)
mean_bi_loaded_duct_tape_RMz,std_bi_loaded_duct_tape_RMz = utils.mean_std_over_subjects(bi_loaded_duct_tape_RMz,avg_trials=False)
# monoarticular
mono_loaded_duct_tape_RMz = utils.normalize_direction_data(mono_loaded_RM_dictionary['duct_tape_joint_Mz'],gl_noload,direction=False)
mean_mono_loaded_duct_tape_RMz,std_mono_loaded_duct_tape_RMz = utils.mean_std_over_subjects(mono_loaded_duct_tape_RMz,avg_trials=False)
# hip joint
# biarticular
bi_loaded_hip_RMz = utils.normalize_direction_data(bi_loaded_RM_dictionary['hip_joint_Mz'],gl_noload,direction=False)
bi_noload_hip_RMz = utils.normalize_direction_data(bi_noload_RM_dictionary['hip_joint_Mz'],gl_noload,direction=False)
mean_bi_loaded_hip_RMz,std_bi_loaded_hip_RMz = utils.mean_std_over_subjects(bi_loaded_hip_RMz,avg_trials=False)
mean_bi_noload_hip_RMz,std_bi_noload_hip_RMz = utils.mean_std_over_subjects(bi_noload_hip_RMz,avg_trials=False)
# Fx
bi_loaded_hip_RFx = utils.normalize_direction_data(bi_loaded_RF_dictionary['hip_joint_Fx'],gl_noload,direction=False)
bi_noload_hip_RFx = utils.normalize_direction_data(bi_noload_RF_dictionary['hip_joint_Fx'],gl_noload,direction=False)
mean_bi_loaded_hip_RFx,std_bi_loaded_hip_RFx = utils.mean_std_over_subjects(bi_loaded_hip_RFx,avg_trials=False)
mean_bi_noload_hip_RFx,std_bi_noload_hip_RFx = utils.mean_std_over_subjects(bi_noload_hip_RFx,avg_trials=False)
# Fy
bi_loaded_hip_RFy = utils.normalize_direction_data(bi_loaded_RF_dictionary['hip_joint_Fy'],gl_noload,direction=False)
bi_noload_hip_RFy = utils.normalize_direction_data(bi_noload_RF_dictionary['hip_joint_Fy'],gl_noload,direction=False)
mean_bi_loaded_hip_RFy,std_bi_loaded_hip_RFy = utils.mean_std_over_subjects(bi_loaded_hip_RFy,avg_trials=False)
mean_bi_noload_hip_RFy,std_bi_noload_hip_RFy = utils.mean_std_over_subjects(bi_noload_hip_RFy,avg_trials=False)
# Fz
bi_loaded_hip_RFz = utils.normalize_direction_data(bi_loaded_RF_dictionary['hip_joint_Fz'],gl_noload,direction=False)
bi_noload_hip_RFz = utils.normalize_direction_data(bi_noload_RF_dictionary['hip_joint_Fz'],gl_noload,direction=False)
mean_bi_loaded_hip_RFz,std_bi_loaded_hip_RFz = utils.mean_std_over_subjects(bi_loaded_hip_RFz,avg_trials=False)
mean_bi_noload_hip_RFz,std_bi_noload_hip_RFz = utils.mean_std_over_subjects(bi_noload_hip_RFz,avg_trials=False)

# monoarticular
mono_loaded_hip_RMz = utils.normalize_direction_data(mono_loaded_RM_dictionary['hip_joint_Mz'],gl_noload,direction=False)
mono_noload_hip_RMz = utils.normalize_direction_data(mono_noload_RM_dictionary['hip_joint_Mz'],gl_noload,direction=False)
mean_mono_loaded_hip_RMz,std_mono_loaded_hip_RMz = utils.mean_std_over_subjects(mono_loaded_hip_RMz,avg_trials=False)
mean_mono_noload_hip_RMz,std_mono_noload_hip_RMz = utils.mean_std_over_subjects(mono_noload_hip_RMz,avg_trials=False)
# Fx
mono_loaded_hip_RFx = utils.normalize_direction_data(mono_loaded_RF_dictionary['hip_joint_Fx'],gl_noload,direction=False)
mono_noload_hip_RFx = utils.normalize_direction_data(mono_noload_RF_dictionary['hip_joint_Fx'],gl_noload,direction=False)
mean_mono_loaded_hip_RFx,std_mono_loaded_hip_RFx = utils.mean_std_over_subjects(mono_loaded_hip_RFx,avg_trials=False)
mean_mono_noload_hip_RFx,std_mono_noload_hip_RFx = utils.mean_std_over_subjects(mono_noload_hip_RFx,avg_trials=False)
# Fy
mono_loaded_hip_RFy = utils.normalize_direction_data(mono_loaded_RF_dictionary['hip_joint_Fy'],gl_noload,direction=False)
mono_noload_hip_RFy = utils.normalize_direction_data(mono_noload_RF_dictionary['hip_joint_Fy'],gl_noload,direction=False)
mean_mono_loaded_hip_RFy,std_mono_loaded_hip_RFy = utils.mean_std_over_subjects(mono_loaded_hip_RFy,avg_trials=False)
mean_mono_noload_hip_RFy,std_mono_noload_hip_RFy = utils.mean_std_over_subjects(mono_noload_hip_RFy,avg_trials=False)
# Fz
mono_loaded_hip_RFz = utils.normalize_direction_data(mono_loaded_RF_dictionary['hip_joint_Fz'],gl_noload,direction=False)
mono_noload_hip_RFz = utils.normalize_direction_data(mono_noload_RF_dictionary['hip_joint_Fz'],gl_noload,direction=False)
mean_mono_loaded_hip_RFz,std_mono_loaded_hip_RFz = utils.mean_std_over_subjects(mono_loaded_hip_RFz,avg_trials=False)
mean_mono_noload_hip_RFz,std_mono_noload_hip_RFz = utils.mean_std_over_subjects(mono_noload_hip_RFz,avg_trials=False)

# knee joint
## MX ##
# biarticular
bi_loaded_knee_RMx = utils.normalize_direction_data(bi_loaded_RM_dictionary['knee_joint_Mx'],gl_noload,direction=False)
bi_noload_knee_RMx = utils.normalize_direction_data(bi_noload_RM_dictionary['knee_joint_Mx'],gl_noload,direction=False)
mean_bi_loaded_knee_RMx,std_bi_loaded_knee_RMx = utils.mean_std_over_subjects(bi_loaded_knee_RMx,avg_trials=False)
mean_bi_noload_knee_RMx,std_bi_noload_knee_RMx = utils.mean_std_over_subjects(bi_noload_knee_RMx,avg_trials=False)
# monoarticular
mono_loaded_knee_RMx = utils.normalize_direction_data(mono_loaded_RM_dictionary['knee_joint_Mx'],gl_noload,direction=False)
mono_noload_knee_RMx = utils.normalize_direction_data(mono_noload_RM_dictionary['knee_joint_Mx'],gl_noload,direction=False)
mean_mono_loaded_knee_RMx,std_mono_loaded_knee_RMx = utils.mean_std_over_subjects(mono_loaded_knee_RMx,avg_trials=False)
mean_mono_noload_knee_RMx,std_mono_noload_knee_RMx = utils.mean_std_over_subjects(mono_noload_knee_RMx,avg_trials=False)
## MY ##
# biarticular
bi_loaded_knee_RMy = utils.normalize_direction_data(bi_loaded_RM_dictionary['knee_joint_My'],gl_noload,direction=False)
bi_noload_knee_RMy = utils.normalize_direction_data(bi_noload_RM_dictionary['knee_joint_My'],gl_noload,direction=False)
mean_bi_loaded_knee_RMy,std_bi_loaded_knee_RMy = utils.mean_std_over_subjects(bi_loaded_knee_RMy,avg_trials=False)
mean_bi_noload_knee_RMy,std_bi_noload_knee_RMy = utils.mean_std_over_subjects(bi_noload_knee_RMy,avg_trials=False)
# monoarticular
mono_loaded_knee_RMy = utils.normalize_direction_data(mono_loaded_RM_dictionary['knee_joint_My'],gl_noload,direction=False)
mono_noload_knee_RMy = utils.normalize_direction_data(mono_noload_RM_dictionary['knee_joint_My'],gl_noload,direction=False)
mean_mono_loaded_knee_RMy,std_mono_loaded_knee_RMy = utils.mean_std_over_subjects(mono_loaded_knee_RMy,avg_trials=False)
mean_mono_noload_knee_RMy,std_mono_noload_knee_RMy = utils.mean_std_over_subjects(mono_noload_knee_RMy,avg_trials=False)
## MZ ##
# biarticular
bi_loaded_knee_RMz = utils.normalize_direction_data(bi_loaded_RM_dictionary['knee_joint_Mz'],gl_noload,direction=False)
bi_noload_knee_RMz = utils.normalize_direction_data(bi_noload_RM_dictionary['knee_joint_Mz'],gl_noload,direction=False)
mean_bi_loaded_knee_RMz,std_bi_loaded_knee_RMz = utils.mean_std_over_subjects(bi_loaded_knee_RMz,avg_trials=False)
mean_bi_noload_knee_RMz,std_bi_noload_knee_RMz = utils.mean_std_over_subjects(bi_noload_knee_RMz,avg_trials=False)
# monoarticular
mono_loaded_knee_RMz = utils.normalize_direction_data(mono_loaded_RM_dictionary['knee_joint_Mz'],gl_noload,direction=False)
mono_noload_knee_RMz = utils.normalize_direction_data(mono_noload_RM_dictionary['knee_joint_Mz'],gl_noload,direction=False)
mean_mono_loaded_knee_RMz,std_mono_loaded_knee_RMz = utils.mean_std_over_subjects(mono_loaded_knee_RMz,avg_trials=False)
mean_mono_noload_knee_RMz,std_mono_noload_knee_RMz = utils.mean_std_over_subjects(mono_noload_knee_RMz,avg_trials=False)
# knee joint
## FX ##
# biarticular
bi_loaded_knee_RFx = utils.normalize_direction_data(bi_loaded_RF_dictionary['knee_joint_Fx'],gl_noload,direction=False)
bi_noload_knee_RFx = utils.normalize_direction_data(bi_noload_RF_dictionary['knee_joint_Fx'],gl_noload,direction=False)
mean_bi_loaded_knee_RFx,std_bi_loaded_knee_RFx = utils.mean_std_over_subjects(bi_loaded_knee_RFx,avg_trials=False)
mean_bi_noload_knee_RFx,std_bi_noload_knee_RFx = utils.mean_std_over_subjects(bi_noload_knee_RFx,avg_trials=False)
# monoarticular
mono_loaded_knee_RFx = utils.normalize_direction_data(mono_loaded_RF_dictionary['knee_joint_Fx'],gl_noload,direction=False)
mono_noload_knee_RFx = utils.normalize_direction_data(mono_noload_RF_dictionary['knee_joint_Fx'],gl_noload,direction=False)
mean_mono_loaded_knee_RFx,std_mono_loaded_knee_RFx = utils.mean_std_over_subjects(mono_loaded_knee_RFx,avg_trials=False)
mean_mono_noload_knee_RFx,std_mono_noload_knee_RFx = utils.mean_std_over_subjects(mono_noload_knee_RFx,avg_trials=False)
## FY ##
# biarticular
bi_loaded_knee_RFy = utils.normalize_direction_data(bi_loaded_RF_dictionary['knee_joint_Fy'],gl_noload,direction=False)
bi_noload_knee_RFy = utils.normalize_direction_data(bi_noload_RF_dictionary['knee_joint_Fy'],gl_noload,direction=False)
mean_bi_loaded_knee_RFy,std_bi_loaded_knee_RFy = utils.mean_std_over_subjects(bi_loaded_knee_RFy,avg_trials=False)
mean_bi_noload_knee_RFy,std_bi_noload_knee_RFy = utils.mean_std_over_subjects(bi_noload_knee_RFy,avg_trials=False)
# monoarticular
mono_loaded_knee_RFy = utils.normalize_direction_data(mono_loaded_RF_dictionary['knee_joint_Fy'],gl_noload,direction=False)
mono_noload_knee_RFy = utils.normalize_direction_data(mono_noload_RF_dictionary['knee_joint_Fy'],gl_noload,direction=False)
mean_mono_loaded_knee_RFy,std_mono_loaded_knee_RFy = utils.mean_std_over_subjects(mono_loaded_knee_RFy,avg_trials=False)
mean_mono_noload_knee_RFy,std_mono_noload_knee_RFy = utils.mean_std_over_subjects(mono_noload_knee_RFy,avg_trials=False)
## MZ ##
# biarticular
bi_loaded_knee_RFz = utils.normalize_direction_data(bi_loaded_RF_dictionary['knee_joint_Fz'],gl_noload,direction=False)
bi_noload_knee_RFz = utils.normalize_direction_data(bi_noload_RF_dictionary['knee_joint_Fz'],gl_noload,direction=False)
mean_bi_loaded_knee_RFz,std_bi_loaded_knee_RFz = utils.mean_std_over_subjects(bi_loaded_knee_RFz,avg_trials=False)
mean_bi_noload_knee_RFz,std_bi_noload_knee_RFz = utils.mean_std_over_subjects(bi_noload_knee_RFz,avg_trials=False)
# monoarticular
mono_loaded_knee_RFz = utils.normalize_direction_data(mono_loaded_RF_dictionary['knee_joint_Fz'],gl_noload,direction=False)
mono_noload_knee_RFz = utils.normalize_direction_data(mono_noload_RF_dictionary['knee_joint_Fz'],gl_noload,direction=False)
mean_mono_loaded_knee_RFz,std_mono_loaded_knee_RFz = utils.mean_std_over_subjects(mono_loaded_knee_RFz,avg_trials=False)
mean_mono_noload_knee_RFz,std_mono_noload_knee_RFz = utils.mean_std_over_subjects(mono_noload_knee_RFz,avg_trials=False)

# patellofemoral joint
## MX ##
# biarticular
bi_loaded_patellofemoral_RMx = utils.normalize_direction_data(bi_loaded_RM_dictionary['patellofemoral_joint_Mx'],gl_noload,direction=False)
bi_noload_patellofemoral_RMx = utils.normalize_direction_data(bi_noload_RM_dictionary['patellofemoral_joint_Mx'],gl_noload,direction=False)
mean_bi_loaded_patellofemoral_RMx,std_bi_loaded_patellofemoral_RMx = utils.mean_std_over_subjects(bi_loaded_patellofemoral_RMx,avg_trials=False)
mean_bi_noload_patellofemoral_RMx,std_bi_noload_patellofemoral_RMx = utils.mean_std_over_subjects(bi_noload_patellofemoral_RMx,avg_trials=False)
# monoarticular
mono_loaded_patellofemoral_RMx = utils.normalize_direction_data(mono_loaded_RM_dictionary['patellofemoral_joint_Mx'],gl_noload,direction=False)
mono_noload_patellofemoral_RMx = utils.normalize_direction_data(mono_noload_RM_dictionary['patellofemoral_joint_Mx'],gl_noload,direction=False)
mean_mono_loaded_patellofemoral_RMx,std_mono_loaded_patellofemoral_RMx = utils.mean_std_over_subjects(mono_loaded_patellofemoral_RMx,avg_trials=False)
mean_mono_noload_patellofemoral_RMx,std_mono_noload_patellofemoral_RMx = utils.mean_std_over_subjects(mono_noload_patellofemoral_RMx,avg_trials=False)
## MY ##
# biarticular
bi_loaded_patellofemoral_RMy = utils.normalize_direction_data(bi_loaded_RM_dictionary['patellofemoral_joint_My'],gl_noload,direction=False)
bi_noload_patellofemoral_RMy = utils.normalize_direction_data(bi_noload_RM_dictionary['patellofemoral_joint_My'],gl_noload,direction=False)
mean_bi_loaded_patellofemoral_RMy,std_bi_loaded_patellofemoral_RMy = utils.mean_std_over_subjects(bi_loaded_patellofemoral_RMy,avg_trials=False)
mean_bi_noload_patellofemoral_RMy,std_bi_noload_patellofemoral_RMy = utils.mean_std_over_subjects(bi_noload_patellofemoral_RMy,avg_trials=False)
# monoarticular
mono_loaded_patellofemoral_RMy = utils.normalize_direction_data(mono_loaded_RM_dictionary['patellofemoral_joint_My'],gl_noload,direction=False)
mono_noload_patellofemoral_RMy = utils.normalize_direction_data(mono_noload_RM_dictionary['patellofemoral_joint_My'],gl_noload,direction=False)
mean_mono_loaded_patellofemoral_RMy,std_mono_loaded_patellofemoral_RMy = utils.mean_std_over_subjects(mono_loaded_patellofemoral_RMy,avg_trials=False)
mean_mono_noload_patellofemoral_RMy,std_mono_noload_patellofemoral_RMy = utils.mean_std_over_subjects(mono_noload_patellofemoral_RMy,avg_trials=False)
## Mz ##
# biarticular
bi_loaded_patellofemoral_RMz = utils.normalize_direction_data(bi_loaded_RM_dictionary['patellofemoral_joint_Mz'],gl_noload,direction=False)
bi_noload_patellofemoral_RMz = utils.normalize_direction_data(bi_noload_RM_dictionary['patellofemoral_joint_Mz'],gl_noload,direction=False)
mean_bi_loaded_patellofemoral_RMz,std_bi_loaded_patellofemoral_RMz = utils.mean_std_over_subjects(bi_loaded_patellofemoral_RMz,avg_trials=False)
mean_bi_noload_patellofemoral_RMz,std_bi_noload_patellofemoral_RMz = utils.mean_std_over_subjects(bi_noload_patellofemoral_RMz,avg_trials=False)
# monoarticular
mono_loaded_patellofemoral_RMz = utils.normalize_direction_data(mono_loaded_RM_dictionary['patellofemoral_joint_Mz'],gl_noload,direction=False)
mono_noload_patellofemoral_RMz = utils.normalize_direction_data(mono_noload_RM_dictionary['patellofemoral_joint_Mz'],gl_noload,direction=False)
mean_mono_loaded_patellofemoral_RMz,std_mono_loaded_patellofemoral_RMz = utils.mean_std_over_subjects(mono_loaded_patellofemoral_RMz,avg_trials=False)
mean_mono_noload_patellofemoral_RMz,std_mono_noload_patellofemoral_RMz = utils.mean_std_over_subjects(mono_noload_patellofemoral_RMz,avg_trials=False)
# patellofemoral joint
## FX ##
# biarticular
bi_loaded_patellofemoral_RFx = utils.normalize_direction_data(bi_loaded_RF_dictionary['patellofemoral_joint_Fx'],gl_noload,direction=False)
bi_noload_patellofemoral_RFx = utils.normalize_direction_data(bi_noload_RF_dictionary['patellofemoral_joint_Fx'],gl_noload,direction=False)
mean_bi_loaded_patellofemoral_RFx,std_bi_loaded_patellofemoral_RFx = utils.mean_std_over_subjects(bi_loaded_patellofemoral_RFx,avg_trials=False)
mean_bi_noload_patellofemoral_RFx,std_bi_noload_patellofemoral_RFx = utils.mean_std_over_subjects(bi_noload_patellofemoral_RFx,avg_trials=False)
# monoarticular
mono_loaded_patellofemoral_RFx = utils.normalize_direction_data(mono_loaded_RF_dictionary['patellofemoral_joint_Fx'],gl_noload,direction=False)
mono_noload_patellofemoral_RFx = utils.normalize_direction_data(mono_noload_RF_dictionary['patellofemoral_joint_Fx'],gl_noload,direction=False)
mean_mono_loaded_patellofemoral_RFx,std_mono_loaded_patellofemoral_RFx = utils.mean_std_over_subjects(mono_loaded_patellofemoral_RFx,avg_trials=False)
mean_mono_noload_patellofemoral_RFx,std_mono_noload_patellofemoral_RFx = utils.mean_std_over_subjects(mono_noload_patellofemoral_RFx,avg_trials=False)
## FY ##
# biarticular
bi_loaded_patellofemoral_RFy = utils.normalize_direction_data(bi_loaded_RF_dictionary['patellofemoral_joint_Fy'],gl_noload,direction=False)
bi_noload_patellofemoral_RFy = utils.normalize_direction_data(bi_noload_RF_dictionary['patellofemoral_joint_Fy'],gl_noload,direction=False)
mean_bi_loaded_patellofemoral_RFy,std_bi_loaded_patellofemoral_RFy = utils.mean_std_over_subjects(bi_loaded_patellofemoral_RFy,avg_trials=False)
mean_bi_noload_patellofemoral_RFy,std_bi_noload_patellofemoral_RFy = utils.mean_std_over_subjects(bi_noload_patellofemoral_RFy,avg_trials=False)
# monoarticular
mono_loaded_patellofemoral_RFy = utils.normalize_direction_data(mono_loaded_RF_dictionary['patellofemoral_joint_Fy'],gl_noload,direction=False)
mono_noload_patellofemoral_RFy = utils.normalize_direction_data(mono_noload_RF_dictionary['patellofemoral_joint_Fy'],gl_noload,direction=False)
mean_mono_loaded_patellofemoral_RFy,std_mono_loaded_patellofemoral_RFy = utils.mean_std_over_subjects(mono_loaded_patellofemoral_RFy,avg_trials=False)
mean_mono_noload_patellofemoral_RFy,std_mono_noload_patellofemoral_RFy = utils.mean_std_over_subjects(mono_noload_patellofemoral_RFy,avg_trials=False)
## Fz ##
# biarticular
bi_loaded_patellofemoral_RFz = utils.normalize_direction_data(bi_loaded_RF_dictionary['patellofemoral_joint_Fz'],gl_noload,direction=False)
bi_noload_patellofemoral_RFz = utils.normalize_direction_data(bi_noload_RF_dictionary['patellofemoral_joint_Fz'],gl_noload,direction=False)
mean_bi_loaded_patellofemoral_RFz,std_bi_loaded_patellofemoral_RFz = utils.mean_std_over_subjects(bi_loaded_patellofemoral_RFz,avg_trials=False)
mean_bi_noload_patellofemoral_RFz,std_bi_noload_patellofemoral_RFz = utils.mean_std_over_subjects(bi_noload_patellofemoral_RFz,avg_trials=False)
# monoarticular
mono_loaded_patellofemoral_RFz = utils.normalize_direction_data(mono_loaded_RF_dictionary['patellofemoral_joint_Fz'],gl_noload,direction=False)
mono_noload_patellofemoral_RFz = utils.normalize_direction_data(mono_noload_RF_dictionary['patellofemoral_joint_Fz'],gl_noload,direction=False)
mean_mono_loaded_patellofemoral_RFz,std_mono_loaded_patellofemoral_RFz = utils.mean_std_over_subjects(mono_loaded_patellofemoral_RFz,avg_trials=False)
mean_mono_noload_patellofemoral_RFz,std_mono_noload_patellofemoral_RFz = utils.mean_std_over_subjects(mono_noload_patellofemoral_RFz,avg_trials=False)

# ankle joint
# biarticular
# Mx
bi_loaded_ankle_RMx = utils.normalize_direction_data(bi_loaded_RM_dictionary['ankle_joint_Mx'],gl_noload,direction=False)
bi_noload_ankle_RMx = utils.normalize_direction_data(bi_noload_RM_dictionary['ankle_joint_Mx'],gl_noload,direction=False)
mean_bi_loaded_ankle_RMx,std_bi_loaded_ankle_RMx = utils.mean_std_over_subjects(bi_loaded_ankle_RMx,avg_trials=False)
mean_bi_noload_ankle_RMx,std_bi_noload_ankle_RMx = utils.mean_std_over_subjects(bi_noload_ankle_RMx,avg_trials=False)
# My
bi_loaded_ankle_RMy = utils.normalize_direction_data(bi_loaded_RM_dictionary['ankle_joint_My'],gl_noload,direction=False)
bi_noload_ankle_RMy = utils.normalize_direction_data(bi_noload_RM_dictionary['ankle_joint_My'],gl_noload,direction=False)
mean_bi_loaded_ankle_RMy,std_bi_loaded_ankle_RMy = utils.mean_std_over_subjects(bi_loaded_ankle_RMy,avg_trials=False)
mean_bi_noload_ankle_RMy,std_bi_noload_ankle_RMy = utils.mean_std_over_subjects(bi_noload_ankle_RMy,avg_trials=False)
# Mz
bi_loaded_ankle_RMz = utils.normalize_direction_data(bi_loaded_RM_dictionary['ankle_joint_Mz'],gl_noload,direction=False)
bi_noload_ankle_RMz = utils.normalize_direction_data(bi_noload_RM_dictionary['ankle_joint_Mz'],gl_noload,direction=False)
mean_bi_loaded_ankle_RMz,std_bi_loaded_ankle_RMz = utils.mean_std_over_subjects(bi_loaded_ankle_RMz,avg_trials=False)
mean_bi_noload_ankle_RMz,std_bi_noload_ankle_RMz = utils.mean_std_over_subjects(bi_noload_ankle_RMz,avg_trials=False)
# Fx
bi_loaded_ankle_RFx = utils.normalize_direction_data(bi_loaded_RF_dictionary['ankle_joint_Fx'],gl_noload,direction=False)
bi_noload_ankle_RFx = utils.normalize_direction_data(bi_noload_RF_dictionary['ankle_joint_Fx'],gl_noload,direction=False)
mean_bi_loaded_ankle_RFx,std_bi_loaded_ankle_RFx = utils.mean_std_over_subjects(bi_loaded_ankle_RFx,avg_trials=False)
mean_bi_noload_ankle_RFx,std_bi_noload_ankle_RFx = utils.mean_std_over_subjects(bi_noload_ankle_RFx,avg_trials=False)
# Fy
bi_loaded_ankle_RFy = utils.normalize_direction_data(bi_loaded_RF_dictionary['ankle_joint_Fy'],gl_noload,direction=False)
bi_noload_ankle_RFy = utils.normalize_direction_data(bi_noload_RF_dictionary['ankle_joint_Fy'],gl_noload,direction=False)
mean_bi_loaded_ankle_RFy,std_bi_loaded_ankle_RFy = utils.mean_std_over_subjects(bi_loaded_ankle_RFy,avg_trials=False)
mean_bi_noload_ankle_RFy,std_bi_noload_ankle_RFy = utils.mean_std_over_subjects(bi_noload_ankle_RFy,avg_trials=False)
# Fz
bi_loaded_ankle_RFz = utils.normalize_direction_data(bi_loaded_RF_dictionary['ankle_joint_Fz'],gl_noload,direction=False)
bi_noload_ankle_RFz = utils.normalize_direction_data(bi_noload_RF_dictionary['ankle_joint_Fz'],gl_noload,direction=False)
mean_bi_loaded_ankle_RFz,std_bi_loaded_ankle_RFz = utils.mean_std_over_subjects(bi_loaded_ankle_RFz,avg_trials=False)
mean_bi_noload_ankle_RFz,std_bi_noload_ankle_RFz = utils.mean_std_over_subjects(bi_noload_ankle_RFz,avg_trials=False)

# monoarticular
# Mx
mono_loaded_ankle_RMx = utils.normalize_direction_data(mono_loaded_RM_dictionary['ankle_joint_Mx'],gl_noload,direction=False)
mono_noload_ankle_RMx = utils.normalize_direction_data(mono_noload_RM_dictionary['ankle_joint_Mx'],gl_noload,direction=False)
mean_mono_loaded_ankle_RMx,std_mono_loaded_ankle_RMx = utils.mean_std_over_subjects(mono_loaded_ankle_RMx,avg_trials=False)
mean_mono_noload_ankle_RMx,std_mono_noload_ankle_RMx = utils.mean_std_over_subjects(mono_noload_ankle_RMx,avg_trials=False)
# My
mono_loaded_ankle_RMy = utils.normalize_direction_data(mono_loaded_RM_dictionary['ankle_joint_My'],gl_noload,direction=False)
mono_noload_ankle_RMy = utils.normalize_direction_data(mono_noload_RM_dictionary['ankle_joint_My'],gl_noload,direction=False)
mean_mono_loaded_ankle_RMy,std_mono_loaded_ankle_RMy = utils.mean_std_over_subjects(mono_loaded_ankle_RMy,avg_trials=False)
mean_mono_noload_ankle_RMy,std_mono_noload_ankle_RMy = utils.mean_std_over_subjects(mono_noload_ankle_RMy,avg_trials=False)
# Mz
mono_loaded_ankle_RMz = utils.normalize_direction_data(mono_loaded_RM_dictionary['ankle_joint_Mz'],gl_noload,direction=False)
mono_noload_ankle_RMz = utils.normalize_direction_data(mono_noload_RM_dictionary['ankle_joint_Mz'],gl_noload,direction=False)
mean_mono_loaded_ankle_RMz,std_mono_loaded_ankle_RMz = utils.mean_std_over_subjects(mono_loaded_ankle_RMz,avg_trials=False)
mean_mono_noload_ankle_RMz,std_mono_noload_ankle_RMz = utils.mean_std_over_subjects(mono_noload_ankle_RMz,avg_trials=False)
# Fx
mono_loaded_ankle_RFx = utils.normalize_direction_data(mono_loaded_RF_dictionary['ankle_joint_Fx'],gl_noload,direction=False)
mono_noload_ankle_RFx = utils.normalize_direction_data(mono_noload_RF_dictionary['ankle_joint_Fx'],gl_noload,direction=False)
mean_mono_loaded_ankle_RFx,std_mono_loaded_ankle_RFx = utils.mean_std_over_subjects(mono_loaded_ankle_RFx,avg_trials=False)
mean_mono_noload_ankle_RFx,std_mono_noload_ankle_RFx = utils.mean_std_over_subjects(mono_noload_ankle_RFx,avg_trials=False)
# Fy
mono_loaded_ankle_RFy = utils.normalize_direction_data(mono_loaded_RF_dictionary['ankle_joint_Fy'],gl_noload,direction=False)
mono_noload_ankle_RFy = utils.normalize_direction_data(mono_noload_RF_dictionary['ankle_joint_Fy'],gl_noload,direction=False)
mean_mono_loaded_ankle_RFy,std_mono_loaded_ankle_RFy = utils.mean_std_over_subjects(mono_loaded_ankle_RFy,avg_trials=False)
mean_mono_noload_ankle_RFy,std_mono_noload_ankle_RFy = utils.mean_std_over_subjects(mono_noload_ankle_RFy,avg_trials=False)
# Fz
mono_loaded_ankle_RFz = utils.normalize_direction_data(mono_loaded_RF_dictionary['ankle_joint_Fz'],gl_noload,direction=False)
mono_noload_ankle_RFz = utils.normalize_direction_data(mono_noload_RF_dictionary['ankle_joint_Fz'],gl_noload,direction=False)
mean_mono_loaded_ankle_RFz,std_mono_loaded_ankle_RFz = utils.mean_std_over_subjects(mono_loaded_ankle_RFz,avg_trials=False)
mean_mono_noload_ankle_RFz,std_mono_noload_ankle_RFz = utils.mean_std_over_subjects(mono_noload_ankle_RFz,avg_trials=False)

#####################################################################################
# Write final data to csv file.
# TODO: optimize data saving method.
# Headers
Headers = ['mean_bi_loaded_back_RMz,std_bi_loaded_back_RMz','mean_mono_loaded_back_RMz','std_mono_loaded_back_RMz',\
           'mean_bi_noload_back_RMz,std_bi_noload_back_RMz','mean_mono_noload_back_RMz','std_mono_noload_back_RMz',\
            # hip joint
           'mean_bi_loaded_hip_RFx','std_bi_loaded_hip_RFx','mean_mono_loaded_hip_RFx','std_mono_loaded_hip_RFx',\
           'mean_bi_noload_hip_RFx','std_bi_noload_hip_RFx','mean_mono_noload_hip_RFx','std_mono_noload_hip_RFx',\
           'mean_bi_loaded_hip_RFy','std_bi_loaded_hip_RFy','mean_mono_loaded_hip_RFy','std_mono_loaded_hip_RFy',\
           'mean_bi_noload_hip_RFy','std_bi_noload_hip_RFy','mean_mono_noload_hip_RFy','std_mono_noload_hip_RFy',\
           'mean_bi_loaded_hip_RFz','std_bi_loaded_hip_RFz','mean_mono_loaded_hip_RFz','std_mono_loaded_hip_RFz',\
           'mean_bi_noload_hip_RFz','std_bi_noload_hip_RFz','mean_mono_noload_hip_RFz','std_mono_noload_hip_RFz',\
            # knee joint
           'mean_bi_loaded_knee_RFx','std_bi_loaded_knee_RFx','mean_mono_loaded_knee_RFx','std_mono_loaded_knee_RFx',\
           'mean_bi_noload_knee_RFx','std_bi_noload_knee_RFx','mean_mono_noload_knee_RFx','std_mono_noload_knee_RFx',\
           'mean_bi_loaded_knee_RFy','std_bi_loaded_knee_RFy','mean_mono_loaded_knee_RFy','std_mono_loaded_knee_RFy',\
           'mean_bi_noload_knee_RFy','std_bi_noload_knee_RFy','mean_mono_noload_knee_RFy','std_mono_noload_knee_RFy',\
           'mean_bi_loaded_knee_RFz','std_bi_loaded_knee_RFz','mean_mono_loaded_knee_RFz','std_mono_loaded_knee_RFz',\
           'mean_bi_noload_knee_RFz','std_bi_noload_knee_RFz','mean_mono_noload_knee_RFz','std_mono_noload_knee_RFz',\
           'mean_bi_loaded_knee_RMx','std_bi_loaded_knee_RMx','mean_mono_loaded_knee_RMx','std_mono_loaded_knee_RMx',\
           'mean_bi_noload_knee_RMx','std_bi_noload_knee_RMx','mean_mono_noload_knee_RMx','std_mono_noload_knee_RMx',\
           'mean_bi_loaded_knee_RMy','std_bi_loaded_knee_RMy','mean_mono_loaded_knee_RMy','std_mono_loaded_knee_RMy',\
           'mean_bi_noload_knee_RMy','std_bi_noload_knee_RMy','mean_mono_noload_knee_RMy','std_mono_noload_knee_RMy',\
           'mean_bi_loaded_knee_RMz','std_bi_loaded_knee_RMz','mean_mono_loaded_knee_RMz','std_mono_loaded_knee_RMz',\
           'mean_bi_noload_knee_RMz','std_bi_noload_knee_RMz','mean_mono_noload_knee_RMz','std_mono_noload_knee_RMz',\
            # patellofemoral
           'mean_bi_loaded_patellofemoral_RFx','std_bi_loaded_patellofemoral_RFx','mean_mono_loaded_patellofemoral_RFx','std_mono_loaded_patellofemoral_RFx',\
           'mean_bi_noload_patellofemoral_RFx','std_bi_noload_patellofemoral_RFx','mean_mono_noload_patellofemoral_RFx','std_mono_noload_patellofemoral_RFx',\
           'mean_bi_loaded_patellofemoral_RFy','std_bi_loaded_patellofemoral_RFy','mean_mono_loaded_patellofemoral_RFy','std_mono_loaded_patellofemoral_RFy',\
           'mean_bi_noload_patellofemoral_RFy','std_bi_noload_patellofemoral_RFy','mean_mono_noload_patellofemoral_RFy','std_mono_noload_patellofemoral_RFy',\
           'mean_bi_loaded_patellofemoral_RFz','std_bi_loaded_patellofemoral_RFz','mean_mono_loaded_patellofemoral_RFz','std_mono_loaded_patellofemoral_RFz',\
           'mean_bi_noload_patellofemoral_RFz','std_bi_noload_patellofemoral_RFz','mean_mono_noload_patellofemoral_RFz','std_mono_noload_patellofemoral_RFz',\
           'mean_bi_loaded_patellofemoral_RMx','std_bi_loaded_patellofemoral_RMx','mean_mono_loaded_patellofemoral_RMx','std_mono_loaded_patellofemoral_RMx',\
           'mean_bi_noload_patellofemoral_RMx','std_bi_noload_patellofemoral_RMx','mean_mono_noload_patellofemoral_RMx','std_mono_noload_patellofemoral_RMx',\
           'mean_bi_loaded_patellofemoral_RMy','std_bi_loaded_patellofemoral_RMy','mean_mono_loaded_patellofemoral_RMy','std_mono_loaded_patellofemoral_RMy',\
           'mean_bi_noload_patellofemoral_RMy','std_bi_noload_patellofemoral_RMy','mean_mono_noload_patellofemoral_RMy','std_mono_noload_patellofemoral_RMy',\
           'mean_bi_loaded_patellofemoral_RMz','std_bi_loaded_patellofemoral_RMz','mean_mono_loaded_patellofemoral_RMz','std_mono_loaded_patellofemoral_RMz',\
           'mean_bi_noload_patellofemoral_RMz','std_bi_noload_patellofemoral_RMz','mean_mono_noload_patellofemoral_RMz','std_mono_noload_patellofemoral_RMz',\
            # ankle
           'mean_bi_loaded_ankle_RFx','std_bi_loaded_ankle_RFx','mean_mono_loaded_ankle_RFx','std_mono_loaded_ankle_RFx',\
           'mean_bi_noload_ankle_RFx','std_bi_noload_ankle_RFx','mean_mono_noload_ankle_RFx','std_mono_noload_ankle_RFx',\
           'mean_bi_loaded_ankle_RFy','std_bi_loaded_ankle_RFy','mean_mono_loaded_ankle_RFy','std_mono_loaded_ankle_RFy',\
           'mean_bi_noload_ankle_RFy','std_bi_noload_ankle_RFy','mean_mono_noload_ankle_RFy','std_mono_noload_ankle_RFy',\
           'mean_bi_loaded_ankle_RFz','std_bi_loaded_ankle_RFz','mean_mono_loaded_ankle_RFz','std_mono_loaded_ankle_RFz',\
           'mean_bi_noload_ankle_RFz','std_bi_noload_ankle_RFz','mean_mono_noload_ankle_RFz','std_mono_noload_ankle_RFz',\
           'mean_bi_loaded_ankle_RMx','std_bi_loaded_ankle_RMx','mean_mono_loaded_ankle_RMx','std_mono_loaded_ankle_RMx',\
           'mean_bi_noload_ankle_RMx','std_bi_noload_ankle_RMx','mean_mono_noload_ankle_RMx','std_mono_noload_ankle_RMx',\
           'mean_bi_loaded_ankle_RMy','std_bi_loaded_ankle_RMy','mean_mono_loaded_ankle_RMy','std_mono_loaded_ankle_RMy',\
           'mean_bi_noload_ankle_RMy','std_bi_noload_ankle_RMy','mean_mono_noload_ankle_RMy','std_mono_noload_ankle_RMy',\
           'mean_bi_loaded_ankle_RMz','std_bi_loaded_ankle_RMz','mean_mono_loaded_ankle_RMz','std_mono_loaded_ankle_RMz',\
           'mean_bi_noload_ankle_RMz','std_bi_noload_ankle_RMz','mean_mono_noload_ankle_RMz','std_mono_noload_ankle_RMz']
# Dataset
Data =[mean_bi_loaded_back_RMz,std_bi_loaded_back_RMz,mean_mono_loaded_back_RMz,std_mono_loaded_back_RMz,\
           mean_bi_noload_back_RMz,std_bi_noload_back_RMz,mean_mono_noload_back_RMz,std_mono_noload_back_RMz,\
            # hip joint
           mean_bi_loaded_hip_RFx,std_bi_loaded_hip_RFx,mean_mono_loaded_hip_RFx,std_mono_loaded_hip_RFx,\
           mean_bi_noload_hip_RFx,std_bi_noload_hip_RFx,mean_mono_noload_hip_RFx,std_mono_noload_hip_RFx,\
           mean_bi_loaded_hip_RFy,std_bi_loaded_hip_RFy,mean_mono_loaded_hip_RFy,std_mono_loaded_hip_RFy,\
           mean_bi_noload_hip_RFy,std_bi_noload_hip_RFy,mean_mono_noload_hip_RFy,std_mono_noload_hip_RFy,\
           mean_bi_loaded_hip_RFz,std_bi_loaded_hip_RFz,mean_mono_loaded_hip_RFz,std_mono_loaded_hip_RFz,\
           mean_bi_noload_hip_RFz,std_bi_noload_hip_RFz,mean_mono_noload_hip_RFz,std_mono_noload_hip_RFz,\
            # knee joint
           mean_bi_loaded_knee_RFx,std_bi_loaded_knee_RFx,mean_mono_loaded_knee_RFx,std_mono_loaded_knee_RFx,\
           mean_bi_noload_knee_RFx,std_bi_noload_knee_RFx,mean_mono_noload_knee_RFx,std_mono_noload_knee_RFx,\
           mean_bi_loaded_knee_RFy,std_bi_loaded_knee_RFy,mean_mono_loaded_knee_RFy,std_mono_loaded_knee_RFy,\
           mean_bi_noload_knee_RFy,std_bi_noload_knee_RFy,mean_mono_noload_knee_RFy,std_mono_noload_knee_RFy,\
           mean_bi_loaded_knee_RFz,std_bi_loaded_knee_RFz,mean_mono_loaded_knee_RFz,std_mono_loaded_knee_RFz,\
           mean_bi_noload_knee_RFz,std_bi_noload_knee_RFz,mean_mono_noload_knee_RFz,std_mono_noload_knee_RFz,\
           mean_bi_loaded_knee_RMx,std_bi_loaded_knee_RMx,mean_mono_loaded_knee_RMx,std_mono_loaded_knee_RMx,\
           mean_bi_noload_knee_RMx,std_bi_noload_knee_RMx,mean_mono_noload_knee_RMx,std_mono_noload_knee_RMx,\
           mean_bi_loaded_knee_RMy,std_bi_loaded_knee_RMy,mean_mono_loaded_knee_RMy,std_mono_loaded_knee_RMy,\
           mean_bi_noload_knee_RMy,std_bi_noload_knee_RMy,mean_mono_noload_knee_RMy,std_mono_noload_knee_RMy,\
           mean_bi_loaded_knee_RMz,std_bi_loaded_knee_RMz,mean_mono_loaded_knee_RMz,std_mono_loaded_knee_RMz,\
           mean_bi_noload_knee_RMz,std_bi_noload_knee_RMz,mean_mono_noload_knee_RMz,std_mono_noload_knee_RMz,\
            # patellofemoral
           mean_bi_loaded_patellofemoral_RFx,std_bi_loaded_patellofemoral_RFx,mean_mono_loaded_patellofemoral_RFx,std_mono_loaded_patellofemoral_RFx,\
           mean_bi_noload_patellofemoral_RFx,std_bi_noload_patellofemoral_RFx,mean_mono_noload_patellofemoral_RFx,std_mono_noload_patellofemoral_RFx,\
           mean_bi_loaded_patellofemoral_RFy,std_bi_loaded_patellofemoral_RFy,mean_mono_loaded_patellofemoral_RFy,std_mono_loaded_patellofemoral_RFy,\
           mean_bi_noload_patellofemoral_RFy,std_bi_noload_patellofemoral_RFy,mean_mono_noload_patellofemoral_RFy,std_mono_noload_patellofemoral_RFy,\
           mean_bi_loaded_patellofemoral_RFz,std_bi_loaded_patellofemoral_RFz,mean_mono_loaded_patellofemoral_RFz,std_mono_loaded_patellofemoral_RFz,\
           mean_bi_noload_patellofemoral_RFz,std_bi_noload_patellofemoral_RFz,mean_mono_noload_patellofemoral_RFz,std_mono_noload_patellofemoral_RFz,\
           mean_bi_loaded_patellofemoral_RMx,std_bi_loaded_patellofemoral_RMx,mean_mono_loaded_patellofemoral_RMx,std_mono_loaded_patellofemoral_RMx,\
           mean_bi_noload_patellofemoral_RMx,std_bi_noload_patellofemoral_RMx,mean_mono_noload_patellofemoral_RMx,std_mono_noload_patellofemoral_RMx,\
           mean_bi_loaded_patellofemoral_RMy,std_bi_loaded_patellofemoral_RMy,mean_mono_loaded_patellofemoral_RMy,std_mono_loaded_patellofemoral_RMy,\
           mean_bi_noload_patellofemoral_RMy,std_bi_noload_patellofemoral_RMy,mean_mono_noload_patellofemoral_RMy,std_mono_noload_patellofemoral_RMy,\
           mean_bi_loaded_patellofemoral_RMz,std_bi_loaded_patellofemoral_RMz,mean_mono_loaded_patellofemoral_RMz,std_mono_loaded_patellofemoral_RMz,\
           mean_bi_noload_patellofemoral_RMz,std_bi_noload_patellofemoral_RMz,mean_mono_noload_patellofemoral_RMz,std_mono_noload_patellofemoral_RMz,\
            # ankle
           mean_bi_loaded_ankle_RFx,std_bi_loaded_ankle_RFx,mean_mono_loaded_ankle_RFx,std_mono_loaded_ankle_RFx,\
           mean_bi_noload_ankle_RFx,std_bi_noload_ankle_RFx,mean_mono_noload_ankle_RFx,std_mono_noload_ankle_RFx,\
           mean_bi_loaded_ankle_RFy,std_bi_loaded_ankle_RFy,mean_mono_loaded_ankle_RFy,std_mono_loaded_ankle_RFy,\
           mean_bi_noload_ankle_RFy,std_bi_noload_ankle_RFy,mean_mono_noload_ankle_RFy,std_mono_noload_ankle_RFy,\
           mean_bi_loaded_ankle_RFz,std_bi_loaded_ankle_RFz,mean_mono_loaded_ankle_RFz,std_mono_loaded_ankle_RFz,\
           mean_bi_noload_ankle_RFz,std_bi_noload_ankle_RFz,mean_mono_noload_ankle_RFz,std_mono_noload_ankle_RFz,\
           mean_bi_loaded_ankle_RMx,std_bi_loaded_ankle_RMx,mean_mono_loaded_ankle_RMx,std_mono_loaded_ankle_RMx,\
           mean_bi_noload_ankle_RMx,std_bi_noload_ankle_RMx,mean_mono_noload_ankle_RMx,std_mono_noload_ankle_RMx,\
           mean_bi_loaded_ankle_RMy,std_bi_loaded_ankle_RMy,mean_mono_loaded_ankle_RMy,std_mono_loaded_ankle_RMy,\
           mean_bi_noload_ankle_RMy,std_bi_noload_ankle_RMy,mean_mono_noload_ankle_RMy,std_mono_noload_ankle_RMy,\
           mean_bi_loaded_ankle_RMz,std_bi_loaded_ankle_RMz,mean_mono_loaded_ankle_RMz,std_mono_loaded_ankle_RMz,\
           mean_bi_noload_ankle_RMz,std_bi_noload_ankle_RMz,mean_mono_noload_ankle_RMz,std_mono_noload_ankle_RMz]
# List of numpy vectors to a numpy ndarray and save to csv file
Data = utils.vec2mat(Data)
with open(r'.\Data\Ideal\jrf_ideal_exo_data.csv', 'wb') as f:
  f.write(bytes(utils.listToString(Headers)+'\n','UTF-8'))
  np.savetxt(f, Data, fmt='%s', delimiter=",")

#####################################################################################
# back joint reaction moment plot dictionaries
bi_loaded_back_RMz_dic = {'pgc':gait_cycle,'avg':utils.smooth(mean_bi_loaded_back_RMz,9),'label':'loaded assisted joint',
                        'std':utils.smooth(std_bi_loaded_back_RMz,9),'avg_toeoff':loaded_mean_toe_off}
bi_noload_back_RMz_dic = {'pgc':gait_cycle,'avg':utils.smooth(mean_bi_noload_back_RMz,9),'label':'unloaded assisted joint',
                        'std':utils.smooth(std_bi_noload_back_RMz,9),'avg_toeoff':noload_mean_toe_off,'load':'noload'}
mono_loaded_back_RMz_dic = {'pgc':gait_cycle,'avg':utils.smooth(mean_mono_loaded_back_RMz,9),'label':'loaded assisted joint',
                        'std':utils.smooth(std_mono_loaded_back_RMz,9),'avg_toeoff':loaded_mean_toe_off}
mono_noload_back_RMz_dic = {'pgc':gait_cycle,'avg':utils.smooth(mean_mono_noload_back_RMz,9),'label':'unloaded assisted joint ',
                        'std':utils.smooth(std_mono_noload_back_RMz,9),'avg_toeoff':noload_mean_toe_off,'load':'noload'}
unassist_loaded_back_RMz_dic = {'pgc':gait_cycle,'avg':utils.smooth(jrf_dataset['mean_loaded_backjoint_RMz'],9),'label':'loaded unassisted joint',
                        'std':utils.smooth(jrf_dataset['std_loaded_backjoint_RMz'],9),'avg_toeoff':loaded_mean_toe_off}
unassist_noload_back_RMz_dic = {'pgc':gait_cycle,'avg':utils.smooth(jrf_dataset['mean_noload_backjoint_RMz'],9),'label':'unloaded unassisted joint',
                        'std':utils.smooth(jrf_dataset['std_noload_backjoint_RMz'],9),'avg_toeoff':noload_mean_toe_off,'load':'noload'}
#*********************
# duct tape joint reaction moment plot dictionaries
bi_loaded_duct_tape_RMz_dic = {'pgc':gait_cycle,'avg':utils.smooth(mean_bi_loaded_duct_tape_RMz,9),'label':'loaded assisted joint',
                        'std':utils.smooth(std_bi_loaded_duct_tape_RMz,9),'avg_toeoff':loaded_mean_toe_off}
mono_loaded_duct_tape_RMz_dic = {'pgc':gait_cycle,'avg':utils.smooth(mean_mono_loaded_duct_tape_RMz,9),'label':'loaded assisted joint',
                        'std':utils.smooth(std_mono_loaded_duct_tape_RMz,9),'avg_toeoff':loaded_mean_toe_off}
unassist_loaded_duct_tape_RMz_dic = {'pgc':gait_cycle,'avg':utils.smooth(jrf_dataset['mean_loaded_duct_tape_joint_RMz'],9),'label':'loaded unassisted joint',
                        'std':utils.smooth(jrf_dataset['std_loaded_duct_tape_joint_RMz'],9),'avg_toeoff':loaded_mean_toe_off}
#*********************
# hip joint reaction moment plot dictionaries
bi_loaded_hip_RMz_dic = {'pgc':gait_cycle,'avg':utils.smooth(mean_bi_loaded_hip_RMz,9),'label':'loaded assisted joint',
                        'std':utils.smooth(std_bi_loaded_hip_RMz,9),'avg_toeoff':loaded_mean_toe_off}
bi_noload_hip_RMz_dic = {'pgc':gait_cycle,'avg':utils.smooth(mean_bi_noload_hip_RMz,9),'label':'unloaded assisted joint',
                        'std':utils.smooth(std_bi_noload_hip_RMz,9),'avg_toeoff':noload_mean_toe_off,'load':'noload'}
mono_loaded_hip_RMz_dic = {'pgc':gait_cycle,'avg':utils.smooth(mean_mono_loaded_hip_RMz,9),'label':'loaded assisted joint',
                        'std':utils.smooth(std_mono_loaded_hip_RMz,9),'avg_toeoff':loaded_mean_toe_off}
mono_noload_hip_RMz_dic = {'pgc':gait_cycle,'avg':utils.smooth(mean_mono_noload_hip_RMz,9),'label':'unloaded assisted joint ',
                        'std':utils.smooth(std_mono_noload_hip_RMz,9),'avg_toeoff':noload_mean_toe_off,'load':'noload'}
unassist_loaded_hip_RMz_dic = {'pgc':gait_cycle,'avg':utils.smooth(jrf_dataset['mean_loaded_hipjoint_RMz'],9),'label':'loaded unassisted joint',
                        'std':utils.smooth(jrf_dataset['std_loaded_hipjoint_RMz'],9),'avg_toeoff':loaded_mean_toe_off}
unassist_noload_hip_RMz_dic = {'pgc':gait_cycle,'avg':utils.smooth(jrf_dataset['mean_noload_hipjoint_RMz'],9),'label':'unloaded unassisted joint',
                        'std':utils.smooth(jrf_dataset['std_noload_hipjoint_RMz'],9),'avg_toeoff':noload_mean_toe_off,'load':'noload'}
# FX
bi_loaded_hip_RFx_dic = {'pgc':gait_cycle,'avg':utils.smooth(mean_bi_loaded_hip_RFx,9),'label':'loaded assisted joint',
                        'std':utils.smooth(std_bi_loaded_hip_RFx,9),'avg_toeoff':loaded_mean_toe_off}
bi_noload_hip_RFx_dic = {'pgc':gait_cycle,'avg':utils.smooth(mean_bi_noload_hip_RFx,9),'label':'unloaded assisted joint',
                        'std':utils.smooth(std_bi_noload_hip_RFx,9),'avg_toeoff':noload_mean_toe_off,'load':'noload'}
mono_loaded_hip_RFx_dic = {'pgc':gait_cycle,'avg':utils.smooth(mean_mono_loaded_hip_RFx,9),'label':'loaded assisted joint',
                        'std':utils.smooth(std_mono_loaded_hip_RFx,9),'avg_toeoff':loaded_mean_toe_off}
mono_noload_hip_RFx_dic = {'pgc':gait_cycle,'avg':utils.smooth(mean_mono_noload_hip_RFx,9),'label':'unloaded assisted joint ',
                        'std':utils.smooth(std_mono_noload_hip_RFx,9),'avg_toeoff':noload_mean_toe_off,'load':'noload'}
unassist_loaded_hip_RFx_dic = {'pgc':gait_cycle,'avg':utils.smooth(jrf_dataset['mean_loaded_hipjoint_RFx'],9),'label':'loaded unassisted joint',
                        'std':utils.smooth(jrf_dataset['std_loaded_hipjoint_RFx'],9),'avg_toeoff':loaded_mean_toe_off}
unassist_noload_hip_RFx_dic = {'pgc':gait_cycle,'avg':utils.smooth(jrf_dataset['mean_noload_hipjoint_RFx'],9),'label':'unloaded unassisted joint',
                        'std':utils.smooth(jrf_dataset['std_noload_hipjoint_RFx'],9),'avg_toeoff':noload_mean_toe_off,'load':'noload'}
# FY
bi_loaded_hip_RFy_dic = {'pgc':gait_cycle,'avg':utils.smooth(mean_bi_loaded_hip_RFy,9),'label':'loaded assisted joint',
                        'std':utils.smooth(std_bi_loaded_hip_RFy,9),'avg_toeoff':loaded_mean_toe_off}
bi_noload_hip_RFy_dic = {'pgc':gait_cycle,'avg':utils.smooth(mean_bi_noload_hip_RFy,9),'label':'unloaded assisted joint',
                        'std':utils.smooth(std_bi_noload_hip_RFy,9),'avg_toeoff':noload_mean_toe_off,'load':'noload'}
mono_loaded_hip_RFy_dic = {'pgc':gait_cycle,'avg':utils.smooth(mean_mono_loaded_hip_RFy,9),'label':'loaded assisted joint',
                        'std':utils.smooth(std_mono_loaded_hip_RFy,9),'avg_toeoff':loaded_mean_toe_off}
mono_noload_hip_RFy_dic = {'pgc':gait_cycle,'avg':utils.smooth(mean_mono_noload_hip_RFy,9),'label':'unloaded assisted joint ',
                        'std':utils.smooth(std_mono_noload_hip_RFy,9),'avg_toeoff':noload_mean_toe_off,'load':'noload'}
unassist_loaded_hip_RFy_dic = {'pgc':gait_cycle,'avg':utils.smooth(jrf_dataset['mean_loaded_hipjoint_RFy'],9),'label':'loaded unassisted joint',
                        'std':utils.smooth(jrf_dataset['std_loaded_hipjoint_RFy'],9),'avg_toeoff':loaded_mean_toe_off}
unassist_noload_hip_RFy_dic = {'pgc':gait_cycle,'avg':utils.smooth(jrf_dataset['mean_noload_hipjoint_RFy'],9),'label':'unloaded unassisted joint',
                        'std':utils.smooth(jrf_dataset['std_noload_hipjoint_RFy'],9),'avg_toeoff':noload_mean_toe_off,'load':'noload'}
# FZ
bi_loaded_hip_RFz_dic = {'pgc':gait_cycle,'avg':utils.smooth(mean_bi_loaded_hip_RFz,9),'label':'loaded assisted joint',
                        'std':utils.smooth(std_bi_loaded_hip_RFz,9),'avg_toeoff':loaded_mean_toe_off}
bi_noload_hip_RFz_dic = {'pgc':gait_cycle,'avg':utils.smooth(mean_bi_noload_hip_RFz,9),'label':'unloaded assisted joint',
                        'std':utils.smooth(std_bi_noload_hip_RFz,9),'avg_toeoff':noload_mean_toe_off,'load':'noload'}
mono_loaded_hip_RFz_dic = {'pgc':gait_cycle,'avg':utils.smooth(mean_mono_loaded_hip_RFz,9),'label':'loaded assisted joint',
                        'std':utils.smooth(std_mono_loaded_hip_RFz,9),'avg_toeoff':loaded_mean_toe_off}
mono_noload_hip_RFz_dic = {'pgc':gait_cycle,'avg':utils.smooth(mean_mono_noload_hip_RFz,9),'label':'unloaded assisted joint ',
                        'std':utils.smooth(std_mono_noload_hip_RFz,9),'avg_toeoff':noload_mean_toe_off,'load':'noload'}
unassist_loaded_hip_RFz_dic = {'pgc':gait_cycle,'avg':utils.smooth(jrf_dataset['mean_loaded_hipjoint_RFz'],9),'label':'loaded unassisted joint',
                        'std':utils.smooth(jrf_dataset['std_loaded_hipjoint_RFz'],9),'avg_toeoff':loaded_mean_toe_off}
unassist_noload_hip_RFz_dic = {'pgc':gait_cycle,'avg':utils.smooth(jrf_dataset['mean_noload_hipjoint_RFz'],9),'label':'unloaded unassisted joint',
                        'std':utils.smooth(jrf_dataset['std_noload_hipjoint_RFz'],9),'avg_toeoff':noload_mean_toe_off,'load':'noload'}

#*********************
# knee joint reaction moment plot dictionaries
## MX ##
bi_loaded_knee_RMx_dic = {'pgc':gait_cycle,'avg':utils.smooth(mean_bi_loaded_knee_RMx,9),'label':'loaded assisted joint',
                        'std':utils.smooth(std_bi_loaded_knee_RMx,9),'avg_toeoff':loaded_mean_toe_off}
bi_noload_knee_RMx_dic = {'pgc':gait_cycle,'avg':utils.smooth(mean_bi_noload_knee_RMx,9),'label':'unloaded assisted joint',
                        'std':utils.smooth(std_bi_noload_knee_RMx,9),'avg_toeoff':noload_mean_toe_off,'load':'noload'}
mono_loaded_knee_RMx_dic = {'pgc':gait_cycle,'avg':utils.smooth(mean_mono_loaded_knee_RMx,9),'label':'loaded assisted joint',
                        'std':utils.smooth(std_mono_loaded_knee_RMx,9),'avg_toeoff':loaded_mean_toe_off}
mono_noload_knee_RMx_dic = {'pgc':gait_cycle,'avg':utils.smooth(mean_mono_noload_knee_RMx,9),'label':'unloaded assisted joint',
                        'std':utils.smooth(std_mono_noload_knee_RMx,9),'avg_toeoff':noload_mean_toe_off,'load':'noload'}
unassist_loaded_knee_RMx_dic = {'pgc':gait_cycle,'avg':utils.smooth(jrf_dataset['mean_loaded_kneejoint_RMx'],9),'label':'loaded unassisted joint',
                        'std':utils.smooth(jrf_dataset['std_loaded_kneejoint_RMx'],9),'avg_toeoff':loaded_mean_toe_off}
unassist_noload_knee_RMx_dic = {'pgc':gait_cycle,'avg':utils.smooth(jrf_dataset['mean_noload_kneejoint_RMx'],9),'label':'unloaded unassisted joint',
                        'std':utils.smooth(jrf_dataset['std_noload_kneejoint_RMx'],9),'avg_toeoff':noload_mean_toe_off,'load':'noload'}
## MY ##
bi_loaded_knee_RMy_dic = {'pgc':gait_cycle,'avg':utils.smooth(mean_bi_loaded_knee_RMy,9),'label':'loaded assisted joint',
                        'std':utils.smooth(std_bi_loaded_knee_RMy,9),'avg_toeoff':loaded_mean_toe_off}
bi_noload_knee_RMy_dic = {'pgc':gait_cycle,'avg':utils.smooth(mean_bi_noload_knee_RMy,9),'label':'unloaded assisted joint',
                        'std':utils.smooth(std_bi_noload_knee_RMy,9),'avg_toeoff':noload_mean_toe_off,'load':'noload'}
mono_loaded_knee_RMy_dic = {'pgc':gait_cycle,'avg':utils.smooth(mean_mono_loaded_knee_RMy,9),'label':'loaded assisted joint',
                        'std':utils.smooth(std_mono_loaded_knee_RMy,9),'avg_toeoff':loaded_mean_toe_off}
mono_noload_knee_RMy_dic = {'pgc':gait_cycle,'avg':utils.smooth(mean_mono_noload_knee_RMy,9),'label':'unloaded assisted joint',
                        'std':utils.smooth(std_mono_noload_knee_RMy,9),'avg_toeoff':noload_mean_toe_off,'load':'noload'}
unassist_loaded_knee_RMy_dic = {'pgc':gait_cycle,'avg':utils.smooth(jrf_dataset['mean_loaded_kneejoint_RMy'],9),'label':'loaded unassisted',
                        'std':utils.smooth(jrf_dataset['std_loaded_kneejoint_RMy'],9),'avg_toeoff':loaded_mean_toe_off}
unassist_noload_knee_RMy_dic = {'pgc':gait_cycle,'avg':utils.smooth(jrf_dataset['mean_noload_kneejoint_RMy'],9),'label':'unloaded unassisted',
                        'std':utils.smooth(jrf_dataset['std_noload_kneejoint_RMy'],9),'avg_toeoff':noload_mean_toe_off,'load':'noload'}
## MZ ##
bi_loaded_knee_RMz_dic = {'pgc':gait_cycle,'avg':utils.smooth(mean_bi_loaded_knee_RMz,9),'label':'loaded assisted joint',
                        'std':utils.smooth(std_bi_loaded_knee_RMz,9),'avg_toeoff':loaded_mean_toe_off}
bi_noload_knee_RMz_dic = {'pgc':gait_cycle,'avg':utils.smooth(mean_bi_noload_knee_RMz,9),'label':'unloaded assisted joint',
                        'std':utils.smooth(std_bi_noload_knee_RMz,9),'avg_toeoff':noload_mean_toe_off,'load':'noload'}
mono_loaded_knee_RMz_dic = {'pgc':gait_cycle,'avg':utils.smooth(mean_mono_loaded_knee_RMz,9),'label':'loaded assisted joint',
                        'std':utils.smooth(std_mono_loaded_knee_RMz,9),'avg_toeoff':loaded_mean_toe_off}
mono_noload_knee_RMz_dic = {'pgc':gait_cycle,'avg':utils.smooth(mean_mono_noload_knee_RMz,9),'label':'unloaded assisted joint',
                        'std':utils.smooth(std_mono_noload_knee_RMz,9),'avg_toeoff':noload_mean_toe_off,'load':'noload'}
unassist_loaded_knee_RMz_dic = {'pgc':gait_cycle,'avg':utils.smooth(jrf_dataset['mean_loaded_kneejoint_RMz'],9),'label':'loaded unassisted',
                        'std':utils.smooth(jrf_dataset['std_loaded_kneejoint_RMz'],9),'avg_toeoff':loaded_mean_toe_off}
unassist_noload_knee_RMz_dic = {'pgc':gait_cycle,'avg':utils.smooth(jrf_dataset['mean_noload_kneejoint_RMz'],9),'label':'unloaded unassisted',
                        'std':utils.smooth(jrf_dataset['std_noload_kneejoint_RMz'],9),'avg_toeoff':noload_mean_toe_off,'load':'noload'}
#*********************
# knee joint reaction forces plot dictionaries
## FX ##
bi_loaded_knee_RFx_dic = {'pgc':gait_cycle,'avg':utils.smooth(mean_bi_loaded_knee_RFx,9),'label':'loaded assisted joint',
                        'std':utils.smooth(std_bi_loaded_knee_RFx,9),'avg_toeoff':loaded_mean_toe_off}
bi_noload_knee_RFx_dic = {'pgc':gait_cycle,'avg':utils.smooth(mean_bi_noload_knee_RFx,9),'label':'unloaded assisted joint',
                        'std':utils.smooth(std_bi_noload_knee_RFx,9),'avg_toeoff':noload_mean_toe_off,'load':'noload'}
mono_loaded_knee_RFx_dic = {'pgc':gait_cycle,'avg':utils.smooth(mean_mono_loaded_knee_RFx,9),'label':'loaded assisted joint',
                        'std':utils.smooth(std_mono_loaded_knee_RFx,9),'avg_toeoff':loaded_mean_toe_off}
mono_noload_knee_RFx_dic = {'pgc':gait_cycle,'avg':utils.smooth(mean_mono_noload_knee_RFx,9),'label':'unloaded assisted joint',
                        'std':utils.smooth(std_mono_noload_knee_RFx,9),'avg_toeoff':noload_mean_toe_off,'load':'noload'}
unassist_loaded_knee_RFx_dic = {'pgc':gait_cycle,'avg':utils.smooth(jrf_dataset['mean_loaded_kneejoint_RFx'],9),'label':'loaded unassisted joint',
                        'std':utils.smooth(jrf_dataset['std_loaded_kneejoint_RFx'],9),'avg_toeoff':loaded_mean_toe_off}
unassist_noload_knee_RFx_dic = {'pgc':gait_cycle,'avg':utils.smooth(jrf_dataset['mean_noload_kneejoint_RFx'],9),'label':'unloaded unassisted joint',
                        'std':utils.smooth(jrf_dataset['std_noload_kneejoint_RFx'],9),'avg_toeoff':noload_mean_toe_off,'load':'noload'}
## FY ##
bi_loaded_knee_RFy_dic = {'pgc':gait_cycle,'avg':utils.smooth(mean_bi_loaded_knee_RFy,9),'label':'loaded assisted joint',
                        'std':utils.smooth(std_bi_loaded_knee_RFy,9),'avg_toeoff':loaded_mean_toe_off}
bi_noload_knee_RFy_dic = {'pgc':gait_cycle,'avg':utils.smooth(mean_bi_noload_knee_RFy,9),'label':'unloaded assisted joint',
                        'std':utils.smooth(std_bi_noload_knee_RFy,9),'avg_toeoff':noload_mean_toe_off,'load':'noload'}
mono_loaded_knee_RFy_dic = {'pgc':gait_cycle,'avg':utils.smooth(mean_mono_loaded_knee_RFy,9),'label':'loaded assisted joint',
                        'std':utils.smooth(std_mono_loaded_knee_RFy,9),'avg_toeoff':loaded_mean_toe_off}
mono_noload_knee_RFy_dic = {'pgc':gait_cycle,'avg':utils.smooth(mean_mono_noload_knee_RFy,9),'label':'unloaded assisted joint',
                        'std':utils.smooth(std_mono_noload_knee_RFy,9),'avg_toeoff':noload_mean_toe_off,'load':'noload'}
unassist_loaded_knee_RFy_dic = {'pgc':gait_cycle,'avg':utils.smooth(jrf_dataset['mean_loaded_kneejoint_RFy'],9),'label':'loaded unassisted',
                        'std':utils.smooth(jrf_dataset['std_loaded_kneejoint_RFy'],9),'avg_toeoff':loaded_mean_toe_off}
unassist_noload_knee_RFy_dic = {'pgc':gait_cycle,'avg':utils.smooth(jrf_dataset['mean_noload_kneejoint_RFy'],9),'label':'unloaded unassisted',
                        'std':utils.smooth(jrf_dataset['std_noload_kneejoint_RFy'],9),'avg_toeoff':noload_mean_toe_off,'load':'noload'}
## FZ ##
bi_loaded_knee_RFz_dic = {'pgc':gait_cycle,'avg':utils.smooth(mean_bi_loaded_knee_RFz,9),'label':'loaded assisted joint',
                        'std':utils.smooth(std_bi_loaded_knee_RFz,9),'avg_toeoff':loaded_mean_toe_off}
bi_noload_knee_RFz_dic = {'pgc':gait_cycle,'avg':utils.smooth(mean_bi_noload_knee_RFz,9),'label':'unloaded assisted joint',
                        'std':utils.smooth(std_bi_noload_knee_RFz,9),'avg_toeoff':noload_mean_toe_off,'load':'noload'}
mono_loaded_knee_RFz_dic = {'pgc':gait_cycle,'avg':utils.smooth(mean_mono_loaded_knee_RFz,9),'label':'loaded assisted joint',
                        'std':utils.smooth(std_mono_loaded_knee_RFz,9),'avg_toeoff':loaded_mean_toe_off}
mono_noload_knee_RFz_dic = {'pgc':gait_cycle,'avg':utils.smooth(mean_mono_noload_knee_RFz,9),'label':'unloaded assisted joint',
                        'std':utils.smooth(std_mono_noload_knee_RFz,9),'avg_toeoff':noload_mean_toe_off,'load':'noload'}
unassist_loaded_knee_RFz_dic = {'pgc':gait_cycle,'avg':utils.smooth(jrf_dataset['mean_loaded_kneejoint_RFz'],9),'label':'loaded unassisted',
                        'std':utils.smooth(jrf_dataset['std_loaded_kneejoint_RFz'],9),'avg_toeoff':loaded_mean_toe_off}
unassist_noload_knee_RFz_dic = {'pgc':gait_cycle,'avg':utils.smooth(jrf_dataset['mean_noload_kneejoint_RFz'],9),'label':'unloaded unassisted',
                        'std':utils.smooth(jrf_dataset['std_noload_kneejoint_RFz'],9),'avg_toeoff':noload_mean_toe_off,'load':'noload'}

# *********************
# patellofemoral joint reaction moment plot dictionaries
## MX ##
bi_loaded_patellofemoral_RMx_dic = {'pgc':gait_cycle,'avg':utils.smooth(mean_bi_loaded_patellofemoral_RMx,9),'label':'loaded assisted joint',
                        'std':utils.smooth(std_bi_loaded_patellofemoral_RMx,9),'avg_toeoff':loaded_mean_toe_off}
bi_noload_patellofemoral_RMx_dic = {'pgc':gait_cycle,'avg':utils.smooth(mean_bi_noload_patellofemoral_RMx,9),'label':'unloaded assisted joint',
                        'std':utils.smooth(std_bi_noload_patellofemoral_RMx,9),'avg_toeoff':noload_mean_toe_off,'load':'noload'}
mono_loaded_patellofemoral_RMx_dic = {'pgc':gait_cycle,'avg':utils.smooth(mean_mono_loaded_patellofemoral_RMx,9),'label':'loaded assisted joint',
                        'std':utils.smooth(std_mono_loaded_patellofemoral_RMx,9),'avg_toeoff':loaded_mean_toe_off}
mono_noload_patellofemoral_RMx_dic = {'pgc':gait_cycle,'avg':utils.smooth(mean_mono_noload_patellofemoral_RMx,9),'label':'unloaded assisted joint',
                        'std':utils.smooth(std_mono_noload_patellofemoral_RMx,9),'avg_toeoff':noload_mean_toe_off,'load':'noload'}
unassist_loaded_patellofemoral_RMx_dic = {'pgc':gait_cycle,'avg':utils.smooth(jrf_dataset['mean_loaded_patellofemoraljoint_RMx'],9),'label':'loaded unassisted',
                        'std':utils.smooth(jrf_dataset['std_loaded_patellofemoraljoint_RMx'],9),'avg_toeoff':loaded_mean_toe_off}
unassist_noload_patellofemoral_RMx_dic = {'pgc':gait_cycle,'avg':utils.smooth(jrf_dataset['mean_noload_patellofemoraljoint_RMx'],9),'label':'unloaded unassisted',
                        'std':utils.smooth(jrf_dataset['std_noload_patellofemoraljoint_RMx'],9),'avg_toeoff':noload_mean_toe_off,'load':'noload'}
## MY ##
bi_loaded_patellofemoral_RMy_dic = {'pgc':gait_cycle,'avg':utils.smooth(mean_bi_loaded_patellofemoral_RMy,9),'label':'loaded assisted joint',
                        'std':utils.smooth(std_bi_loaded_patellofemoral_RMy,9),'avg_toeoff':loaded_mean_toe_off}
bi_noload_patellofemoral_RMy_dic = {'pgc':gait_cycle,'avg':utils.smooth(mean_bi_noload_patellofemoral_RMy,9),'label':'unloaded assisted joint',
                        'std':utils.smooth(std_bi_noload_patellofemoral_RMy,9),'avg_toeoff':noload_mean_toe_off,'load':'noload'}
mono_loaded_patellofemoral_RMy_dic = {'pgc':gait_cycle,'avg':utils.smooth(mean_mono_loaded_patellofemoral_RMy,9),'label':'loaded assisted joint',
                        'std':utils.smooth(std_mono_loaded_patellofemoral_RMy,9),'avg_toeoff':loaded_mean_toe_off}
mono_noload_patellofemoral_RMy_dic = {'pgc':gait_cycle,'avg':utils.smooth(mean_mono_noload_patellofemoral_RMy,9),'label':'unloaded assisted joint',
                        'std':utils.smooth(std_mono_noload_patellofemoral_RMy,9),'avg_toeoff':noload_mean_toe_off,'load':'noload'}
unassist_loaded_patellofemoral_RMy_dic = {'pgc':gait_cycle,'avg':utils.smooth(jrf_dataset['mean_loaded_patellofemoraljoint_RMy'],9),'label':'loaded unassisted',
                        'std':utils.smooth(jrf_dataset['std_loaded_patellofemoraljoint_RMy'],9),'avg_toeoff':loaded_mean_toe_off}
unassist_noload_patellofemoral_RMy_dic = {'pgc':gait_cycle,'avg':utils.smooth(jrf_dataset['mean_noload_patellofemoraljoint_RMy'],9),'label':'unloaded unassisted',
                        'std':utils.smooth(jrf_dataset['std_noload_patellofemoraljoint_RMy'],9),'avg_toeoff':noload_mean_toe_off,'load':'noload'}
## MZ ##
bi_loaded_patellofemoral_RMz_dic = {'pgc':gait_cycle,'avg':utils.smooth(mean_bi_loaded_patellofemoral_RMz,9),'label':'loaded assisted joint',
                        'std':utils.smooth(std_bi_loaded_patellofemoral_RMz,9),'avg_toeoff':loaded_mean_toe_off}
bi_noload_patellofemoral_RMz_dic = {'pgc':gait_cycle,'avg':utils.smooth(mean_bi_noload_patellofemoral_RMz,9),'label':'unloaded assisted joint',
                        'std':utils.smooth(std_bi_noload_patellofemoral_RMz,9),'avg_toeoff':noload_mean_toe_off,'load':'noload'}
mono_loaded_patellofemoral_RMz_dic = {'pgc':gait_cycle,'avg':utils.smooth(mean_mono_loaded_patellofemoral_RMz,9),'label':'loaded assisted joint',
                        'std':utils.smooth(std_mono_loaded_patellofemoral_RMz,9),'avg_toeoff':loaded_mean_toe_off}
mono_noload_patellofemoral_RMz_dic = {'pgc':gait_cycle,'avg':utils.smooth(mean_mono_noload_patellofemoral_RMz,9),'label':'unloaded assisted joint',
                        'std':utils.smooth(std_mono_noload_patellofemoral_RMz,9),'avg_toeoff':noload_mean_toe_off,'load':'noload'}
unassist_loaded_patellofemoral_RMz_dic = {'pgc':gait_cycle,'avg':utils.smooth(jrf_dataset['mean_loaded_patellofemoraljoint_RMz'],9),'label':'loaded unassisted',
                        'std':utils.smooth(jrf_dataset['std_loaded_patellofemoraljoint_RMz'],9),'avg_toeoff':loaded_mean_toe_off}
unassist_noload_patellofemoral_RMz_dic = {'pgc':gait_cycle,'avg':utils.smooth(jrf_dataset['mean_noload_patellofemoraljoint_RMz'],9),'label':'unloaded unassisted',
                        'std':utils.smooth(jrf_dataset['std_noload_patellofemoraljoint_RMz'],9),'avg_toeoff':noload_mean_toe_off,'load':'noload'}
# *********************
# patellofemoral joint reaction force plot dictionaries
## FX ##
bi_loaded_patellofemoral_RFx_dic = {'pgc':gait_cycle,'avg':utils.smooth(mean_bi_loaded_patellofemoral_RFx,9),'label':'loaded assisted joint',
                        'std':utils.smooth(std_bi_loaded_patellofemoral_RFx,9),'avg_toeoff':loaded_mean_toe_off}
bi_noload_patellofemoral_RFx_dic = {'pgc':gait_cycle,'avg':utils.smooth(mean_bi_noload_patellofemoral_RFx,9),'label':'unloaded assisted joint',
                        'std':utils.smooth(std_bi_noload_patellofemoral_RFx,9),'avg_toeoff':noload_mean_toe_off,'load':'noload'}
mono_loaded_patellofemoral_RFx_dic = {'pgc':gait_cycle,'avg':utils.smooth(mean_mono_loaded_patellofemoral_RFx,9),'label':'loaded assisted joint',
                        'std':utils.smooth(std_mono_loaded_patellofemoral_RFx,9),'avg_toeoff':loaded_mean_toe_off}
mono_noload_patellofemoral_RFx_dic = {'pgc':gait_cycle,'avg':utils.smooth(mean_mono_noload_patellofemoral_RFx,9),'label':'unloaded assisted joint',
                        'std':utils.smooth(std_mono_noload_patellofemoral_RFx,9),'avg_toeoff':noload_mean_toe_off,'load':'noload'}
unassist_loaded_patellofemoral_RFx_dic = {'pgc':gait_cycle,'avg':utils.smooth(jrf_dataset['mean_loaded_patellofemoraljoint_RFx'],9),'label':'loaded unassisted',
                        'std':utils.smooth(jrf_dataset['std_loaded_patellofemoraljoint_RFx'],9),'avg_toeoff':loaded_mean_toe_off}
unassist_noload_patellofemoral_RFx_dic = {'pgc':gait_cycle,'avg':utils.smooth(jrf_dataset['mean_noload_patellofemoraljoint_RFx'],9),'label':'unloaded unassisted',
                        'std':utils.smooth(jrf_dataset['std_noload_patellofemoraljoint_RFx'],9),'avg_toeoff':noload_mean_toe_off,'load':'noload'}
## FY ##
bi_loaded_patellofemoral_RFy_dic = {'pgc':gait_cycle,'avg':utils.smooth(mean_bi_loaded_patellofemoral_RFy,9),'label':'loaded assisted joint',
                        'std':utils.smooth(std_bi_loaded_patellofemoral_RFy,9),'avg_toeoff':loaded_mean_toe_off}
bi_noload_patellofemoral_RFy_dic = {'pgc':gait_cycle,'avg':utils.smooth(mean_bi_noload_patellofemoral_RFy,9),'label':'unloaded assisted joint',
                        'std':utils.smooth(std_bi_noload_patellofemoral_RFy,9),'avg_toeoff':noload_mean_toe_off,'load':'noload'}
mono_loaded_patellofemoral_RFy_dic = {'pgc':gait_cycle,'avg':utils.smooth(mean_mono_loaded_patellofemoral_RFy,9),'label':'loaded assisted joint',
                        'std':utils.smooth(std_mono_loaded_patellofemoral_RFy,9),'avg_toeoff':loaded_mean_toe_off}
mono_noload_patellofemoral_RFy_dic = {'pgc':gait_cycle,'avg':utils.smooth(mean_mono_noload_patellofemoral_RFy,9),'label':'unloaded assisted joint',
                        'std':utils.smooth(std_mono_noload_patellofemoral_RFy,9),'avg_toeoff':noload_mean_toe_off,'load':'noload'}
unassist_loaded_patellofemoral_RFy_dic = {'pgc':gait_cycle,'avg':utils.smooth(jrf_dataset['mean_loaded_patellofemoraljoint_RFy'],9),'label':'loaded unassisted',
                        'std':utils.smooth(jrf_dataset['std_loaded_patellofemoraljoint_RFy'],9),'avg_toeoff':loaded_mean_toe_off}
unassist_noload_patellofemoral_RFy_dic = {'pgc':gait_cycle,'avg':utils.smooth(jrf_dataset['mean_noload_patellofemoraljoint_RFy'],9),'label':'unloaded unassisted',
                        'std':utils.smooth(jrf_dataset['std_noload_patellofemoraljoint_RFy'],9),'avg_toeoff':noload_mean_toe_off,'load':'noload'}
## FZ ##
bi_loaded_patellofemoral_RFz_dic = {'pgc':gait_cycle,'avg':utils.smooth(mean_bi_loaded_patellofemoral_RFz,9),'label':'loaded assisted joint',
                        'std':utils.smooth(std_bi_loaded_patellofemoral_RFz,9),'avg_toeoff':loaded_mean_toe_off}
bi_noload_patellofemoral_RFz_dic = {'pgc':gait_cycle,'avg':utils.smooth(mean_bi_noload_patellofemoral_RFz,9),'label':'unloaded assisted joint',
                        'std':utils.smooth(std_bi_noload_patellofemoral_RFz,9),'avg_toeoff':noload_mean_toe_off,'load':'noload'}
mono_loaded_patellofemoral_RFz_dic = {'pgc':gait_cycle,'avg':utils.smooth(mean_mono_loaded_patellofemoral_RFz,9),'label':'loaded assisted joint',
                        'std':utils.smooth(std_mono_loaded_patellofemoral_RFz,9),'avg_toeoff':loaded_mean_toe_off}
mono_noload_patellofemoral_RFz_dic = {'pgc':gait_cycle,'avg':utils.smooth(mean_mono_noload_patellofemoral_RFz,9),'label':'unloaded assisted joint',
                        'std':utils.smooth(std_mono_noload_patellofemoral_RFz,9),'avg_toeoff':noload_mean_toe_off,'load':'noload'}
unassist_loaded_patellofemoral_RFz_dic = {'pgc':gait_cycle,'avg':utils.smooth(jrf_dataset['mean_loaded_patellofemoraljoint_RFz'],9),'label':'loaded unassisted',
                        'std':utils.smooth(jrf_dataset['std_loaded_patellofemoraljoint_RFz'],9),'avg_toeoff':loaded_mean_toe_off}
unassist_noload_patellofemoral_RFz_dic = {'pgc':gait_cycle,'avg':utils.smooth(jrf_dataset['mean_noload_patellofemoraljoint_RFz'],9),'label':'unloaded unassisted',
                        'std':utils.smooth(jrf_dataset['std_noload_patellofemoraljoint_RFz'],9),'avg_toeoff':noload_mean_toe_off,'load':'noload'}

#*********************
# ankle joint reaction moment plot dictionaries
# MX
bi_loaded_ankle_RMx_dic = {'pgc':gait_cycle,'avg':utils.smooth(mean_bi_loaded_ankle_RMx,9),'label':'loaded assisted joint',
                        'std':utils.smooth(std_bi_loaded_ankle_RMx,9),'avg_toeoff':loaded_mean_toe_off}
bi_noload_ankle_RMx_dic = {'pgc':gait_cycle,'avg':utils.smooth(mean_bi_noload_ankle_RMx,9),'label':'unloaded assisted joint',
                        'std':utils.smooth(std_bi_noload_ankle_RMx,9),'avg_toeoff':noload_mean_toe_off,'load':'noload'}
mono_loaded_ankle_RMx_dic = {'pgc':gait_cycle,'avg':utils.smooth(mean_mono_loaded_ankle_RMx,9),'label':'loaded assisted joint',
                        'std':utils.smooth(std_mono_loaded_ankle_RMx,9),'avg_toeoff':loaded_mean_toe_off}
mono_noload_ankle_RMx_dic = {'pgc':gait_cycle,'avg':utils.smooth(mean_mono_noload_ankle_RMx,9),'label':'unloaded assisted joint ',
                        'std':utils.smooth(std_mono_noload_ankle_RMx,9),'avg_toeoff':noload_mean_toe_off,'load':'noload'}
unassist_loaded_ankle_RMx_dic = {'pgc':gait_cycle,'avg':utils.smooth(jrf_dataset['mean_loaded_anklejoint_RMx'],9),'label':'loaded unassisted joint',
                        'std':utils.smooth(jrf_dataset['std_loaded_anklejoint_RMx'],9),'avg_toeoff':loaded_mean_toe_off}
unassist_noload_ankle_RMx_dic = {'pgc':gait_cycle,'avg':utils.smooth(jrf_dataset['mean_noload_anklejoint_RMx'],9),'label':'unloaded unassisted joint',
                        'std':utils.smooth(jrf_dataset['std_noload_anklejoint_RMx'],9),'avg_toeoff':noload_mean_toe_off,'load':'noload'}
# MY
bi_loaded_ankle_RMy_dic = {'pgc':gait_cycle,'avg':utils.smooth(mean_bi_loaded_ankle_RMy,9),'label':'loaded assisted joint',
                        'std':utils.smooth(std_bi_loaded_ankle_RMy,9),'avg_toeoff':loaded_mean_toe_off}
bi_noload_ankle_RMy_dic = {'pgc':gait_cycle,'avg':utils.smooth(mean_bi_noload_ankle_RMy,9),'label':'unloaded assisted joint',
                        'std':utils.smooth(std_bi_noload_ankle_RMy,9),'avg_toeoff':noload_mean_toe_off,'load':'noload'}
mono_loaded_ankle_RMy_dic = {'pgc':gait_cycle,'avg':utils.smooth(mean_mono_loaded_ankle_RMy,9),'label':'loaded assisted joint',
                        'std':utils.smooth(std_mono_loaded_ankle_RMy,9),'avg_toeoff':loaded_mean_toe_off}
mono_noload_ankle_RMy_dic = {'pgc':gait_cycle,'avg':utils.smooth(mean_mono_noload_ankle_RMy,9),'label':'unloaded assisted joint ',
                        'std':utils.smooth(std_mono_noload_ankle_RMy,9),'avg_toeoff':noload_mean_toe_off,'load':'noload'}
unassist_loaded_ankle_RMy_dic = {'pgc':gait_cycle,'avg':utils.smooth(jrf_dataset['mean_loaded_anklejoint_RMy'],9),'label':'loaded unassisted joint',
                        'std':utils.smooth(jrf_dataset['std_loaded_anklejoint_RMy'],9),'avg_toeoff':loaded_mean_toe_off}
unassist_noload_ankle_RMy_dic = {'pgc':gait_cycle,'avg':utils.smooth(jrf_dataset['mean_noload_anklejoint_RMy'],9),'label':'unloaded unassisted joint',
                        'std':utils.smooth(jrf_dataset['std_noload_anklejoint_RMy'],9),'avg_toeoff':noload_mean_toe_off,'load':'noload'}
# MZ
bi_loaded_ankle_RMz_dic = {'pgc':gait_cycle,'avg':utils.smooth(mean_bi_loaded_ankle_RMz,9),'label':'loaded assisted joint',
                        'std':utils.smooth(std_bi_loaded_ankle_RMz,9),'avg_toeoff':loaded_mean_toe_off}
bi_noload_ankle_RMz_dic = {'pgc':gait_cycle,'avg':utils.smooth(mean_bi_noload_ankle_RMz,9),'label':'unloaded assisted joint',
                        'std':utils.smooth(std_bi_noload_ankle_RMz,9),'avg_toeoff':noload_mean_toe_off,'load':'noload'}
mono_loaded_ankle_RMz_dic = {'pgc':gait_cycle,'avg':utils.smooth(mean_mono_loaded_ankle_RMz,9),'label':'loaded assisted joint',
                        'std':utils.smooth(std_mono_loaded_ankle_RMz,9),'avg_toeoff':loaded_mean_toe_off}
mono_noload_ankle_RMz_dic = {'pgc':gait_cycle,'avg':utils.smooth(mean_mono_noload_ankle_RMz,9),'label':'unloaded assisted joint ',
                        'std':utils.smooth(std_mono_noload_ankle_RMz,9),'avg_toeoff':noload_mean_toe_off,'load':'noload'}
unassist_loaded_ankle_RMz_dic = {'pgc':gait_cycle,'avg':utils.smooth(jrf_dataset['mean_loaded_anklejoint_RMz'],9),'label':'loaded unassisted joint',
                        'std':utils.smooth(jrf_dataset['std_loaded_anklejoint_RMz'],9),'avg_toeoff':loaded_mean_toe_off}
unassist_noload_ankle_RMz_dic = {'pgc':gait_cycle,'avg':utils.smooth(jrf_dataset['mean_noload_anklejoint_RMz'],9),'label':'unloaded unassisted joint',
                        'std':utils.smooth(jrf_dataset['std_noload_anklejoint_RMz'],9),'avg_toeoff':noload_mean_toe_off,'load':'noload'}
#*********************
# ankle joint reaction force plot dictionaries
# FX
bi_loaded_ankle_RFx_dic = {'pgc':gait_cycle,'avg':utils.smooth(mean_bi_loaded_ankle_RFx,9),'label':'loaded assisted joint',
                        'std':utils.smooth(std_bi_loaded_ankle_RFx,9),'avg_toeoff':loaded_mean_toe_off}
bi_noload_ankle_RFx_dic = {'pgc':gait_cycle,'avg':utils.smooth(mean_bi_noload_ankle_RFx,9),'label':'unloaded assisted joint',
                        'std':utils.smooth(std_bi_noload_ankle_RFx,9),'avg_toeoff':noload_mean_toe_off,'load':'noload'}
mono_loaded_ankle_RFx_dic = {'pgc':gait_cycle,'avg':utils.smooth(mean_mono_loaded_ankle_RFx,9),'label':'loaded assisted joint',
                        'std':utils.smooth(std_mono_loaded_ankle_RFx,9),'avg_toeoff':loaded_mean_toe_off}
mono_noload_ankle_RFx_dic = {'pgc':gait_cycle,'avg':utils.smooth(mean_mono_noload_ankle_RFx,9),'label':'unloaded assisted joint ',
                        'std':utils.smooth(std_mono_noload_ankle_RFx,9),'avg_toeoff':noload_mean_toe_off,'load':'noload'}
unassist_loaded_ankle_RFx_dic = {'pgc':gait_cycle,'avg':utils.smooth(jrf_dataset['mean_loaded_anklejoint_RFx'],9),'label':'loaded unassisted joint',
                        'std':utils.smooth(jrf_dataset['std_loaded_anklejoint_RFx'],9),'avg_toeoff':loaded_mean_toe_off}
unassist_noload_ankle_RFx_dic = {'pgc':gait_cycle,'avg':utils.smooth(jrf_dataset['mean_noload_anklejoint_RFx'],9),'label':'unloaded unassisted joint',
                        'std':utils.smooth(jrf_dataset['std_noload_anklejoint_RFx'],9),'avg_toeoff':noload_mean_toe_off,'load':'noload'}
# FY
bi_loaded_ankle_RFy_dic = {'pgc':gait_cycle,'avg':utils.smooth(mean_bi_loaded_ankle_RFy,9),'label':'loaded assisted joint',
                        'std':utils.smooth(std_bi_loaded_ankle_RFy,9),'avg_toeoff':loaded_mean_toe_off}
bi_noload_ankle_RFy_dic = {'pgc':gait_cycle,'avg':utils.smooth(mean_bi_noload_ankle_RFy,9),'label':'unloaded assisted joint',
                        'std':utils.smooth(std_bi_noload_ankle_RFy,9),'avg_toeoff':noload_mean_toe_off,'load':'noload'}
mono_loaded_ankle_RFy_dic = {'pgc':gait_cycle,'avg':utils.smooth(mean_mono_loaded_ankle_RFy,9),'label':'loaded assisted joint',
                        'std':utils.smooth(std_mono_loaded_ankle_RFy,9),'avg_toeoff':loaded_mean_toe_off}
mono_noload_ankle_RFy_dic = {'pgc':gait_cycle,'avg':utils.smooth(mean_mono_noload_ankle_RFy,9),'label':'unloaded assisted joint ',
                        'std':utils.smooth(std_mono_noload_ankle_RFy,9),'avg_toeoff':noload_mean_toe_off,'load':'noload'}
unassist_loaded_ankle_RFy_dic = {'pgc':gait_cycle,'avg':utils.smooth(jrf_dataset['mean_loaded_anklejoint_RFy'],9),'label':'loaded unassisted joint',
                        'std':utils.smooth(jrf_dataset['std_loaded_anklejoint_RFy'],9),'avg_toeoff':loaded_mean_toe_off}
unassist_noload_ankle_RFy_dic = {'pgc':gait_cycle,'avg':utils.smooth(jrf_dataset['mean_noload_anklejoint_RFy'],9),'label':'unloaded unassisted joint',
                        'std':utils.smooth(jrf_dataset['std_noload_anklejoint_RFy'],9),'avg_toeoff':noload_mean_toe_off,'load':'noload'}
# FZ
bi_loaded_ankle_RFz_dic = {'pgc':gait_cycle,'avg':utils.smooth(mean_bi_loaded_ankle_RFz,9),'label':'loaded assisted joint',
                        'std':utils.smooth(std_bi_loaded_ankle_RFz,9),'avg_toeoff':loaded_mean_toe_off}
bi_noload_ankle_RFz_dic = {'pgc':gait_cycle,'avg':utils.smooth(mean_bi_noload_ankle_RFz,9),'label':'unloaded assisted joint',
                        'std':utils.smooth(std_bi_noload_ankle_RFz,9),'avg_toeoff':noload_mean_toe_off,'load':'noload'}
mono_loaded_ankle_RFz_dic = {'pgc':gait_cycle,'avg':utils.smooth(mean_mono_loaded_ankle_RFz,9),'label':'loaded assisted joint',
                        'std':utils.smooth(std_mono_loaded_ankle_RFz,9),'avg_toeoff':loaded_mean_toe_off}
mono_noload_ankle_RFz_dic = {'pgc':gait_cycle,'avg':utils.smooth(mean_mono_noload_ankle_RFz,9),'label':'unloaded assisted joint ',
                        'std':utils.smooth(std_mono_noload_ankle_RFz,9),'avg_toeoff':noload_mean_toe_off,'load':'noload'}
unassist_loaded_ankle_RFz_dic = {'pgc':gait_cycle,'avg':utils.smooth(jrf_dataset['mean_loaded_anklejoint_RFz'],9),'label':'loaded unassisted joint',
                        'std':utils.smooth(jrf_dataset['std_loaded_anklejoint_RFz'],9),'avg_toeoff':loaded_mean_toe_off}
unassist_noload_ankle_RFz_dic = {'pgc':gait_cycle,'avg':utils.smooth(jrf_dataset['mean_noload_anklejoint_RFz'],9),'label':'unloaded unassisted joint',
                        'std':utils.smooth(jrf_dataset['std_noload_anklejoint_RFz'],9),'avg_toeoff':noload_mean_toe_off,'load':'noload'}
#******************************************************************************************************************************
#******************************************************************************************************************************
# defualt color dictionary
default_color_dic = {
'color_1_list' : ['k','k','k','k','k','k','xkcd:irish green','xkcd:irish green','xkcd:irish green',\
                  'xkcd:irish green','xkcd:irish green','xkcd:irish green'],
'color_3_list' : [mycolors['cyan blue'],mycolors['cyan blue'],mycolors['cyan blue'],\
                  mycolors['cyan blue'],mycolors['cyan blue'],mycolors['cyan blue'],\
                  mycolors['olympic blue'],mycolors['olympic blue'],mycolors['olympic blue'],\
                  mycolors['olympic blue'],mycolors['olympic blue'],mycolors['olympic blue']],
'color_2_list' : [mycolors['crimson red'],mycolors['crimson red'],mycolors['crimson red'],\
                  mycolors['cyan blue'],mycolors['cyan blue'],mycolors['cyan blue'],\
                  mycolors['french rose'],mycolors['french rose'],mycolors['french rose'],\
                  mycolors['olympic blue'],mycolors['olympic blue'],mycolors['olympic blue']]}
monovsbi_color_dic = {
'color_1_list' : ['k','xkcd:irish green','k','xkcd:irish green'],
'color_3_list' : [mycolors['cyan blue'],mycolors['cyan blue'],mycolors['cyan blue'],\
                  mycolors['cyan blue'],mycolors['cyan blue'],mycolors['cyan blue'],\
                  mycolors['olympic blue'],mycolors['olympic blue'],mycolors['olympic blue'],\
                  mycolors['olympic blue'],mycolors['olympic blue'],mycolors['olympic blue']],
'color_2_list' : [mycolors['crimson red'],mycolors['crimson red'],mycolors['crimson red'],\
                  mycolors['cyan blue'],mycolors['cyan blue'],mycolors['cyan blue'],\
                  mycolors['french rose'],mycolors['french rose'],mycolors['french rose'],\
                  mycolors['olympic blue'],mycolors['olympic blue'],mycolors['olympic blue']]
}
'''
# ***************************
# back joint moment figure
# required dictionary
plot_dic={
'plot_1_list' : [unassist_loaded_back_RMz_dic,unassist_noload_back_RMz_dic,\
                     unassist_loaded_back_RMz_dic,unassist_noload_back_RMz_dic],
'plot_2_list' : [bi_loaded_back_RMz_dic,bi_noload_back_RMz_dic,\
                      mono_loaded_back_RMz_dic,mono_noload_back_RMz_dic],
'plot_titles' : ['loaded biarticular back joint','unloaded biarticular back joint','loaded monoarticular back joint','unloaded monoarticular back joint']
}
# plot
fig = plt.figure(num='Loaded Back Reaction Moment',figsize=(9.4, 6.8))
utils.plot_joint_muscle_exo(nrows=2,ncols=2,plot_dic=plot_dic,thirdplot=False,
                            color_dic=default_color_dic,ylabel='back reaction\n moment (N-m/kg)')#,
#                            y_ticks = [-0.2,0,0.2,0.4,0.6,0.8,1])
fig.tight_layout()
fig.savefig('./Figures/Ideal/JRF/Back_Joint_ReactionMoment.pdf',orientation='landscape',bbox_inches='tight')
plt.show()

# ***************************
# duct tape joint moment figure
# required dictionary
plot_dic={
'plot_1_list' : [unassist_loaded_duct_tape_RMz_dic,unassist_loaded_duct_tape_RMz_dic],
'plot_2_list' : [bi_loaded_duct_tape_RMz_dic,mono_loaded_duct_tape_RMz_dic],
'plot_titles' : ['loaded bi-articular\nduck tape joint','loaded mono-articular\nduck tape joint']
}
# plot
fig = plt.figure(num='Loaded Back Reaction Moment',figsize=(6.4, 4.8))
utils.plot_joint_muscle_exo(nrows=1,ncols=2,plot_dic=plot_dic,thirdplot=False,
                            color_dic=default_color_dic,ylabel='duct tape reaction\n moment (N-m/kg)')#,
#                            y_ticks = )[-0.2,-0.1,0,0.1])
fig.tight_layout()
fig.savefig('./Figures/Ideal/JRF/Duck_Tape_Joint_ReactionMoment.pdf',orientation='landscape',bbox_inches='tight')
plt.show()

# ***************************
# hip joint moment figure
# required dictionary
plot_dic={
'plot_1_list' : [unassist_loaded_hip_RMz_dic,unassist_noload_hip_RMz_dic,\
                     unassist_loaded_hip_RMz_dic,unassist_noload_hip_RMz_dic],
'plot_2_list' : [bi_loaded_hip_RMz_dic,bi_noload_hip_RMz_dic,\
                      mono_loaded_hip_RMz_dic,mono_noload_hip_RMz_dic],
'plot_titles' : ['loaded biarticular hip joint','unloaded biarticular hip joint','loaded monoarticular hip joint','unloaded monoarticular hip joint']
}
# plot
fig = plt.figure(num='Loaded Hip Reaction Moment',figsize=(9.4, 6.8))
utils.plot_joint_muscle_exo(nrows=2,ncols=2,plot_dic=plot_dic,thirdplot=False,
                            color_dic=default_color_dic,ylabel='hip reaction\n moment (N-m/kg)')#,
#                            y_ticks = )[-0.06,-0.03,0,0.03,0.06,0.09])
fig.tight_layout()
fig.savefig('./Figures/Ideal/JRF/Hip_Joint_ReactionMoment.pdf',orientation='landscape',bbox_inches='tight')
plt.show()
'''
# ***************************
# hip joint force figure
# Fx
# required dictionary
plt.rcParams.update({'font.size': 12})
plot_dic={
'plot_1_list' : [unassist_loaded_hip_RFx_dic,unassist_loaded_hip_RFy_dic,unassist_loaded_hip_RFz_dic,\
                 unassist_loaded_hip_RFx_dic,unassist_loaded_hip_RFy_dic,unassist_loaded_hip_RFz_dic,\
                 unassist_noload_hip_RFx_dic,unassist_noload_hip_RFy_dic,unassist_noload_hip_RFz_dic,\
                 unassist_noload_hip_RFx_dic,unassist_noload_hip_RFy_dic,unassist_noload_hip_RFz_dic],
'plot_2_list' : [bi_loaded_hip_RFx_dic,bi_loaded_hip_RFy_dic,bi_loaded_hip_RFz_dic,\
                 mono_loaded_hip_RFx_dic,mono_loaded_hip_RFy_dic,mono_loaded_hip_RFz_dic,\
                 bi_noload_hip_RFx_dic,bi_noload_hip_RFy_dic,bi_noload_hip_RFz_dic,\
                 mono_noload_hip_RFx_dic,mono_noload_hip_RFy_dic,mono_noload_hip_RFz_dic],
'plot_titles' : ['loaded bi-articular\n hip joint (FX)','loaded bi-articular\n hip joint (FY)','loaded bi-articular\n hip joint (FZ)',\
                 'loaded mono-articular\n hip joint (FX)','loaded mono-articular\n hip joint (FY)','loaded mono-articular\n hip joint (FZ)',\
                 'unloaded bi-articular\n hip joint (FX)','unloaded biarticular hip\n joint (FY)','unloaded bi-articular\n hip joint (FZ)',\
                 'unloaded mono-articular\n hip joint (FX)','unloaded mono-articular\n hip joint (FY)','unloaded mono-articular\n hip joint (FZ)'],
'y_ticks' : [[-40,-20,0,20,40],[-80,-60,-40,-20,0,20],[-10,0,10,20],[-40,-20,0,20,40],[-80,-60,-40,-20,0,20],[-10,0,10,20],\
             [-40,-20,0,20,40],[-80,-60,-40,-20,0,20],[-10,0,10,20],[-40,-20,0,20,40],[-80,-60,-40,-20,0,20],[-10,0,10,20]]
}
# plot
fig = plt.figure(num='Loaded Hip Reaction Force',figsize=(16.4, 12.8))
utils.plot_joint_muscle_exo(nrows=4,ncols=3,plot_dic=plot_dic,thirdplot=False,legend_loc=[0,6],
                            color_dic=default_color_dic,ylabel='hip reaction\n force (N/kg)',xlabel_loc=[9,10,11],
                            ylabel_loc=[0,3,6,9])
fig.tight_layout(h_pad=-1.5, w_pad=-1.5)
fig.subplots_adjust(top=0.99, bottom=0.075, left=0.100, right=0.975,hspace=0.40,wspace=0.15)
fig.savefig('./Figures/Ideal/JRF/PaperFigure_Hip_Joint_ReactionForce.pdf',bbox_inches='tight')
plt.show()
'''
# Fy
# required dictionary
plot_dic={
'plot_1_list' : [unassist_loaded_hip_RFy_dic,unassist_noload_hip_RFy_dic,\
                     unassist_loaded_hip_RFy_dic,unassist_noload_hip_RFy_dic],
'plot_2_list' : [bi_loaded_hip_RFy_dic,bi_noload_hip_RFy_dic,\
                      mono_loaded_hip_RFy_dic,mono_noload_hip_RFy_dic],
'plot_titles' : ['loaded biarticular hip joint (FY)','unloaded biarticular hip joint (FY)',\
               'loaded monoarticular hip joint (FY)','unloaded monoarticular hip joint (FY)']
}
# plot
fig = plt.figure(num='Loaded Hip Reaction Force',figsize=(9.4, 6.8))
utils.plot_joint_muscle_exo(nrows=2,ncols=2,plot_dic=plot_dic,thirdplot=False,
                            color_dic=default_color_dic,ylabel='hip reaction\n force (N/kg)')
                            #y_ticks = [-0.06,-0.03,0,0.03,0.06,0.09])
fig.tight_layout()
fig.savefig('./Figures/Ideal/JRF/Hip_Joint_ReactionForce_Fy.pdf',orientation='landscape',bbox_inches='tight')
plt.show()
# Fz
# required dictionary
plot_dic={
'plot_1_list' : [unassist_loaded_hip_RFz_dic,unassist_noload_hip_RFz_dic,\
                     unassist_loaded_hip_RFz_dic,unassist_noload_hip_RFz_dic],
'plot_2_list' : [bi_loaded_hip_RFz_dic,bi_noload_hip_RFz_dic,\
                      mono_loaded_hip_RFz_dic,mono_noload_hip_RFz_dic],
'plot_titles' : ['loaded biarticular hip joint (FZ)','unloaded biarticular hip joint (FZ)',\
               'loaded monoarticular hip joint (FZ)','unloaded monoarticular hip joint (FZ)']
}
# plot
fig = plt.figure(num='Loaded Hip Reaction Force',figsize=(9.4, 6.8))
utils.plot_joint_muscle_exo(nrows=2,ncols=2,plot_dic=plot_dic,thirdplot=False,
                            color_dic=default_color_dic,ylabel='hip reaction\n force (N/kg)')
                            #y_ticks = [-0.06,-0.03,0,0.03,0.06,0.09])
fig.tight_layout()
fig.savefig('./Figures/Ideal/JRF/Hip_Joint_ReactionForce_Fz.pdf',orientation='landscape',bbox_inches='tight')
plt.show()
'''
# ***************************
# knee joint moment figure
## MX ##
# required dictionary
plot_dic={
'plot_1_list' : [unassist_loaded_knee_RMx_dic,unassist_loaded_knee_RMy_dic,unassist_loaded_knee_RMz_dic,\
                 unassist_loaded_knee_RMx_dic,unassist_loaded_knee_RMy_dic,unassist_loaded_knee_RMz_dic,\
                 unassist_noload_knee_RMx_dic,unassist_noload_knee_RMy_dic,unassist_noload_knee_RMz_dic,\
                 unassist_noload_knee_RMx_dic,unassist_noload_knee_RMy_dic,unassist_noload_knee_RMz_dic],
'plot_2_list' : [bi_loaded_knee_RMx_dic,bi_loaded_knee_RMy_dic,bi_loaded_knee_RMz_dic,\
                 mono_loaded_knee_RMx_dic,mono_loaded_knee_RMy_dic,mono_loaded_knee_RMz_dic,\
                 bi_noload_knee_RMx_dic,bi_noload_knee_RMy_dic,bi_noload_knee_RMz_dic,\
                 mono_noload_knee_RMx_dic,mono_noload_knee_RMy_dic,mono_noload_knee_RMz_dic],
'plot_titles' : ['loaded bi-articular\nknee joint (Mx)','loaded bi-articular\nknee joint (My)','loaded bi-articular\nknee joint (Mz)',\
                 'loaded mono-articular\nknee joint (Mx)','loaded mono-articular\nknee joint (My)','loaded mono-articular\nknee joint (Mz)',\
                 'unloaded bi-articular\nknee joint (Mx)','unloaded bi-articular\nknee joint (My)','unloaded bi-articular\nknee joint (Mz)',\
                 'unloaded mono-articular\nknee joint (Mx)','unloaded mono-articular\nknee joint (My)','unloaded mono-articular\nknee joint (Mz)'],
'y_ticks' : [[-1,0,1],[-0.5,0,0.5],[-1,-0.5,0,0.5],[-1,0,1],[-0.5,0,0.5],[-1,-0.5,0,0.5],\
             [-1,0,1],[-0.5,0,0.5],[-1,-0.5,0,0.5],[-1,0,1],[-0.5,0,0.5],[-1,-0.5,0,0.5]]
}
# plot
fig = plt.figure(num=' Knee Reaction Moment',figsize=(16.4, 12.8))
utils.plot_joint_muscle_exo(nrows=4,ncols=3,plot_dic=plot_dic,thirdplot=False,legend_loc=[0,6],
                            color_dic=default_color_dic,ylabel='hip reaction\n moment (N-m/kg)',xlabel_loc=[9,10,11],
                            ylabel_loc=[0,3,6,9])
fig.tight_layout(h_pad=-1.5, w_pad=-1.5)
fig.subplots_adjust(top=0.99, bottom=0.075, left=0.100, right=0.975,hspace=0.40,wspace=0.15)
fig.savefig('./Figures/Ideal/JRF/PaperFigure_Knee_Joint_ReactionMoment.pdf',bbox_inches='tight')
plt.show()
'''
## MY ##
# required dictionary
plot_dic={
'plot_1_list' : [unassist_loaded_knee_RMy_dic,unassist_noload_knee_RMy_dic,\
                     unassist_loaded_knee_RMy_dic,unassist_noload_knee_RMy_dic],
'plot_2_list' : [bi_loaded_knee_RMy_dic,bi_noload_knee_RMy_dic,\
                      mono_loaded_knee_RMy_dic,mono_noload_knee_RMy_dic],
'plot_titles' : ['loaded bi-articular\nknee joint (My)','unloaded bi-articular\nknee joint (My)','loaded mono-articular\nknee joint (My)','unloaded mono-articular\nknee joint (My)']
}
# plot
fig = plt.figure(num='Loaded Knee Reaction Moment',figsize=(9.4, 6.8))
utils.plot_joint_muscle_exo(nrows=2,ncols=2,plot_dic=plot_dic,thirdplot=False,
                            color_dic=default_color_dic,ylabel='knee reaction\n moment (N-m/kg)')#,
#                            y_ticks = )[0.4,0.2,0,-0.2,-0.4])
fig.tight_layout()
fig.savefig('./Figures/Ideal/JRF/Knee_Joint_ReactionMoment_My.pdf',orientation='landscape',bbox_inches='tight')
plt.show()

## MZ ##
# required dictionary
plot_dic={
'plot_1_list' : [unassist_loaded_knee_RMz_dic,unassist_noload_knee_RMz_dic,\
                     unassist_loaded_knee_RMz_dic,unassist_noload_knee_RMz_dic],
'plot_2_list' : [bi_loaded_knee_RMz_dic,bi_noload_knee_RMz_dic,\
                      mono_loaded_knee_RMz_dic,mono_noload_knee_RMz_dic],
'plot_titles' : ['loaded bi-articular\nknee joint (Mz)','unloaded bi-articular\nknee joint (Mz)','loaded mono-articular\nknee joint (Mz)','unloaded mono-articular\nknee joint (Mz)']
}
# plot
fig = plt.figure(num='Loaded Knee Reaction Moment',figsize=(9.4, 6.8))
utils.plot_joint_muscle_exo(nrows=2,ncols=2,plot_dic=plot_dic,thirdplot=False,
                            color_dic=default_color_dic,ylabel='knee reaction\n moment (N-m/kg)')#,
#                            y_ticks = )[0,-0.2,-0.4,-0.6])
fig.tight_layout()
fig.savefig('./Figures/Ideal/JRF/Knee_Joint_ReactionMoment_Mz.pdf',orientation='landscape',bbox_inches='tight')
plt.show()
'''
# ***************************
# knee joint force figure
## FX ##
# required dictionary
plot_dic={
'plot_1_list' : [unassist_loaded_knee_RFx_dic,unassist_loaded_knee_RFy_dic,unassist_loaded_knee_RFz_dic,\
                 unassist_loaded_knee_RFx_dic,unassist_loaded_knee_RFy_dic,unassist_loaded_knee_RFz_dic,\
                 unassist_noload_knee_RFx_dic,unassist_noload_knee_RFy_dic,unassist_noload_knee_RFz_dic,\
                 unassist_noload_knee_RFx_dic,unassist_noload_knee_RFy_dic,unassist_noload_knee_RFz_dic],
'plot_2_list' : [bi_loaded_knee_RFx_dic,bi_loaded_knee_RFy_dic,bi_loaded_knee_RFz_dic,\
                 mono_loaded_knee_RFx_dic,mono_loaded_knee_RFy_dic,mono_loaded_knee_RFz_dic,\
                 bi_noload_knee_RFx_dic,bi_noload_knee_RFy_dic,bi_noload_knee_RFz_dic,\
                 mono_noload_knee_RFx_dic,mono_noload_knee_RFy_dic,mono_noload_knee_RFz_dic],
'plot_titles' : ['loaded bi-articular\nknee joint (Fx)','loaded bi-articular\nknee joint (Fy)','loaded bi-articular\nknee joint (Fz)',\
                 'loaded mono-articular\nknee joint (Fx)','loaded mono-articular\nknee joint (Fy)','loaded mono-articular\nknee joint (Fz)',\
                 'unloaded bi-articular\nknee joint (Fx)','unloaded bi-articular\nknee joint (Fy)','unloaded bi-articular\nknee joint (Fz)',\
                 'unloaded mono-articular\nknee joint (Fx)','unloaded mono-articular\nknee joint (Fy)','unloaded mono-articular\nknee joint (Fz)'],
'y_ticks' : [[-40,-20,0,20,40],[-60,-40,-20,0,20],[-10,5,0,5],[-40,-20,0,20,40],[-60,-40,-20,0,20],[-10,5,0,5],\
             [-40,-20,0,20,40],[-60,-40,-20,0,20],[-10,5,0,5],[-40,-20,0,20,40],[-60,-40,-20,0,20],[-10,5,0,5]]
}
# plot
fig = plt.figure(num=' Knee Reaction Force',figsize=(16.4, 12.8))
utils.plot_joint_muscle_exo(nrows=4,ncols=3,plot_dic=plot_dic,thirdplot=False,legend_loc=[0,6],
                            color_dic=default_color_dic,ylabel='hip reaction\n force (N/kg)',xlabel_loc=[9,10,11],
                            ylabel_loc=[0,3,6,9])
fig.tight_layout(h_pad=-1.5, w_pad=-1.5)
fig.subplots_adjust(top=0.99, bottom=0.075, left=0.100, right=0.975,hspace=0.40,wspace=0.15)
fig.savefig('./Figures/Ideal/JRF/PaperFigure_Knee_Joint_ReactionForce.pdf',bbox_inches='tight')
plt.show()
'''
## FY ##
# required dictionary
plot_dic={
'plot_1_list' : [unassist_loaded_knee_RFy_dic,unassist_noload_knee_RFy_dic,\
                     unassist_loaded_knee_RFy_dic,unassist_noload_knee_RFy_dic],
'plot_2_list' : [bi_loaded_knee_RFy_dic,bi_noload_knee_RFy_dic,\
                      mono_loaded_knee_RFy_dic,mono_noload_knee_RFy_dic],
'plot_titles' : ['loaded bi-articular\nknee joint (Fy)','unloaded bi-articular\nknee joint (Fy)',\
                'loaded mono-articular\nknee joint (Fy)','unloaded mono-articular\nknee joint (Fy)']
}
# plot
fig = plt.figure(num='Loaded Knee Reaction Force',figsize=(9.4, 6.8))
utils.plot_joint_muscle_exo(nrows=2,ncols=2,plot_dic=plot_dic,thirdplot=False,
                            color_dic=default_color_dic,ylabel='knee reaction\n force (N/kg)')
                            #y_ticks = [0.4,0.2,0,-0.2,-0.4])
fig.tight_layout()
fig.savefig('./Figures/Ideal/JRF/Knee_Joint_ReactionForce_Fy.pdf',orientation='landscape',bbox_inches='tight')
plt.show()

## FZ ##
# required dictionary
plot_dic={
'plot_1_list' : [unassist_loaded_knee_RFz_dic,unassist_noload_knee_RFz_dic,\
                     unassist_loaded_knee_RFz_dic,unassist_noload_knee_RFz_dic],
'plot_2_list' : [bi_loaded_knee_RFz_dic,bi_noload_knee_RFz_dic,\
                      mono_loaded_knee_RFz_dic,mono_noload_knee_RFz_dic],
'plot_titles' : ['loaded bi-articular\nknee joint (Fz)','unloaded bi-articular\nknee joint (Fz)',\
              'loaded mono-articular\nknee joint (Fz)','unloaded mono-articular\nknee joint (Fz)']
}
# plot
fig = plt.figure(num='Loaded Knee Reaction Force',figsize=(9.4, 6.8))
utils.plot_joint_muscle_exo(nrows=2,ncols=2,plot_dic=plot_dic,thirdplot=False,
                            color_dic=default_color_dic,ylabel='knee reaction\n force (N/kg)')
                            #y_ticks = [0,-0.2,-0.4,-0.6])
fig.tight_layout()
fig.savefig('./Figures/Ideal/JRF/Knee_Joint_ReactionForce_Fz.pdf',orientation='landscape',bbox_inches='tight')
plt.show()
'''
# ***************************
# patellofemoral joint moment figure
## MX ##
# required dictionary
plot_dic={
'plot_1_list' : [unassist_loaded_patellofemoral_RMx_dic,unassist_loaded_patellofemoral_RMy_dic,unassist_loaded_patellofemoral_RMz_dic,\
                 unassist_loaded_patellofemoral_RMx_dic,unassist_loaded_patellofemoral_RMy_dic,unassist_loaded_patellofemoral_RMz_dic,\
                 unassist_noload_patellofemoral_RMx_dic,unassist_noload_patellofemoral_RMy_dic,unassist_noload_patellofemoral_RMz_dic,\
                 unassist_noload_patellofemoral_RMx_dic,unassist_noload_patellofemoral_RMy_dic,unassist_noload_patellofemoral_RMz_dic],
'plot_2_list' : [bi_loaded_patellofemoral_RMx_dic,bi_loaded_patellofemoral_RMy_dic,bi_loaded_patellofemoral_RMz_dic,\
                 mono_loaded_patellofemoral_RMx_dic,mono_loaded_patellofemoral_RMy_dic,mono_loaded_patellofemoral_RMz_dic,\
                 bi_noload_patellofemoral_RMx_dic,bi_noload_patellofemoral_RMy_dic,bi_noload_patellofemoral_RMz_dic,\
                 mono_noload_patellofemoral_RMx_dic,mono_noload_patellofemoral_RMy_dic,mono_noload_patellofemoral_RMz_dic],
'plot_titles' : ['loaded bi-articular\npatellofemoral joint (Mx)','loaded bi-articular\npatellofemoral joint (My)','loaded bi-articular\npatellofemoral joint (Mz)',\
                 'loaded mono-articular\npatellofemoral joint (Mx)','loaded mono-articular\npatellofemoral joint (My)','loaded mono-articular\npatellofemoral joint (Mz)',\
                 'unloaded bi-articular\npatellofemoral joint (Mx)','unloaded bi-articular\npatellofemoral joint (My)','unloaded bi-articular\npatellofemoral joint (Mz)',\
                 'unloaded mono-articular\npatellofemoral joint (Mx)','unloaded mono-articular\npatellofemoral joint (My)','unloaded mono-articular\npatellofemoral joint (Mz)'],
'y_ticks' : [[-0.25,0,0.25],[-0.25,0,0.25],[-1,-0.5,0,0.5],[-0.25,0,0.25],[-0.25,0,0.25],[-1,-0.5,0,0.5],\
             [-0.25,0,0.25],[-0.25,0,0.25],[-1,-0.5,0,0.5],[-0.25,0,0.25],[-0.25,0,0.25],[-1,-0.5,0,0.5]]
}
# plot
fig = plt.figure(num=' Patellofemoral Reaction Moment',figsize=(16.4, 12.8))
utils.plot_joint_muscle_exo(nrows=4,ncols=3,plot_dic=plot_dic,thirdplot=False,legend_loc=[0,6],
                            color_dic=default_color_dic,ylabel='hip reaction\n moment (N-m/kg)',xlabel_loc=[9,10,11],
                            ylabel_loc=[0,3,6,9])
fig.tight_layout(h_pad=-1.5, w_pad=-1.5)
fig.subplots_adjust(top=0.99, bottom=0.075, left=0.100, right=0.975,hspace=0.40,wspace=0.15)
fig.savefig('./Figures/Ideal/JRF/PaperFigure_Patellofemoral_Joint_ReactionMoment.pdf',bbox_inches='tight')
plt.show()
'''
## MY ##
# required dictionary
plot_dic={
'plot_1_list' : [unassist_loaded_patellofemoral_RMy_dic,unassist_noload_patellofemoral_RMy_dic,\
                     unassist_loaded_patellofemoral_RMy_dic,unassist_noload_patellofemoral_RMy_dic],
'plot_2_list' : [bi_loaded_patellofemoral_RMy_dic,bi_noload_patellofemoral_RMy_dic,\
                      mono_loaded_patellofemoral_RMy_dic,mono_noload_patellofemoral_RMy_dic],
'plot_titles' : ['loaded bi-articular\npatellofemoral joint (My)','unloaded bi-articular\npatellofemoral joint (My)','loaded mono-articular\npatellofemoral joint (My)','unloaded mono-articular\npatellofemoral joint (My)']
}
# plot
fig = plt.figure(num='Loaded Patellofemoral Reaction Moment',figsize=(9.4, 6.8))
utils.plot_joint_muscle_exo(nrows=2,ncols=2,plot_dic=plot_dic,thirdplot=False,
                            color_dic=default_color_dic,ylabel='patellofemoral reaction\n moment (N-m/kg)',
                            y_ticks =[0.2,0.1,0,-0.1,-0.2])
fig.tight_layout()
fig.savefig('./Figures/Ideal/JRF/Patellofemoral_Joint_ReactionMoment_My.pdf',orientation='landscape',bbox_inches='tight')
plt.show()

## MZ ##
# required dictionary
plot_dic={
'plot_1_list' : [unassist_loaded_patellofemoral_RMz_dic,unassist_noload_patellofemoral_RMz_dic,\
                     unassist_loaded_patellofemoral_RMz_dic,unassist_noload_patellofemoral_RMz_dic],
'plot_2_list' : [bi_loaded_patellofemoral_RMz_dic,bi_noload_patellofemoral_RMz_dic,\
                      mono_loaded_patellofemoral_RMz_dic,mono_noload_patellofemoral_RMz_dic],
'plot_titles' : ['loaded bi-articular\npatellofemoral joint (Mz)','unloaded bi-articular\npatellofemoral joint (Mz)','loaded mono-articular\npatellofemoral joint (Mz)','unloaded mono-articular\npatellofemoral joint (Mz)']
}
# plot
fig = plt.figure(num='Loaded Patellofemoral Reaction Moment',figsize=(9.4, 6.8))
utils.plot_joint_muscle_exo(nrows=2,ncols=2,plot_dic=plot_dic,thirdplot=False,
                            color_dic=default_color_dic,ylabel='patellofemoral reaction\n moment (N-m/kg)')#,
#                            y_ticks = )[0,-0.5,-1])
fig.tight_layout()
fig.savefig('./Figures/Ideal/JRF/Patellofemoral_Joint_ReactionMoment_Mz.pdf',orientation='landscape',bbox_inches='tight')
plt.show()
'''
# ***************************
# patellofemoral joint force figure
## FX ##
# required dictionary
plot_dic={
'plot_1_list' : [unassist_loaded_patellofemoral_RFx_dic,unassist_loaded_patellofemoral_RFy_dic,unassist_loaded_patellofemoral_RFz_dic,\
                 unassist_loaded_patellofemoral_RFx_dic,unassist_loaded_patellofemoral_RFy_dic,unassist_loaded_patellofemoral_RFz_dic,\
                 unassist_noload_patellofemoral_RFx_dic,unassist_noload_patellofemoral_RFy_dic,unassist_noload_patellofemoral_RFz_dic,\
                 unassist_noload_patellofemoral_RFx_dic,unassist_noload_patellofemoral_RFy_dic,unassist_noload_patellofemoral_RFz_dic],
'plot_2_list' : [bi_loaded_patellofemoral_RFx_dic,bi_loaded_patellofemoral_RFy_dic,bi_loaded_patellofemoral_RFz_dic,\
                 mono_loaded_patellofemoral_RFx_dic,mono_loaded_patellofemoral_RFy_dic,mono_loaded_patellofemoral_RFz_dic,\
                 bi_noload_patellofemoral_RFx_dic,bi_noload_patellofemoral_RFy_dic,bi_noload_patellofemoral_RFz_dic,\
                 mono_noload_patellofemoral_RFx_dic,mono_noload_patellofemoral_RFy_dic,mono_noload_patellofemoral_RFz_dic],
'plot_titles' : ['loaded bi-articular\npatellofemoral joint (Fx)','loaded bi-articular\npatellofemoral joint (Fy)','loaded bi-articular\npatellofemoral joint (Fz)',\
                 'loaded mono-articular\npatellofemoral joint (Fx)','loaded mono-articular\npatellofemoral joint (Fy)','loaded mono-articular\npatellofemoral joint (Fz)',\
                 'unloaded bi-articular\npatellofemoral joint (Fx)','unloaded bi-articular\npatellofemoral joint (Fy)','unloaded bi-articular\npatellofemoral joint (Fz)',\
                 'unloaded mono-articular\npatellofemoral joint (Fx)','unloaded mono-articular\npatellofemoral joint (Fy)','unloaded mono-articular\npatellofemoral joint (Fz)'],
'y_ticks' : [[-10,0,10,20,30],[-20,-10,0,10],[-10,0,10,20],[-10,0,10,20,30],[-20,-10,0,10],[-10,0,10,20],\
             [-10,0,10,20,30],[-20,-10,0,10],[-10,0,10,20],[-10,0,10,20,30],[-20,-10,0,10],[-10,0,10,20]]
}
# plot
fig = plt.figure(num=' Patellofemoral Reaction Force',figsize=(16.4, 12.8))
utils.plot_joint_muscle_exo(nrows=4,ncols=3,plot_dic=plot_dic,thirdplot=False,legend_loc=[0,6],
                            color_dic=default_color_dic,ylabel='hip reaction\n force (N/kg)',xlabel_loc=[9,10,11],
                            ylabel_loc=[0,3,6,9])
fig.tight_layout(h_pad=-1.5, w_pad=-1.5)
fig.subplots_adjust(top=0.99, bottom=0.075, left=0.100, right=0.975,hspace=0.40,wspace=0.15)
fig.savefig('./Figures/Ideal/JRF/PaperFigure_Patellofemoral_Joint_ReactionForce.pdf',bbox_inches='tight')
plt.show()
'''
## FY ##
# required dictionary
plot_dic={
'plot_1_list' : [unassist_loaded_patellofemoral_RFy_dic,unassist_noload_patellofemoral_RFy_dic,\
                     unassist_loaded_patellofemoral_RFy_dic,unassist_noload_patellofemoral_RFy_dic],
'plot_2_list' : [bi_loaded_patellofemoral_RFy_dic,bi_noload_patellofemoral_RFy_dic,\
                      mono_loaded_patellofemoral_RFy_dic,mono_noload_patellofemoral_RFy_dic],
'plot_titles' : ['loaded bi-articular\npatellofemoral joint (Fy)','unloaded bi-articular\npatellofemoral joint (Fy)',\
                'loaded mono-articular\npatellofemoral joint (Fy)','unloaded mono-articular\npatellofemoral joint (Fy)']
}
# plot
fig = plt.figure(num='Loaded Patellofemoral Reaction Force',figsize=(9.4, 6.8))
utils.plot_joint_muscle_exo(nrows=2,ncols=2,plot_dic=plot_dic,thirdplot=False,
                            color_dic=default_color_dic,ylabel='patellofemoral reaction\n force (N/kg)')
                            #y_ticks =[0.2,0.1,0,-0.1,-0.2])
fig.tight_layout()
fig.savefig('./Figures/Ideal/JRF/Patellofemoral_Joint_ReactionForce_Fy.pdf',orientation='landscape',bbox_inches='tight')
plt.show()

## FZ ##
# required dictionary
plot_dic={
'plot_1_list' : [unassist_loaded_patellofemoral_RFz_dic,unassist_noload_patellofemoral_RFz_dic,\
                     unassist_loaded_patellofemoral_RFz_dic,unassist_noload_patellofemoral_RFz_dic],
'plot_2_list' : [bi_loaded_patellofemoral_RFz_dic,bi_noload_patellofemoral_RFz_dic,\
                      mono_loaded_patellofemoral_RFz_dic,mono_noload_patellofemoral_RFz_dic],
'plot_titles' : ['loaded bi-articular\npatellofemoral joint (Fz)','unloaded bi-articular\npatellofemoral joint (Fz)',\
                'loaded mono-articular\npatellofemoral joint (Fz)','unloaded mono-articular\npatellofemoral joint (Fz)']
}
# plot
fig = plt.figure(num='Loaded Patellofemoral Reaction Force',figsize=(9.4, 6.8))
utils.plot_joint_muscle_exo(nrows=2,ncols=2,plot_dic=plot_dic,thirdplot=False,
                            color_dic=default_color_dic,ylabel='patellofemoral reaction\n force (N/kg)')
                            #y_ticks = [0,-0.5,-1])
fig.tight_layout()
fig.savefig('./Figures/Ideal/JRF/Patellofemoral_Joint_ReactionForce_Fz.pdf',orientation='landscape',bbox_inches='tight')
plt.show()
'''
# ***************************
# ankle joint moment figure
# MX
# required dictionary
plot_dic={
'plot_1_list' : [unassist_loaded_ankle_RMx_dic,unassist_loaded_ankle_RMy_dic,unassist_loaded_ankle_RMz_dic,\
                 unassist_loaded_ankle_RMx_dic,unassist_loaded_ankle_RMy_dic,unassist_loaded_ankle_RMz_dic,\
                 unassist_noload_ankle_RMx_dic,unassist_noload_ankle_RMy_dic,unassist_noload_ankle_RMz_dic,\
                 unassist_noload_ankle_RMx_dic,unassist_noload_ankle_RMy_dic,unassist_noload_ankle_RMz_dic],
'plot_2_list' : [bi_loaded_ankle_RMx_dic,bi_loaded_ankle_RMy_dic,bi_loaded_ankle_RMz_dic,\
                 mono_loaded_ankle_RMx_dic,mono_loaded_ankle_RMy_dic,mono_loaded_ankle_RMz_dic,\
                 bi_noload_ankle_RMx_dic,bi_noload_ankle_RMy_dic,bi_noload_ankle_RMz_dic,\
                 mono_noload_ankle_RMx_dic,mono_noload_ankle_RMy_dic,mono_noload_ankle_RMz_dic],
'plot_titles' : ['loaded bi-articular\nankle joint (Mx)','loaded bi-articular\nankle joint (My)','loaded bi-articular\nankle joint (Mz)',\
                 'loaded mono-articular\nankle joint (Mx)','loaded mono-articular\nankle joint (My)','loaded mono-articular\nankle joint (Mz)',\
                 'unloaded bi-articular\nankle joint (Mx)','unloaded bi-articular\nankle joint (My)','unloaded bi-articular\nankle joint (Mz)',\
                 'unloaded mono-articular\nankle joint (Mx)','unloaded mono-articular\nankle joint (My)','unloaded mono-articular\nankle joint (Mz)'],
'y_ticks' : [[-0.25,0,0.25,0.50],[-0.50,-0.25,0,0.25],[-0.1,0,0.1],[-0.25,0,0.25,0.50],[-0.50,-0.25,0,0.25],[-0.1,0,0.1],\
             [-0.25,0,0.25,0.50],[-0.50,-0.25,0,0.25],[-0.1,0,0.1],[-0.25,0,0.25,0.50],[-0.50,-0.25,0,0.25],[-0.1,0,0.1]]
}
# plot
fig = plt.figure(num=' Ankle Reaction Moment',figsize=(16.4, 12.8))
utils.plot_joint_muscle_exo(nrows=4,ncols=3,plot_dic=plot_dic,thirdplot=False,legend_loc=[0,6],
                            color_dic=default_color_dic,ylabel='hip reaction\n moment (N-m/kg)',xlabel_loc=[9,10,11],
                            ylabel_loc=[0,3,6,9])
fig.tight_layout(h_pad=-1.5, w_pad=-1.5)
fig.subplots_adjust(top=0.99, bottom=0.075, left=0.100, right=0.975,hspace=0.40,wspace=0.15)
fig.savefig('./Figures/Ideal/JRF/PaperFigure_Ankle_Joint_ReactionMoment.pdf',bbox_inches='tight')
plt.show()
'''
# MY
# required dictionary
plot_dic={
'plot_1_list' : [unassist_loaded_ankle_RMy_dic,unassist_noload_ankle_RMy_dic,\
                     unassist_loaded_ankle_RMy_dic,unassist_noload_ankle_RMy_dic],
'plot_2_list' : [bi_loaded_ankle_RMy_dic,bi_noload_ankle_RMy_dic,\
                      mono_loaded_ankle_RMy_dic,mono_noload_ankle_RMy_dic],
'plot_titles' : ['loaded biarticular ankle joint (My)','unloaded biarticular ankle joint (My)',\
               'loaded monoarticular ankle joint (My)','unloaded monoarticular ankle joint (My)']
}
# plot
fig = plt.figure(num='Loaded Ankle Reaction Moment',figsize=(9.4, 6.8))
utils.plot_joint_muscle_exo(nrows=2,ncols=2,plot_dic=plot_dic,thirdplot=False,
                            color_dic=default_color_dic,ylabel='ankle reaction\n moment (N-m/kg)')#,
#                            y_ticks = )[-0.06,-0.03,0,0.03,0.06,0.09])
fig.tight_layout()
fig.savefig('./Figures/Ideal/JRF/Ankle_Joint_ReactionMoment_My.pdf',orientation='landscape',bbox_inches='tight')
plt.show()
# MZ
# required dictionary
plot_dic={
'plot_1_list' : [unassist_loaded_ankle_RMz_dic,unassist_noload_ankle_RMz_dic,\
                     unassist_loaded_ankle_RMz_dic,unassist_noload_ankle_RMz_dic],
'plot_2_list' : [bi_loaded_ankle_RMz_dic,bi_noload_ankle_RMz_dic,\
                      mono_loaded_ankle_RMz_dic,mono_noload_ankle_RMz_dic],
'plot_titles' : ['loaded biarticular ankle joint (Mz)','unloaded biarticular ankle joint (Mz)',\
                'loaded monoarticular ankle joint (Mz)','unloaded monoarticular ankle joint (Mz)']
}
# plot
fig = plt.figure(num='Loaded Ankle Reaction Moment',figsize=(9.4, 6.8))
utils.plot_joint_muscle_exo(nrows=2,ncols=2,plot_dic=plot_dic,thirdplot=False,
                            color_dic=default_color_dic,ylabel='ankle reaction\n moment (N-m/kg)')#,
#                            y_ticks = )[-0.06,-0.03,0,0.03,0.06,0.09])
fig.tight_layout()
fig.savefig('./Figures/Ideal/JRF/Ankle_Joint_ReactionMoment_Mz.pdf',orientation='landscape',bbox_inches='tight')
plt.show()
'''
# ***************************
# ankle joint moment figure
# FX
# required dictionary
plot_dic={
'plot_1_list' : [unassist_loaded_ankle_RFx_dic,unassist_loaded_ankle_RFy_dic,unassist_loaded_ankle_RFz_dic,\
                 unassist_loaded_ankle_RFx_dic,unassist_loaded_ankle_RFy_dic,unassist_loaded_ankle_RFz_dic,\
                 unassist_noload_ankle_RFx_dic,unassist_noload_ankle_RFy_dic,unassist_noload_ankle_RFz_dic,\
                 unassist_noload_ankle_RFx_dic,unassist_noload_ankle_RFy_dic,unassist_noload_ankle_RFz_dic],
'plot_2_list' : [bi_loaded_ankle_RFx_dic,bi_loaded_ankle_RFy_dic,bi_loaded_ankle_RFz_dic,\
                 mono_loaded_ankle_RFx_dic,mono_loaded_ankle_RFy_dic,mono_loaded_ankle_RFz_dic,\
                 bi_noload_ankle_RFx_dic,bi_noload_ankle_RFy_dic,bi_noload_ankle_RFz_dic,\
                 mono_noload_ankle_RFx_dic,mono_noload_ankle_RFy_dic,mono_noload_ankle_RFz_dic],
'plot_titles' : ['loaded bi-articular\nankle joint (Fx)','loaded bi-articular\nankle joint (Fy)','loaded bi-articular\nankle joint (Fz)',\
                 'loaded mono-articular\nankle joint (Fx)','loaded mono-articular\nankle joint (Fy)','loaded mono-articular\nankle joint (Fz)',\
                 'unloaded bi-articular\nankle joint (Fx)','unloaded bi-articular\nankle joint (Fy)','unloaded bi-articular\nankle joint (Fz)',\
                 'unloaded mono-articular\nankle joint (Fx)','unloaded mono-articular\nankle joint (Fy)','unloaded mono-articular\nankle joint (Fz)'],
'y_ticks' : [[-60,-40,-20,0,20],[-80,-60,-40,-20,0,20],[-10,-5,0,5],[-60,-40,-20,0,20],[-80,-60,-40,-20,0,20],[-10,-5,0,5],\
             [-60,-40,-20,0,20],[-80,-60,-40,-20,0,20],[-10,-5,0,5],[-60,-40,-20,0,20],[-80,-60,-40,-20,0,20],[-10,-5,0,5]]
}
# plot
fig = plt.figure(num=' Ankle Reaction Force',figsize=(16.4, 12.8))
utils.plot_joint_muscle_exo(nrows=4,ncols=3,plot_dic=plot_dic,thirdplot=False,legend_loc=[0,6],
                            color_dic=default_color_dic,ylabel='hip reaction\n force (N/kg)',xlabel_loc=[9,10,11],
                            ylabel_loc=[0,3,6,9])
fig.tight_layout(h_pad=-1.5, w_pad=-1.5)
fig.subplots_adjust(top=0.99, bottom=0.075, left=0.100, right=0.975,hspace=0.40,wspace=0.15)
fig.savefig('./Figures/Ideal/JRF/PaperFigure_Ankle_Joint_ReactionForce.pdf',bbox_inches='tight')
plt.show()
'''
# FY
# required dictionary
plot_dic={
'plot_1_list' : [unassist_loaded_ankle_RFy_dic,unassist_noload_ankle_RFy_dic,\
                     unassist_loaded_ankle_RFy_dic,unassist_noload_ankle_RFy_dic],
'plot_2_list' : [bi_loaded_ankle_RFy_dic,bi_noload_ankle_RFy_dic,\
                      mono_loaded_ankle_RFy_dic,mono_noload_ankle_RFy_dic],
'plot_titles' : ['loaded biarticular ankle joint (Fy)','unloaded biarticular ankle joint (Fy)',\
               'loaded monoarticular ankle joint (Fy)','unloaded monoarticular ankle joint (Fy)']
}
# plot
fig = plt.figure(num='Loaded Ankle Reaction Force',figsize=(9.4, 6.8))
utils.plot_joint_muscle_exo(nrows=2,ncols=2,plot_dic=plot_dic,thirdplot=False,
                            color_dic=default_color_dic,ylabel='ankle reaction\n force (N/kg)')
                            #,y_ticks = [-0.06,-0.03,0,0.03,0.06,0.09])
fig.tight_layout()
fig.savefig('./Figures/Ideal/JRF/Ankle_Joint_ReactionForce_Fy.pdf',orientation='landscape',bbox_inches='tight')
plt.show()
# FZ
# required dictionary
plot_dic={
'plot_1_list' : [unassist_loaded_ankle_RFz_dic,unassist_noload_ankle_RFz_dic,\
                     unassist_loaded_ankle_RFz_dic,unassist_noload_ankle_RFz_dic],
'plot_2_list' : [bi_loaded_ankle_RFz_dic,bi_noload_ankle_RFz_dic,\
                      mono_loaded_ankle_RFz_dic,mono_noload_ankle_RFz_dic],
'plot_titles' : ['loaded biarticular ankle joint (Fz)','unloaded biarticular ankle joint (Fz)',\
                'loaded monoarticular ankle joint (Fz)','unloaded monoarticular ankle joint (Fz)']
}
# plot
fig = plt.figure(num='Loaded Ankle Reaction Force',figsize=(9.4, 6.8))
utils.plot_joint_muscle_exo(nrows=2,ncols=2,plot_dic=plot_dic,thirdplot=False,
                            color_dic=default_color_dic,ylabel='ankle reaction\n force (N/kg)')
                            #,y_ticks = [-0.06,-0.03,0,0.03,0.06,0.09])
fig.tight_layout()
fig.savefig('./Figures/Ideal/JRF/Ankle_Joint_ReactionForce_Fz.pdf',orientation='landscape',bbox_inches='tight')
plt.show()
'''
########################################################################################################################################
default_color_dic = {
'color_1_list' : ['k','xkcd:irish green','k','xkcd:irish green','k','xkcd:irish green'],
'color_2_list' : [mycolors['cyan blue'],mycolors['olympic blue'],mycolors['cyan blue'],mycolors['olympic blue'],mycolors['cyan blue'],mycolors['olympic blue']],
'color_3_list' : [mycolors['crimson red'],mycolors['french rose'],mycolors['crimson red'],mycolors['french rose'],mycolors['crimson red'],mycolors['french rose']]
}
# Knee and Patellofemoral moment comparison
# Knee Joint
# required dictionary
plot_dic={
'plot_1_list' : [unassist_loaded_knee_RMx_dic,unassist_noload_knee_RMx_dic,\
                 unassist_loaded_knee_RMy_dic,unassist_noload_knee_RMy_dic,\
                 unassist_loaded_knee_RMz_dic,unassist_noload_knee_RMz_dic,],
'plot_2_list' : [bi_loaded_knee_RMx_dic,bi_noload_knee_RMx_dic,\
                 bi_loaded_knee_RMy_dic,bi_noload_knee_RMy_dic,\
                 bi_loaded_knee_RMz_dic,bi_noload_knee_RMz_dic],
'plot_3_list' : [mono_loaded_knee_RMx_dic,mono_noload_knee_RMx_dic,\
                 mono_loaded_knee_RMy_dic,mono_noload_knee_RMy_dic,\
                 mono_loaded_knee_RMz_dic,mono_noload_knee_RMz_dic],
'plot_titles' : ['loaded knee joint (Mx)','unloaded knee joint (Mx)',\
                 'loaded knee joint (My)','unloaded knee joint (My)',\
                 'loaded knee joint (Mz)','unloaded knee joint (Mz)']
}
# plot
fig = plt.figure(num='Loaded Knee Reaction Moment Comparison',figsize=(8.4, 4.8))
utils.plot_joint_muscle_exo(nrows=3,ncols=2,plot_dic=plot_dic,thirdplot=True,
                            color_dic=default_color_dic,ylabel='reaction\n moment (N-m/kg)')#,
#                            y_ticks = )[-0.6,-0.3,0,0.3,0.6],xlabel_loc=[4,5],ylabel_loc=[0,2,4])
fig.tight_layout(h_pad=-1, w_pad=-1.5)
fig.subplots_adjust(top=0.95, bottom=0.075, left=0.100, right=0.975,hspace=0.40,wspace=0.20)
fig.savefig('./Figures/Ideal/JRF/Knee_Joint_ReactionMoment_Comparison.pdf',orientation='landscape',bbox_inches='tight')
plt.show()

# Patellofemoral Joint
# required dictionary
plot_dic={
'plot_1_list' : [unassist_loaded_patellofemoral_RMx_dic,unassist_noload_patellofemoral_RMx_dic,\
                 unassist_loaded_patellofemoral_RMy_dic,unassist_noload_patellofemoral_RMy_dic,\
                 unassist_loaded_patellofemoral_RMz_dic,unassist_noload_patellofemoral_RMz_dic],
'plot_2_list' : [bi_loaded_patellofemoral_RMx_dic,bi_noload_patellofemoral_RMx_dic,\
                 bi_loaded_patellofemoral_RMy_dic,bi_noload_patellofemoral_RMy_dic,\
                 bi_loaded_patellofemoral_RMz_dic,bi_noload_patellofemoral_RMz_dic],
'plot_3_list' : [mono_loaded_patellofemoral_RMx_dic,mono_noload_patellofemoral_RMx_dic,\
                 mono_loaded_patellofemoral_RMy_dic,mono_noload_patellofemoral_RMy_dic,\
                 mono_loaded_patellofemoral_RMz_dic,mono_noload_patellofemoral_RMz_dic],
'plot_titles' : ['loaded patellofemoral joint (Mx)','unloaded patellofemoral joint (Mx)',\
                 'loaded patellofemoral joint (My)','unloaded patellofemoral joint (My)',\
                 'loaded patellofemoral joint (Mz)','unloaded patellofemoral joint (Mz)']
}
# plot
fig = plt.figure(num='Loaded Patellofemoral Reaction Moment Comparison',figsize=(8.4, 4.8))
utils.plot_joint_muscle_exo(nrows=3,ncols=2,plot_dic=plot_dic,thirdplot=True,
                            color_dic=default_color_dic,ylabel='reaction\n moment (N-m/kg)')#,
#                            y_ticks = )[-0.5,0,0.5],xlabel_loc=[4,5],ylabel_loc=[0,2,4])
fig.tight_layout(h_pad=-1, w_pad=-1.5)
fig.subplots_adjust(top=0.95, bottom=0.075, left=0.100, right=0.975,hspace=0.40,wspace=0.20)
fig.savefig('./Figures/Ideal/JRF/Patellofemoral_Joint_ReactionMoment_Comparison.pdf',orientation='landscape',bbox_inches='tight')
plt.show()
#*******************************************************************************************************************
# Knee and Patellofemoral force comparison
# Knee Joint
# required dictionary
plot_dic={
'plot_1_list' : [unassist_loaded_knee_RFx_dic,unassist_noload_knee_RFx_dic,\
                 unassist_loaded_knee_RFy_dic,unassist_noload_knee_RFy_dic,\
                 unassist_loaded_knee_RFz_dic,unassist_noload_knee_RFz_dic,],
'plot_2_list' : [bi_loaded_knee_RFx_dic,bi_noload_knee_RFx_dic,\
                 bi_loaded_knee_RFy_dic,bi_noload_knee_RFy_dic,\
                 bi_loaded_knee_RFz_dic,bi_noload_knee_RFz_dic],
'plot_3_list' : [mono_loaded_knee_RFx_dic,mono_noload_knee_RFx_dic,\
                 mono_loaded_knee_RFy_dic,mono_noload_knee_RFy_dic,\
                 mono_loaded_knee_RFz_dic,mono_noload_knee_RFz_dic],
'plot_titles' : ['loaded knee joint (Fx)','unloaded knee joint (Fx)',\
                 'loaded knee joint (Fy)','unloaded knee joint (Fy)',\
                 'loaded knee joint (Fz)','unloaded knee joint (Fz)']
}
# plot
fig = plt.figure(num='Loaded Knee Reaction Force Comparison',figsize=(8.4, 4.8))
utils.plot_joint_muscle_exo(nrows=3,ncols=2,plot_dic=plot_dic,thirdplot=True,
                            color_dic=default_color_dic,ylabel='reaction\n force (N/kg)')#,
#                            y_ticks = )[-0.6,-0.3,0,0.3,0.6],xlabel_loc=[4,5],ylabel_loc=[0,2,4])
fig.tight_layout(h_pad=-1, w_pad=-1.5)
fig.subplots_adjust(top=0.95, bottom=0.075, left=0.100, right=0.975,hspace=0.40,wspace=0.20)
fig.savefig('./Figures/Ideal/JRF/Knee_Joint_ReactionForce_Comparison.pdf',orientation='landscape',bbox_inches='tight')
plt.show()

# Patellofemoral Joint
# required dictionary
plot_dic={
'plot_1_list' : [unassist_loaded_patellofemoral_RFx_dic,unassist_noload_patellofemoral_RFx_dic,\
                 unassist_loaded_patellofemoral_RFy_dic,unassist_noload_patellofemoral_RFy_dic,\
                 unassist_loaded_patellofemoral_RFz_dic,unassist_noload_patellofemoral_RFz_dic],
'plot_2_list' : [bi_loaded_patellofemoral_RFx_dic,bi_noload_patellofemoral_RFx_dic,\
                 bi_loaded_patellofemoral_RFy_dic,bi_noload_patellofemoral_RFy_dic,\
                 bi_loaded_patellofemoral_RFz_dic,bi_noload_patellofemoral_RFz_dic],
'plot_3_list' : [mono_loaded_patellofemoral_RFx_dic,mono_noload_patellofemoral_RFx_dic,\
                 mono_loaded_patellofemoral_RFy_dic,mono_noload_patellofemoral_RFy_dic,\
                 mono_loaded_patellofemoral_RFz_dic,mono_noload_patellofemoral_RFz_dic],
'plot_titles' : ['loaded patellofemoral joint (Fx)','unloaded patellofemoral joint (Fx)',\
                 'loaded patellofemoral joint (Fy)','unloaded patellofemoral joint (Fy)',\
                 'loaded patellofemoral joint (Fz)','unloaded patellofemoral joint (Fz)']
}
# plot
fig = plt.figure(num='Loaded Patellofemoral Reaction Force Comparison',figsize=(8.4, 4.8))
utils.plot_joint_muscle_exo(nrows=3,ncols=2,plot_dic=plot_dic,thirdplot=True,
                            color_dic=default_color_dic,ylabel='reaction\n force (N/kg)')#,
#                            y_ticks = )[-0.5,0,0.5],xlabel_loc=[4,5],ylabel_loc=[0,2,4])
fig.tight_layout(h_pad=-1, w_pad=-1.5)
fig.subplots_adjust(top=0.95, bottom=0.075, left=0.100, right=0.975,hspace=0.40,wspace=0.20)
fig.savefig('./Figures/Ideal/JRF/Patellofemoral_Joint_ReactionForce_Comparison.pdf',orientation='landscape',bbox_inches='tight')
plt.show()
