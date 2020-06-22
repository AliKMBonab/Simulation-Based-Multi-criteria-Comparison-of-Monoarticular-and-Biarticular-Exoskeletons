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
rra_dataset = utils.csv2numpy('./Data/RRA/rra_final_data.csv') 
unassist_dataset = utils.csv2numpy('./Data/Unassist/unassist_final_data.csv') 
stiffness_dataset = utils.csv2numpy('./Data/Unassist/unassist_stiffness_data.csv') 
bias_dataset = utils.csv2numpy('./Data/Unassist/unassist_stiffness_bias_data.csv') 
unnormalized_moment_dataset = utils.csv2numpy('./Data/Unassist/unassist_unnormalized_moment_data.csv') 
# ideal exo torque dataset
directory = './Data/Ideal/*_torque.csv'
files = enumerate(glob.iglob(directory), 1)
exo_torque_dataset = {pathlib.PurePath(f[1]).stem: np.loadtxt(f[1], delimiter=',') for f in files}
# ideal exo torque dataset
directory = './Data/Ideal/*_kinematics.csv'
files = enumerate(glob.iglob(directory), 1)
jointkinematics_dataset = {pathlib.PurePath(f[1]).stem: np.loadtxt(f[1], delimiter=',') for f in files}
# muscles moment dataset
directory = './Data/Ideal/*_musclesmoment.csv'
files = enumerate(glob.iglob(directory), 1)
musclesmoment_dataset = {pathlib.PurePath(f[1]).stem: np.loadtxt(f[1], delimiter=',') for f in files}
# gls
gl_noload = {'noload_subject{}_trial{}'.format(i,j): utils.construct_gl_mass_side(subjectno=i,trialno=j,loadcond='noload') for i in subjects for j in trials_num}
gl_loaded = {'loaded_subject{}_trial{}'.format(i,j): utils.construct_gl_mass_side(subjectno=i,trialno=j,loadcond='loaded') for i in subjects for j in trials_num}
#####################################################################################
# Processing Data
# toe-off
noload_mean_toe_off,_,loaded_mean_toe_off,_,subjects_loaded_toe_off,subjects_noload_toe_off = utils.toe_off_avg_std(gl_noload,gl_loaded,subjects=True)
# exoskeleton torque profiles
# biarticular
# hip
bi_loaded_hip_torque = utils.normalize_direction_data(exo_torque_dataset['biarticular_ideal_loaded_hipactuator_torque'],gl_noload, normalize=True,direction=False)
bi_noload_hip_torque = utils.normalize_direction_data(exo_torque_dataset['biarticular_ideal_noload_hipactuator_torque'],gl_noload, normalize=True,direction=False)
mean_bi_loaded_hip_torque,std_bi_loaded_hip_torque = utils.mean_std_over_subjects(bi_loaded_hip_torque,avg_trials=False)
mean_bi_noload_hip_torque,std_bi_noload_hip_torque = utils.mean_std_over_subjects(bi_noload_hip_torque,avg_trials=False)
# knee
bi_loaded_knee_torque = utils.normalize_direction_data(exo_torque_dataset['biarticular_ideal_loaded_kneeactuator_torque'],gl_noload, normalize=True,direction=True)
bi_noload_knee_torque = utils.normalize_direction_data(exo_torque_dataset['biarticular_ideal_noload_kneeactuator_torque'],gl_noload, normalize=True,direction=True)
mean_bi_loaded_knee_torque,std_bi_loaded_knee_torque = utils.mean_std_over_subjects(bi_loaded_knee_torque,avg_trials=False)
mean_bi_noload_knee_torque,std_bi_noload_knee_torque = utils.mean_std_over_subjects(bi_noload_knee_torque,avg_trials=False)
# monoarticular
# hip
mono_loaded_hip_torque = utils.normalize_direction_data(exo_torque_dataset['monoarticular_ideal_loaded_hipactuator_torque'],gl_noload, normalize=True,direction=False)
mono_noload_hip_torque = utils.normalize_direction_data(exo_torque_dataset['monoarticular_ideal_noload_hipactuator_torque'],gl_noload, normalize=True,direction=False)
mean_mono_loaded_hip_torque,std_mono_loaded_hip_torque = utils.mean_std_over_subjects(mono_loaded_hip_torque,avg_trials=False)
mean_mono_noload_hip_torque,std_mono_noload_hip_torque = utils.mean_std_over_subjects(mono_noload_hip_torque,avg_trials=False)
# knee
mono_loaded_knee_torque = utils.normalize_direction_data(exo_torque_dataset['monoarticular_ideal_loaded_kneeactuator_torque'],gl_noload, normalize=True,direction=True)
mono_noload_knee_torque = utils.normalize_direction_data(exo_torque_dataset['monoarticular_ideal_noload_kneeactuator_torque'],gl_noload, normalize=True,direction=True)
mean_mono_loaded_knee_torque,std_mono_loaded_knee_torque = utils.mean_std_over_subjects(mono_loaded_knee_torque,avg_trials=False)
mean_mono_noload_knee_torque,std_mono_noload_knee_torque = utils.mean_std_over_subjects(mono_noload_knee_torque,avg_trials=False)
#******************************
# hip muscles moment
# biarticular
bi_loaded_hipmuscles_moment = utils.normalize_direction_data(musclesmoment_dataset['biarticular_ideal_loaded_hip_musclesmoment'],gl_noload, normalize=True,direction=True)
bi_noload_hipmuscles_moment = utils.normalize_direction_data(musclesmoment_dataset['biarticular_ideal_noload_hip_musclesmoment'],gl_noload, normalize=True,direction=True)
mean_bi_loaded_hipmuscles_moment,std_bi_loaded_hipmuscles_moment = utils.mean_std_over_subjects(bi_loaded_hipmuscles_moment,avg_trials=False)
mean_bi_noload_hipmuscles_moment,std_bi_noload_hipmuscles_moment = utils.mean_std_over_subjects(bi_noload_hipmuscles_moment,avg_trials=False)
# monoarticular
mono_loaded_hipmuscles_moment = utils.normalize_direction_data(musclesmoment_dataset['monoarticular_ideal_loaded_hip_musclesmoment'],gl_noload, normalize=True,direction=True)
mono_noload_hipmuscles_moment = utils.normalize_direction_data(musclesmoment_dataset['monoarticular_ideal_noload_hip_musclesmoment'],gl_noload, normalize=True,direction=True)
mean_mono_loaded_hipmuscles_moment,std_mono_loaded_hipmuscles_moment = utils.mean_std_over_subjects(mono_loaded_hipmuscles_moment,avg_trials=False)
mean_mono_noload_hipmuscles_moment,std_mono_noload_hipmuscles_moment = utils.mean_std_over_subjects(mono_noload_hipmuscles_moment,avg_trials=False)
# knee muscles moment
# biarticular
bi_loaded_kneemuscles_moment = utils.normalize_direction_data(musclesmoment_dataset['biarticular_ideal_loaded_knee_musclesmoment'],gl_noload, normalize=True,direction=True)
bi_noload_kneemuscles_moment = utils.normalize_direction_data(musclesmoment_dataset['biarticular_ideal_noload_knee_musclesmoment'],gl_noload, normalize=True,direction=True)
mean_bi_loaded_kneemuscles_moment,std_bi_loaded_kneemuscles_moment = utils.mean_std_over_subjects(bi_loaded_kneemuscles_moment,avg_trials=False)
mean_bi_noload_kneemuscles_moment,std_bi_noload_kneemuscles_moment = utils.mean_std_over_subjects(bi_noload_kneemuscles_moment,avg_trials=False)
# monoarticular
mono_loaded_kneemuscles_moment = utils.normalize_direction_data(musclesmoment_dataset['monoarticular_ideal_loaded_knee_musclesmoment'],gl_noload, normalize=True,direction=True)
mono_noload_kneemuscles_moment = utils.normalize_direction_data(musclesmoment_dataset['monoarticular_ideal_noload_knee_musclesmoment'],gl_noload, normalize=True,direction=True)
mean_mono_loaded_kneemuscles_moment,std_mono_loaded_kneemuscles_moment = utils.mean_std_over_subjects(mono_loaded_kneemuscles_moment,avg_trials=False)
mean_mono_noload_kneemuscles_moment,std_mono_noload_kneemuscles_moment = utils.mean_std_over_subjects(mono_noload_kneemuscles_moment,avg_trials=False)
#******************************
# hip kinematics
# biarticular
bi_loaded_hip_kinematics = utils.normalize_direction_data(jointkinematics_dataset['biarticular_ideal_loaded_hip_kinematics'],gl_noload, normalize=False,direction=False)
bi_noload_hip_kinematics = utils.normalize_direction_data(jointkinematics_dataset['biarticular_ideal_noload_hip_kinematics'],gl_noload, normalize=False,direction=False)
mean_bi_loaded_hip_kinematics,std_bi_loaded_hip_kinematics = utils.mean_std_over_subjects(bi_loaded_hip_kinematics,avg_trials=False)
mean_bi_noload_hip_kinematics,std_bi_noload_hip_kinematics = utils.mean_std_over_subjects(bi_noload_hip_kinematics,avg_trials=False)
# monoarticular
mono_loaded_hip_kinematics = utils.normalize_direction_data(jointkinematics_dataset['monoarticular_ideal_loaded_hip_kinematics'],gl_noload, normalize=False,direction=False)
mono_noload_hip_kinematics = utils.normalize_direction_data(jointkinematics_dataset['monoarticular_ideal_noload_hip_kinematics'],gl_noload, normalize=False,direction=False)
mean_mono_loaded_hip_kinematics,std_mono_loaded_hip_kinematics = utils.mean_std_over_subjects(mono_loaded_hip_kinematics,avg_trials=False)
mean_mono_noload_hip_kinematics,std_mono_noload_hip_kinematics = utils.mean_std_over_subjects(mono_noload_hip_kinematics,avg_trials=False)
# knee kinematics
# biarticular
bi_loaded_knee_kinematics = utils.normalize_direction_data(jointkinematics_dataset['biarticular_ideal_loaded_knee_kinematics']-jointkinematics_dataset['biarticular_ideal_loaded_hip_kinematics'],gl_noload, normalize=False,direction=False)
bi_noload_knee_kinematics = utils.normalize_direction_data(jointkinematics_dataset['biarticular_ideal_noload_knee_kinematics']-jointkinematics_dataset['biarticular_ideal_noload_hip_kinematics'],gl_noload, normalize=False,direction=False)
mean_bi_loaded_knee_kinematics,std_bi_loaded_knee_kinematics = utils.mean_std_over_subjects(bi_loaded_knee_kinematics,avg_trials=False)
mean_bi_noload_knee_kinematics,std_bi_noload_knee_kinematics = utils.mean_std_over_subjects(bi_noload_knee_kinematics,avg_trials=False)
# monoarticular
mono_loaded_knee_kinematics = utils.normalize_direction_data(jointkinematics_dataset['monoarticular_ideal_loaded_knee_kinematics'],gl_noload, normalize=False,direction=False)
mono_noload_knee_kinematics = utils.normalize_direction_data(jointkinematics_dataset['monoarticular_ideal_noload_knee_kinematics'],gl_noload, normalize=False,direction=False)
mean_mono_loaded_knee_kinematics,std_mono_loaded_knee_kinematics = utils.mean_std_over_subjects(mono_loaded_knee_kinematics,avg_trials=False)
mean_mono_noload_knee_kinematics,std_mono_noload_knee_kinematics = utils.mean_std_over_subjects(mono_noload_knee_kinematics,avg_trials=False)

##########################################################################################################################################################
# muscles and actuators stiffness
# bi loaded hip actuator
bi_loaded_hip_actuator_stiffness_dict, bi_loaded_hip_actuator_Rsquare_dict, bi_loaded_hip_actuator_bias_dict\
= utils.calculate_quasi_stiffness(angle=bi_loaded_hip_kinematics,moment=bi_loaded_hip_torque,toe_off=subjects_loaded_toe_off,joint='hip')
# bi loaded hip muscles moment
bi_loaded_hip_musclesmoment_stiffness_dict, bi_loaded_hip_musclesmoment_Rsquare_dict, bi_loaded_hip_musclesmoment_bias_dict\
= utils.calculate_quasi_stiffness(angle=bi_loaded_hip_kinematics,moment=bi_loaded_hipmuscles_moment,toe_off=subjects_loaded_toe_off,joint='hip')
# bi noload hip actuator
bi_noload_hip_actuator_stiffness_dict, bi_noload_hip_actuator_Rsquare_dict, bi_noload_hip_actuator_bias_dict\
= utils.calculate_quasi_stiffness(angle=bi_noload_hip_kinematics,moment=bi_noload_hip_torque,toe_off=subjects_noload_toe_off,joint='hip')
# bi noload hip muscles moment
bi_noload_hip_musclesmoment_stiffness_dict, bi_noload_hip_musclesmoment_Rsquare_dict, bi_noload_hip_musclesmoment_bias_dict\
= utils.calculate_quasi_stiffness(angle=bi_noload_hip_kinematics,moment=bi_noload_hipmuscles_moment,toe_off=subjects_noload_toe_off,joint='hip')
#*********************************
# bi loaded knee actuator
bi_loaded_knee_actuator_stiffness_dict, bi_loaded_knee_actuator_Rsquare_dict, bi_loaded_knee_actuator_bias_dict\
= utils.calculate_quasi_stiffness(angle=bi_loaded_knee_kinematics,moment=bi_loaded_knee_torque,toe_off=subjects_loaded_toe_off,joint='knee')
# bi loaded knee muscles moment (the joint kinematics should be applied that's why it is mono)
bi_loaded_knee_musclesmoment_stiffness_dict, bi_loaded_knee_musclesmoment_Rsquare_dict, bi_loaded_knee_musclesmoment_bias_dict\
= utils.calculate_quasi_stiffness(angle=mono_loaded_knee_kinematics,moment=bi_loaded_kneemuscles_moment,toe_off=subjects_loaded_toe_off,joint='knee')
# bi noload knee actuator
bi_noload_knee_actuator_stiffness_dict, bi_noload_knee_actuator_Rsquare_dict, bi_noload_knee_actuator_bias_dict\
= utils.calculate_quasi_stiffness(angle=bi_noload_knee_kinematics,moment=bi_noload_knee_torque,toe_off=subjects_noload_toe_off,joint='knee')
# bi noload knee muscles moment (the joint kinematics should be applied that's why it is mono)
bi_noload_knee_musclesmoment_stiffness_dict, bi_noload_knee_musclesmoment_Rsquare_dict, bi_noload_knee_musclesmoment_bias_dict\
= utils.calculate_quasi_stiffness(angle=mono_noload_knee_kinematics,moment=bi_noload_kneemuscles_moment,toe_off=subjects_noload_toe_off,joint='knee')
#*********************************
# mono loaded hip actuator
mono_loaded_hip_actuator_stiffness_dict, mono_loaded_hip_actuator_Rsquare_dict, mono_loaded_hip_actuator_bias_dict\
= utils.calculate_quasi_stiffness(angle=mono_loaded_hip_kinematics,moment=mono_loaded_hip_torque,toe_off=subjects_loaded_toe_off,joint='hip')
# mono loaded hip muscles moment
mono_loaded_hip_musclesmoment_stiffness_dict, mono_loaded_hip_musclesmoment_Rsquare_dict, mono_loaded_hip_musclesmoment_bias_dict\
= utils.calculate_quasi_stiffness(angle=mono_loaded_hip_kinematics,moment=mono_loaded_hipmuscles_moment,toe_off=subjects_loaded_toe_off,joint='hip')
# mono noload hip actuator
mono_noload_hip_actuator_stiffness_dict, mono_noload_hip_actuator_Rsquare_dict, mono_noload_hip_actuator_bias_dict\
= utils.calculate_quasi_stiffness(angle=mono_noload_hip_kinematics,moment=mono_noload_hip_torque,toe_off=subjects_noload_toe_off,joint='hip')
# mono noload hip muscles moment
mono_noload_hip_musclesmoment_stiffness_dict, mono_noload_hip_musclesmoment_Rsquare_dict, mono_noload_hip_musclesmoment_bias_dict\
= utils.calculate_quasi_stiffness(angle=mono_noload_hip_kinematics,moment=mono_noload_hipmuscles_moment,toe_off=subjects_noload_toe_off,joint='hip')
#*********************************
# mono loaded knee actuator
mono_loaded_knee_actuator_stiffness_dict, mono_loaded_knee_actuator_Rsquare_dict, mono_loaded_knee_actuator_bias_dict\
= utils.calculate_quasi_stiffness(angle=mono_loaded_knee_kinematics,moment=mono_loaded_knee_torque,toe_off=subjects_loaded_toe_off,joint='knee')
# mono loaded knee muscles moment
mono_loaded_knee_musclesmoment_stiffness_dict, mono_loaded_knee_musclesmoment_Rsquare_dict, mono_loaded_knee_musclesmoment_bias_dict\
= utils.calculate_quasi_stiffness(angle=mono_loaded_knee_kinematics,moment=mono_loaded_kneemuscles_moment,toe_off=subjects_loaded_toe_off,joint='knee')
# mono noload knee actuator
mono_noload_knee_actuator_stiffness_dict, mono_noload_knee_actuator_Rsquare_dict, mono_noload_knee_actuator_bias_dict\
= utils.calculate_quasi_stiffness(angle=mono_noload_knee_kinematics,moment=mono_noload_knee_torque,toe_off=subjects_noload_toe_off,joint='knee')
# mono noload knee muscles moment
mono_noload_knee_musclesmoment_stiffness_dict, mono_noload_knee_musclesmoment_Rsquare_dict, mono_noload_knee_musclesmoment_bias_dict\
= utils.calculate_quasi_stiffness(angle=mono_noload_knee_kinematics,moment=mono_noload_kneemuscles_moment,toe_off=subjects_noload_toe_off,joint='knee')
##########################################################################################################################################################
# The linear moment and kinematics
# bi loaded hip actuator
bi_loaded_hip_actuator_linear_angle_dict,bi_loaded_hip_actuator_linear_moment_dict,bi_loaded_hip_actuator_fitted_line =\
utils.mean_linear_phases(mean_bi_loaded_hip_kinematics,mean_bi_loaded_hip_torque,loaded_mean_toe_off,'hip',\
                         bi_loaded_hip_actuator_bias_dict,bi_loaded_hip_actuator_stiffness_dict)
# bi loaded hip muscles
bi_loaded_hip_musclesmoment_linear_angle_dict,bi_loaded_hip_musclesmoment_linear_moment_dict,bi_loaded_hip_musclesmoment_fitted_line =\
utils.mean_linear_phases(mean_bi_loaded_hip_kinematics,mean_bi_loaded_hipmuscles_moment,loaded_mean_toe_off,'hip',\
                         bi_loaded_hip_musclesmoment_bias_dict,bi_loaded_hip_musclesmoment_stiffness_dict)                       
# bi noload hip actuator
bi_noload_hip_actuator_linear_angle_dict,bi_noload_hip_actuator_linear_moment_dict,bi_noload_hip_actuator_fitted_line =\
utils.mean_linear_phases(mean_bi_noload_hip_kinematics,mean_bi_noload_hip_torque,noload_mean_toe_off,'hip',\
                         bi_noload_hip_actuator_bias_dict,bi_noload_hip_actuator_stiffness_dict)
# bi noload hip musclesmoment
bi_noload_hip_musclesmoment_linear_angle_dict,bi_noload_hip_musclesmoment_linear_moment_dict,bi_noload_hip_musclesmoment_fitted_line =\
utils.mean_linear_phases(mean_bi_noload_hip_kinematics,mean_bi_noload_hipmuscles_moment,noload_mean_toe_off,'hip',\
                         bi_noload_hip_musclesmoment_bias_dict,bi_noload_hip_musclesmoment_stiffness_dict)
#*********************************
# bi loaded knee actuator
bi_loaded_knee_actuator_linear_angle_dict,bi_loaded_knee_actuator_linear_moment_dict,bi_loaded_knee_actuator_fitted_line =\
utils.mean_linear_phases(mean_bi_loaded_knee_kinematics,mean_bi_loaded_knee_torque,loaded_mean_toe_off,'knee',\
                         bi_loaded_knee_actuator_bias_dict,bi_loaded_knee_actuator_stiffness_dict)
# bi loaded knee muscles (the kinematics resamble to the knee joint)
bi_loaded_knee_musclesmoment_linear_angle_dict,bi_loaded_knee_musclesmoment_linear_moment_dict,bi_loaded_knee_musclesmoment_fitted_line =\
utils.mean_linear_phases(mean_mono_loaded_knee_kinematics,mean_bi_loaded_kneemuscles_moment,loaded_mean_toe_off,'knee',\
                         bi_loaded_knee_musclesmoment_bias_dict,bi_loaded_knee_musclesmoment_stiffness_dict)                       
# bi noload knee actuator
bi_noload_knee_actuator_linear_angle_dict,bi_noload_knee_actuator_linear_moment_dict,bi_noload_knee_actuator_fitted_line =\
utils.mean_linear_phases(mean_bi_noload_knee_kinematics,mean_bi_noload_knee_torque,noload_mean_toe_off,'knee',\
                         bi_noload_knee_actuator_bias_dict,bi_noload_knee_actuator_stiffness_dict)
# bi noload knee musclesmoment (the kinematics resamble to the knee joint)
bi_noload_knee_musclesmoment_linear_angle_dict,bi_noload_knee_musclesmoment_linear_moment_dict,bi_noload_knee_musclesmoment_fitted_line =\
utils.mean_linear_phases(mean_mono_noload_knee_kinematics,mean_bi_noload_kneemuscles_moment,noload_mean_toe_off,'knee',\
                         bi_noload_knee_musclesmoment_bias_dict,bi_noload_knee_musclesmoment_stiffness_dict)
#*********************************
# mono loaded hip actuator
mono_loaded_hip_actuator_linear_angle_dict,mono_loaded_hip_actuator_linear_moment_dict,mono_loaded_hip_actuator_fitted_line =\
utils.mean_linear_phases(mean_mono_loaded_hip_kinematics,mean_mono_loaded_hip_torque,loaded_mean_toe_off,'hip',\
                         mono_loaded_hip_actuator_bias_dict,mono_loaded_hip_actuator_stiffness_dict)
# mono loaded hip muscles
mono_loaded_hip_musclesmoment_linear_angle_dict,mono_loaded_hip_musclesmoment_linear_moment_dict,mono_loaded_hip_musclesmoment_fitted_line =\
utils.mean_linear_phases(mean_mono_loaded_hip_kinematics,mean_mono_loaded_hipmuscles_moment,loaded_mean_toe_off,'hip',\
                         mono_loaded_hip_musclesmoment_bias_dict,mono_loaded_hip_musclesmoment_stiffness_dict)                       
# mono noload hip actuator
mono_noload_hip_actuator_linear_angle_dict,mono_noload_hip_actuator_linear_moment_dict,mono_noload_hip_actuator_fitted_line =\
utils.mean_linear_phases(mean_mono_noload_hip_kinematics,mean_mono_noload_hip_torque,noload_mean_toe_off,'hip',\
                         mono_noload_hip_actuator_bias_dict,mono_noload_hip_actuator_stiffness_dict)
# mono noload hip musclesmoment
mono_noload_hip_musclesmoment_linear_angle_dict,mono_noload_hip_musclesmoment_linear_moment_dict,mono_noload_hip_musclesmoment_fitted_line =\
utils.mean_linear_phases(mean_mono_noload_hip_kinematics,mean_mono_noload_hipmuscles_moment,noload_mean_toe_off,'hip',\
                         mono_noload_hip_musclesmoment_bias_dict,mono_noload_hip_musclesmoment_stiffness_dict)
#*********************************
# mono loaded knee actuator
mono_loaded_knee_actuator_linear_angle_dict,mono_loaded_knee_actuator_linear_moment_dict,mono_loaded_knee_actuator_fitted_line =\
utils.mean_linear_phases(mean_mono_loaded_knee_kinematics,mean_mono_loaded_knee_torque,loaded_mean_toe_off,'knee',\
                         mono_loaded_knee_actuator_bias_dict,mono_loaded_knee_actuator_stiffness_dict)
# mono loaded knee muscles
mono_loaded_knee_musclesmoment_linear_angle_dict,mono_loaded_knee_musclesmoment_linear_moment_dict,mono_loaded_knee_musclesmoment_fitted_line =\
utils.mean_linear_phases(mean_mono_loaded_knee_kinematics,mean_mono_loaded_kneemuscles_moment,loaded_mean_toe_off,'knee',\
                         mono_loaded_knee_musclesmoment_bias_dict,mono_loaded_knee_musclesmoment_stiffness_dict)                       
# mono noload knee actuator
mono_noload_knee_actuator_linear_angle_dict,mono_noload_knee_actuator_linear_moment_dict,mono_noload_knee_actuator_fitted_line =\
utils.mean_linear_phases(mean_mono_noload_knee_kinematics,mean_mono_noload_knee_torque,noload_mean_toe_off,'knee',\
                         mono_noload_knee_actuator_bias_dict,mono_noload_knee_actuator_stiffness_dict)
# mono noload knee musclesmoment
mono_noload_knee_musclesmoment_linear_angle_dict,mono_noload_knee_musclesmoment_linear_moment_dict,mono_noload_knee_musclesmoment_fitted_line =\
utils.mean_linear_phases(mean_mono_noload_knee_kinematics,mean_mono_noload_kneemuscles_moment,noload_mean_toe_off,'knee',\
                         mono_noload_knee_musclesmoment_bias_dict,mono_noload_knee_musclesmoment_stiffness_dict)
#*************************************
# joint dataset
# hip loaded
hip_loaded_linear_angle_dict, hip_loaded_linear_moment_dict, hip_loaded_fitted_line =\
utils.recover_unassist_linear_phases(rra_dataset['mean_loaded_hipjoint_kinematics'],unnormalized_moment_dataset['mean_loaded_hipmuscles_moment'],\
                                     loaded_mean_toe_off,stiffness_dataset,bias_dataset,joint='hip',load='loaded')
# hip noload
hip_noload_linear_angle_dict, hip_noload_linear_moment_dict, hip_noload_fitted_line =\
utils.recover_unassist_linear_phases(rra_dataset['mean_noload_hipjoint_kinematics'],unnormalized_moment_dataset['mean_noload_hipmuscles_moment'],\
                                     noload_mean_toe_off,stiffness_dataset,bias_dataset,joint='hip',load='noload')
# knee loaded
knee_loaded_linear_angle_dict, knee_loaded_linear_moment_dict, knee_loaded_fitted_line =\
utils.recover_unassist_linear_phases(rra_dataset['mean_loaded_kneejoint_kinematics'],unnormalized_moment_dataset['mean_loaded_kneemuscles_moment'],\
                                     loaded_mean_toe_off,stiffness_dataset,bias_dataset,joint='knee',load='loaded')
# knee noload
knee_noload_linear_angle_dict, knee_noload_linear_moment_dict, knee_noload_fitted_line =\
utils.recover_unassist_linear_phases(rra_dataset['mean_noload_kneejoint_kinematics'],unnormalized_moment_dataset['mean_noload_kneemuscles_moment'],\
                                     noload_mean_toe_off,stiffness_dataset,bias_dataset,joint='knee',load='noload')

##########################################################################################################################################################
# muscles dataset
muscles_mean_cellText = np.c_[bi_loaded_hip_musclesmoment_stiffness_dict['mean_hip_extension_stiffness'],bi_loaded_hip_musclesmoment_stiffness_dict['mean_hip_flexion_stiffness'],bi_loaded_hip_musclesmoment_stiffness_dict['mean_hip_total_stiffness'],\
                         bi_loaded_hip_musclesmoment_stiffness_dict['std_hip_extension_stiffness'],bi_loaded_hip_musclesmoment_stiffness_dict['std_hip_flexion_stiffness'],bi_loaded_hip_musclesmoment_stiffness_dict['std_hip_total_stiffness'],\
                         bi_loaded_knee_musclesmoment_stiffness_dict['mean_knee_extension_stiffness'],bi_loaded_knee_musclesmoment_stiffness_dict['mean_knee_flexion_stiffness'],bi_loaded_knee_musclesmoment_stiffness_dict['mean_knee_total_stiffness'],\
                         bi_loaded_knee_musclesmoment_stiffness_dict['std_knee_extension_stiffness'],bi_loaded_knee_musclesmoment_stiffness_dict['std_knee_flexion_stiffness'],bi_loaded_knee_musclesmoment_stiffness_dict['std_knee_total_stiffness'],\
                        # monoarticular loaded
                         mono_loaded_hip_musclesmoment_stiffness_dict['mean_hip_extension_stiffness'],mono_loaded_hip_musclesmoment_stiffness_dict['mean_hip_flexion_stiffness'],mono_loaded_hip_musclesmoment_stiffness_dict['mean_hip_total_stiffness'],\
                         mono_loaded_hip_musclesmoment_stiffness_dict['std_hip_extension_stiffness'],mono_loaded_hip_musclesmoment_stiffness_dict['std_hip_flexion_stiffness'],mono_loaded_hip_musclesmoment_stiffness_dict['std_hip_total_stiffness'],\
                         mono_loaded_knee_musclesmoment_stiffness_dict['mean_knee_extension_stiffness'],mono_loaded_knee_musclesmoment_stiffness_dict['mean_knee_flexion_stiffness'],mono_loaded_knee_musclesmoment_stiffness_dict['mean_knee_total_stiffness'],\
                         mono_loaded_knee_musclesmoment_stiffness_dict['std_knee_extension_stiffness'],mono_loaded_knee_musclesmoment_stiffness_dict['std_knee_flexion_stiffness'],mono_loaded_knee_musclesmoment_stiffness_dict['std_knee_total_stiffness'],\
                        # biarticular noload
                         bi_noload_hip_musclesmoment_stiffness_dict['mean_hip_extension_stiffness'],bi_noload_hip_musclesmoment_stiffness_dict['mean_hip_flexion_stiffness'],bi_noload_hip_musclesmoment_stiffness_dict['mean_hip_total_stiffness'],\
                         bi_noload_hip_musclesmoment_stiffness_dict['std_hip_extension_stiffness'],bi_noload_hip_musclesmoment_stiffness_dict['std_hip_flexion_stiffness'],bi_noload_hip_musclesmoment_stiffness_dict['std_hip_total_stiffness'],\
                         bi_noload_knee_musclesmoment_stiffness_dict['mean_knee_extension_stiffness'],bi_noload_knee_musclesmoment_stiffness_dict['mean_knee_flexion_stiffness'],bi_noload_knee_musclesmoment_stiffness_dict['mean_knee_total_stiffness'],\
                         bi_noload_knee_musclesmoment_stiffness_dict['std_knee_extension_stiffness'],bi_noload_knee_musclesmoment_stiffness_dict['std_knee_flexion_stiffness'],bi_noload_knee_musclesmoment_stiffness_dict['std_knee_total_stiffness'],\
                        # monoarticular noload
                         mono_noload_hip_musclesmoment_stiffness_dict['mean_hip_extension_stiffness'],mono_noload_hip_musclesmoment_stiffness_dict['mean_hip_flexion_stiffness'],mono_noload_hip_musclesmoment_stiffness_dict['mean_hip_total_stiffness'],\
                         mono_noload_hip_musclesmoment_stiffness_dict['std_hip_extension_stiffness'],mono_noload_hip_musclesmoment_stiffness_dict['std_hip_flexion_stiffness'],mono_noload_hip_musclesmoment_stiffness_dict['std_hip_total_stiffness'],\
                         mono_noload_knee_musclesmoment_stiffness_dict['mean_knee_extension_stiffness'],mono_noload_knee_musclesmoment_stiffness_dict['mean_knee_flexion_stiffness'],mono_noload_knee_musclesmoment_stiffness_dict['mean_knee_total_stiffness'],\
                         mono_noload_knee_musclesmoment_stiffness_dict['std_knee_extension_stiffness'],mono_noload_knee_musclesmoment_stiffness_dict['std_knee_flexion_stiffness'],mono_noload_knee_musclesmoment_stiffness_dict['std_knee_total_stiffness']]
actuators_mean_cellText =np.c_[bi_loaded_hip_actuator_stiffness_dict['mean_hip_extension_stiffness'],bi_loaded_hip_actuator_stiffness_dict['mean_hip_flexion_stiffness'],bi_loaded_hip_actuator_stiffness_dict['mean_hip_total_stiffness'],\
                         bi_loaded_hip_actuator_stiffness_dict['std_hip_extension_stiffness'],bi_loaded_hip_actuator_stiffness_dict['std_hip_flexion_stiffness'],bi_loaded_hip_actuator_stiffness_dict['std_hip_total_stiffness'],\
                         bi_loaded_knee_actuator_stiffness_dict['mean_knee_extension_stiffness'],bi_loaded_knee_actuator_stiffness_dict['mean_knee_flexion_stiffness'],bi_loaded_knee_actuator_stiffness_dict['mean_knee_total_stiffness'],\
                         bi_loaded_knee_actuator_stiffness_dict['std_knee_extension_stiffness'],bi_loaded_knee_actuator_stiffness_dict['std_knee_flexion_stiffness'],bi_loaded_knee_actuator_stiffness_dict['std_knee_total_stiffness'],\
                        # monoarticular loaded
                         mono_loaded_hip_actuator_stiffness_dict['mean_hip_extension_stiffness'],mono_loaded_hip_actuator_stiffness_dict['mean_hip_flexion_stiffness'],mono_loaded_hip_actuator_stiffness_dict['mean_hip_total_stiffness'],\
                         mono_loaded_hip_actuator_stiffness_dict['std_hip_extension_stiffness'],mono_loaded_hip_actuator_stiffness_dict['std_hip_flexion_stiffness'],mono_loaded_hip_actuator_stiffness_dict['std_hip_total_stiffness'],\
                         mono_loaded_knee_actuator_stiffness_dict['mean_knee_extension_stiffness'],mono_loaded_knee_actuator_stiffness_dict['mean_knee_flexion_stiffness'],mono_loaded_knee_actuator_stiffness_dict['mean_knee_total_stiffness'],\
                         mono_loaded_knee_actuator_stiffness_dict['std_knee_extension_stiffness'],mono_loaded_knee_actuator_stiffness_dict['std_knee_flexion_stiffness'],mono_loaded_knee_actuator_stiffness_dict['std_knee_total_stiffness'],\
                        # biarticular noload
                         bi_noload_hip_actuator_stiffness_dict['mean_hip_extension_stiffness'],bi_noload_hip_actuator_stiffness_dict['mean_hip_flexion_stiffness'],bi_noload_hip_actuator_stiffness_dict['mean_hip_total_stiffness'],\
                         bi_noload_hip_actuator_stiffness_dict['std_hip_extension_stiffness'],bi_noload_hip_actuator_stiffness_dict['std_hip_flexion_stiffness'],bi_noload_hip_actuator_stiffness_dict['std_hip_total_stiffness'],\
                         bi_noload_knee_actuator_stiffness_dict['mean_knee_extension_stiffness'],bi_noload_knee_actuator_stiffness_dict['mean_knee_flexion_stiffness'],bi_noload_knee_actuator_stiffness_dict['mean_knee_total_stiffness'],\
                         bi_noload_knee_actuator_stiffness_dict['std_knee_extension_stiffness'],bi_noload_knee_actuator_stiffness_dict['std_knee_flexion_stiffness'],bi_noload_knee_actuator_stiffness_dict['std_knee_total_stiffness'],\
                         # monoarticular noload
                         mono_noload_hip_actuator_stiffness_dict['mean_hip_extension_stiffness'],mono_noload_hip_actuator_stiffness_dict['mean_hip_flexion_stiffness'],mono_noload_hip_actuator_stiffness_dict['mean_hip_total_stiffness'],\
                         mono_noload_hip_actuator_stiffness_dict['std_hip_extension_stiffness'],mono_noload_hip_actuator_stiffness_dict['std_hip_flexion_stiffness'],mono_noload_hip_actuator_stiffness_dict['std_hip_total_stiffness'],\
                         mono_noload_knee_actuator_stiffness_dict['mean_knee_extension_stiffness'],mono_noload_knee_actuator_stiffness_dict['mean_knee_flexion_stiffness'],mono_noload_knee_actuator_stiffness_dict['mean_knee_total_stiffness'],\
                         mono_noload_knee_actuator_stiffness_dict['std_knee_extension_stiffness'],mono_noload_knee_actuator_stiffness_dict['std_knee_flexion_stiffness'],mono_noload_knee_actuator_stiffness_dict['std_knee_total_stiffness']]
actuators_header = []
muscles_header = []
for load in ['loaded','noload']:
        for device in ['bi','mono']:
                for joint in ['hip','knee']:
                        for data_type in ['mean','std']:
                                for phase in ['extension','flexion','total']:
                                        actuators_header.append('{}_{}_{}actuator_{}_{}_stiffness'.format(device,load,joint,data_type,phase))
                                        muscles_header.append('{}_{}_{}muscles_{}_{}_stiffness'.format(device,load,joint,data_type,phase))
# save data
headers = actuators_header + muscles_header
dataset = np.concatenate((actuators_mean_cellText,muscles_mean_cellText),axis=1)
with open(r'.\Data\Ideal\ideal_exos_stiffness_dataset.csv', 'wb') as f:
  f.write(bytes(utils.listToString(headers)+'\n','UTF-8'))
  np.savetxt(f, dataset, fmt='%s', delimiter=",")                                     
##########################################################################################################################################################
# joints stiffness
fig, ax = plt.subplots(nrows=2,ncols=2,figsize=(12.8, 9.6))
#*******************************
# hip loaded muscles boxplot
# box plot
names = ['extension,\n unassist','flexion,\n unassist','extension,\n bi','flexion,\n bi','extension,\n mono','flexion,\n mono']
x = np.arange(1,len(names)+1,1)
data = [stiffness_dataset['loaded_hip_extension_stiffness'],stiffness_dataset['loaded_hip_flexion_stiffness'],\
        bi_loaded_hip_musclesmoment_stiffness_dict['hip_extension_stiffness'],bi_loaded_hip_musclesmoment_stiffness_dict['hip_flexion_stiffness'],\
        mono_loaded_hip_musclesmoment_stiffness_dict['hip_extension_stiffness'],mono_loaded_hip_musclesmoment_stiffness_dict['hip_flexion_stiffness']]
bp = ax[0,0].boxplot(data, patch_artist=True)
utils.beautiful_boxplot(bp)
ax[0,0].set_ylabel('Joint Stiffness (N-m/kg-rad)')
ax[0,0].set_xticks(x)
ax[0,0].set_yticks([0,2,4,6,8])
ax[0,0].set_xticklabels(names)
ax[0,0].set_title('hip muscles stiffness, loaded')
ax[0,0].tick_params(axis='both',direction='in')
utils.no_top_right(ax[0,0])
#*******************************
# hip noload muscles boxplot
# box plot
names = ['extension,\n unassist','flexion,\n unassist','extension,\n bi','flexion,\n bi','extension,\n mono','flexion,\n mono']
x = np.arange(1,len(names)+1,1)
data = [stiffness_dataset['noload_hip_extension_stiffness'],stiffness_dataset['noload_hip_flexion_stiffness'],\
        bi_noload_hip_musclesmoment_stiffness_dict['hip_extension_stiffness'],bi_noload_hip_musclesmoment_stiffness_dict['hip_flexion_stiffness'],\
        mono_noload_hip_musclesmoment_stiffness_dict['hip_extension_stiffness'],mono_noload_hip_musclesmoment_stiffness_dict['hip_flexion_stiffness']]
bp = ax[0,1].boxplot(data, patch_artist=True)
utils.beautiful_boxplot(bp)
ax[0,1].set_ylabel('Joint Stiffness (N-m/kg-rad)')
ax[0,1].set_xticks(x)
ax[0,1].set_yticks([0,2,4,6,8])
ax[0,1].set_xticklabels(names)
ax[0,1].set_title('hip muscles stiffness, noload')
ax[0,1].tick_params(axis='both',direction='in')
utils.no_top_right(ax[0,1])
#*******************************
# knee loaded muscles boxplot
# box plot
names = ['extension,\n unassist','flexion,\n unassist','extension,\n bi','flexion,\n bi','extension,\n mono','flexion,\n mono']
x = np.arange(1,len(names)+1,1)
data = [stiffness_dataset['loaded_knee_extension_stiffness'],stiffness_dataset['loaded_knee_flexion_stiffness'],\
        bi_loaded_knee_musclesmoment_stiffness_dict['knee_extension_stiffness'],bi_loaded_knee_musclesmoment_stiffness_dict['knee_flexion_stiffness'],\
        mono_loaded_knee_musclesmoment_stiffness_dict['knee_extension_stiffness'],mono_loaded_knee_musclesmoment_stiffness_dict['knee_flexion_stiffness']]
bp = ax[1,0].boxplot(data, patch_artist=True)
utils.beautiful_boxplot(bp)
ax[1,0].set_ylabel('Joint Stiffness (N-m/kg-rad)')
ax[1,0].set_xticks(x)
ax[1,0].set_yticks([0,2,4,6,8,10,12,14])
ax[1,0].set_xticklabels(names)
ax[1,0].set_title('knee muscles stiffness, loaded')
ax[1,0].tick_params(axis='both',direction='in')
utils.no_top_right(ax[1,0])
#*******************************
# knee noload muscles boxplot
# box plot
names = ['extension,\n unassist','flexion,\n unassist','extension,\n bi','flexion,\n bi','extension,\n mono','flexion,\n mono']
x = np.arange(1,len(names)+1,1)
data = [stiffness_dataset['noload_knee_extension_stiffness'],stiffness_dataset['noload_knee_flexion_stiffness'],\
        bi_noload_knee_musclesmoment_stiffness_dict['knee_extension_stiffness'],bi_noload_knee_musclesmoment_stiffness_dict['knee_flexion_stiffness'],\
        mono_noload_knee_musclesmoment_stiffness_dict['knee_extension_stiffness'],mono_noload_knee_musclesmoment_stiffness_dict['knee_flexion_stiffness']]
bp = ax[1,1].boxplot(data, patch_artist=True)
utils.beautiful_boxplot(bp)
ax[1,1].set_xticks(x)
ax[1,0].set_ylabel('Joint Stiffness (N-m/kg-rad)')
ax[1,1].set_yticks([0,2,4,6,8,10])
ax[1,1].set_xticklabels(names)
ax[1,1].set_title('knee muscles stiffness, noload')
ax[1,1].tick_params(axis='both',direction='in')
utils.no_top_right(ax[1,1])
fig.tight_layout(h_pad=-1, w_pad=-1.5)
fig.subplots_adjust(top=0.98, bottom=0.075, left=0.100, right=0.975,hspace=0.45,wspace=0.25)
plt.show()
fig.savefig('./Figures/Ideal/Joint_Stiffness_BoxPlot.pdf',orientation='landscape',bbox_inches='tight')
#*******************************
fig, ax = plt.subplots(nrows=2,ncols=2,num='actuators stiffness mono vs bi',figsize=(12.8, 9.6))
#*******************************
# hip loaded actuators boxplot
# box plot
names = ['extension,\n bi','flexion,\n bi','extension,\n mono','flexion,\n mono']
x = np.arange(1,len(names)+1,1)
data = [bi_loaded_hip_actuator_stiffness_dict['hip_extension_stiffness'],bi_loaded_hip_actuator_stiffness_dict['hip_flexion_stiffness'],\
        mono_loaded_hip_actuator_stiffness_dict['hip_extension_stiffness'],mono_loaded_hip_actuator_stiffness_dict['hip_flexion_stiffness']]
bp = ax[0,0].boxplot(data, patch_artist=True)
utils.beautiful_boxplot(bp)
ax[0,0].set_ylabel('Joint Stiffness (N-m/kg-rad)')
ax[0,0].set_xticks(x)
ax[0,0].set_yticks([0,2,4,6,8,10])
ax[0,0].set_xticklabels(names)
ax[0,0].set_title('hip actuator stiffness, loaded')
ax[0,0].tick_params(axis='both',direction='in')
utils.no_top_right(ax[0,0])
#*******************************
# hip noload actuator boxplot
# box plot
names = ['extension,\n bi','flexion,\n bi','extension,\n mono','flexion,\n mono']
x = np.arange(1,len(names)+1,1)
data = [bi_noload_hip_actuator_stiffness_dict['hip_extension_stiffness'],bi_noload_hip_actuator_stiffness_dict['hip_flexion_stiffness'],\
        mono_noload_hip_actuator_stiffness_dict['hip_extension_stiffness'],mono_noload_hip_actuator_stiffness_dict['hip_flexion_stiffness']]
bp = ax[0,1].boxplot(data, patch_artist=True)
utils.beautiful_boxplot(bp)
ax[0,1].set_ylabel('Joint Stiffness (N-m/kg-rad)')
ax[0,1].set_xticks(x)
ax[0,1].set_yticks([0,2,4,6,8,10])
ax[0,1].set_xticklabels(names)
ax[0,1].set_title('hip actuator stiffness, noload')
ax[0,1].tick_params(axis='both',direction='in')
utils.no_top_right(ax[0,1])
#*******************************
# knee loaded actuator boxplot
# box plot
names = ['extension,\n loaded','flexion,\n bi','extension,\n mono','flexion,\n mono']
x = np.arange(1,len(names)+1,1)
data = [bi_loaded_knee_actuator_stiffness_dict['knee_extension_stiffness'],bi_loaded_knee_actuator_stiffness_dict['knee_flexion_stiffness'],\
        mono_loaded_knee_actuator_stiffness_dict['knee_extension_stiffness'],mono_loaded_knee_actuator_stiffness_dict['knee_flexion_stiffness']]
bp = ax[1,0].boxplot(data, patch_artist=True)
utils.beautiful_boxplot(bp)
ax[1,0].set_ylabel('Joint Stiffness (N-m/kg-rad)')
ax[1,0].set_xticks(x)
ax[1,0].set_yticks([0,2,4,6,8,10,12,14])
ax[1,0].set_xticklabels(names)
ax[1,0].set_title('knee muscles stiffness, loaded')
ax[1,0].tick_params(axis='both',direction='in')
utils.no_top_right(ax[1,0])
#*******************************
# knee noload muscles boxplot
# box plot
names = ['extension,\n bi','flexion,\n bi','extension,\n mono','flexion,\n mono']
x = np.arange(1,len(names)+1,1)
data = [bi_noload_knee_actuator_stiffness_dict['knee_extension_stiffness'],bi_noload_knee_actuator_stiffness_dict['knee_flexion_stiffness'],\
        mono_noload_knee_actuator_stiffness_dict['knee_extension_stiffness'],mono_noload_knee_actuator_stiffness_dict['knee_flexion_stiffness']]
bp = ax[1,1].boxplot(data, patch_artist=True)
utils.beautiful_boxplot(bp)
ax[1,1].set_xticks(x)
ax[1,0].set_ylabel('Joint Stiffness (N-m/kg-rad)')
ax[1,1].set_yticks([0,2,4,6,8,10])
ax[1,1].set_xticklabels(names)
ax[1,1].set_title('knee muscles stiffness, noload')
ax[1,1].tick_params(axis='both',direction='in')
utils.no_top_right(ax[1,1])
fig.tight_layout(h_pad=-1, w_pad=-1.5)
fig.subplots_adjust(top=0.98, bottom=0.075, left=0.100, right=0.975,hspace=0.45,wspace=0.25)
plt.show()
fig.savefig('./Figures/Ideal/Actuators_Stiffness_BoxPlot_DeviceComparison.pdf',orientation='landscape',bbox_inches='tight')
#*******************************
fig, ax = plt.subplots(nrows=2,ncols=2,num='actuators stiffness loaded vs noload',figsize=(12.8, 9.6))
#*******************************
# hip loaded actuators boxplot
# box plot
names = ['extension,\n loaded','flexion,\n loaded','extension,\n noload','flexion,\n noload']
x = np.arange(1,len(names)+1,1)
data = [bi_loaded_hip_actuator_stiffness_dict['hip_extension_stiffness'],bi_loaded_hip_actuator_stiffness_dict['hip_flexion_stiffness'],\
        bi_noload_hip_actuator_stiffness_dict['hip_extension_stiffness'],bi_noload_hip_actuator_stiffness_dict['hip_flexion_stiffness']]
bp = ax[0,0].boxplot(data, patch_artist=True)
utils.beautiful_boxplot(bp)
ax[0,0].set_ylabel('Joint Stiffness (N-m/kg-rad)')
ax[0,0].set_xticks(x)
ax[0,0].set_yticks([0,2,4,6,8,10])
ax[0,0].set_xticklabels(names)
ax[0,0].set_title('hip actuator stiffness, biarticular')
ax[0,0].tick_params(axis='both',direction='in')
utils.no_top_right(ax[0,0])
#*******************************
# hip noload actuator boxplot
# box plot
names = ['extension,\n loaded','flexion,\n loaded','extension,\n noload','flexion,\n noload']
x = np.arange(1,len(names)+1,1)
data = [mono_loaded_hip_actuator_stiffness_dict['hip_extension_stiffness'],mono_loaded_hip_actuator_stiffness_dict['hip_flexion_stiffness'],\
        mono_noload_hip_actuator_stiffness_dict['hip_extension_stiffness'],mono_noload_hip_actuator_stiffness_dict['hip_flexion_stiffness']]
bp = ax[0,1].boxplot(data, patch_artist=True)
utils.beautiful_boxplot(bp)
ax[0,1].set_ylabel('Joint Stiffness (N-m/kg-rad)')
ax[0,1].set_xticks(x)
ax[0,1].set_yticks([0,2,4,6,8,10])
ax[0,1].set_xticklabels(names)
ax[0,1].set_title('hip actuator stiffness, monoarticular')
ax[0,1].tick_params(axis='both',direction='in')
utils.no_top_right(ax[0,1])
#*******************************
# knee loaded actuator boxplot
# box plot
names = ['extension,\n loaded','flexion,\n loaded','extension,\n noload','flexion,\n noload']
x = np.arange(1,len(names)+1,1)
data = [bi_loaded_knee_actuator_stiffness_dict['knee_extension_stiffness'],bi_loaded_knee_actuator_stiffness_dict['knee_flexion_stiffness'],\
        bi_noload_knee_actuator_stiffness_dict['knee_extension_stiffness'],bi_noload_knee_actuator_stiffness_dict['knee_flexion_stiffness']]
bp = ax[1,0].boxplot(data, patch_artist=True)
utils.beautiful_boxplot(bp)
ax[1,0].set_ylabel('Joint Stiffness (N-m/kg-rad)')
ax[1,0].set_xticks(x)
ax[1,0].set_yticks([0,2,4,6,8,10,12,14])
ax[1,0].set_xticklabels(names)
ax[1,0].set_title('knee muscles stiffness, biarticular')
ax[1,0].tick_params(axis='both',direction='in')
utils.no_top_right(ax[1,0])
#*******************************
# knee noload muscles boxplot
# box plot
names = ['extension,\n loaded','flexion,\n loaded','extension,\n noload','flexion,\n noload']
x = np.arange(1,len(names)+1,1)
data = [mono_loaded_knee_actuator_stiffness_dict['knee_extension_stiffness'],mono_loaded_knee_actuator_stiffness_dict['knee_flexion_stiffness'],\
        mono_noload_knee_actuator_stiffness_dict['knee_extension_stiffness'],mono_noload_knee_actuator_stiffness_dict['knee_flexion_stiffness']]
bp = ax[1,1].boxplot(data, patch_artist=True)
utils.beautiful_boxplot(bp)
ax[1,1].set_xticks(x)
ax[1,0].set_ylabel('Joint Stiffness (N-m/kg-rad)')
ax[1,1].set_yticks([0,2,4,6,8,10])
ax[1,1].set_xticklabels(names)
ax[1,1].set_title('knee muscles stiffness, monoarticular')
ax[1,1].tick_params(axis='both',direction='in')
utils.no_top_right(ax[1,1])
fig.tight_layout(h_pad=-1, w_pad=-1.5)
fig.subplots_adjust(top=0.98, bottom=0.075, left=0.100, right=0.975,hspace=0.45,wspace=0.25)
plt.show()
fig.savefig('./Figures/Ideal/Actuators_Stiffness_BoxPlot_LoadComparison.pdf',orientation='landscape',bbox_inches='tight')

##########################################################################################################################################################
# biarticular loaded vs noload hip joint stiffness
fig = plt.figure(num='Hip Joint Stiffness',figsize=(15, 12))
gridsize = (3, 4)
ax1 = plt.subplot2grid(gridsize, (0, 0), colspan=2, rowspan=2) # stiffness plot
ax2 = plt.subplot2grid(gridsize, (2, 0)) # kinematics plot
ax3 = plt.subplot2grid(gridsize, (2, 1)) # moment plot
ax4 = plt.subplot2grid(gridsize, (0, 2), colspan=2, rowspan=2) # stiffness plot
ax5 = plt.subplot2grid(gridsize, (2, 2)) # kinematics plot
ax6 = plt.subplot2grid(gridsize, (2, 3)) # moment plot
# hip loaded case
hip_stiffness_plot_dic = {'loaded_toe_off':loaded_mean_toe_off,'noload_toe_off':noload_mean_toe_off,'kinematics':utils.smooth(rra_dataset['mean_loaded_hipjoint_kinematics'],5),
                          'kinematics_std':utils.smooth(rra_dataset['std_loaded_hipjoint_kinematics'],5),'moment':utils.smooth(unnormalized_moment_dataset['mean_loaded_hipmuscles_moment'],5),
                          'moment_std':utils.smooth(unnormalized_moment_dataset['std_loaded_hipmuscles_moment'],5),'color':'k','toe_off_color':'grey','label':'hip joint'}
utils.plot_stiffness(plot_dic = hip_stiffness_plot_dic, load_condition='loaded',\
                     kinematics_ticks=[-20,-10,0,10,20,30,40,50],moment_ticks=[-2,-1,0,1,2],ax1=ax1,ax2=ax2,ax3=ax3)
# hip noload case
hip_stiffness_plot_dic = {'loaded_toe_off':loaded_mean_toe_off,'noload_toe_off':noload_mean_toe_off,'kinematics':utils.smooth(rra_dataset['mean_noload_hipjoint_kinematics'],5),
                          'kinematics_std':utils.smooth(rra_dataset['std_noload_hipjoint_kinematics'],5),'moment':utils.smooth(unnormalized_moment_dataset['mean_noload_hipmuscles_moment'],5),
                          'moment_std':utils.smooth(unnormalized_moment_dataset['std_noload_hipmuscles_moment'],5),'color':'xkcd:irish green','toe_off_color':'xkcd:shamrock green','label':'hip joint'}
utils.plot_stiffness(plot_dic = hip_stiffness_plot_dic, load_condition='noload',
kinematics_ticks=[-20,-10,0,10,20,30,40,50],moment_ticks=[-2,-1,0,1,2],ax1=ax4,ax2=ax5,ax3=ax6)
# biarticular hip loaded case
hip_stiffness_plot_dic = {'loaded_toe_off':loaded_mean_toe_off,'noload_toe_off':noload_mean_toe_off,'kinematics':utils.smooth(mean_bi_loaded_hip_kinematics,5),
                          'kinematics_std':utils.smooth(std_bi_loaded_hip_kinematics,5),'moment':utils.smooth(mean_bi_loaded_hip_torque,5),
                          'moment_std':utils.smooth(std_bi_loaded_hip_torque,5),'color':mycolors['cyan blue'],'toe_off_color':mycolors['cyan blue'],'label':'bi loaded hip actuator'}
utils.plot_stiffness(plot_dic = hip_stiffness_plot_dic, load_condition='loaded',\
                     kinematics_ticks=[-20,-10,0,10,20,30,40,50],moment_ticks=[-2,-1,0,1,2],ax1=ax1,ax2=ax2,ax3=ax3)
# biarticular hip noload case
hip_stiffness_plot_dic = {'loaded_toe_off':loaded_mean_toe_off,'noload_toe_off':noload_mean_toe_off,'kinematics':utils.smooth(mean_bi_noload_hip_kinematics,5),
                          'kinematics_std':utils.smooth(std_bi_noload_hip_kinematics,5),'moment':utils.smooth(mean_bi_noload_hip_torque,5),
                          'moment_std':utils.smooth(std_bi_noload_hip_torque,5),'color':mycolors['olympic blue'],'toe_off_color':mycolors['olympic blue'],'label':'bi noload hip actuator'}
utils.plot_stiffness(plot_dic = hip_stiffness_plot_dic, load_condition='noload',
kinematics_ticks=[-20,-10,0,10,20,30,40,50],moment_ticks=[-2,-1,0,1,2],ax1=ax4,ax2=ax5,ax3=ax6)
# biarticular hip muscles loaded case
hip_stiffness_plot_dic = {'loaded_toe_off':loaded_mean_toe_off,'noload_toe_off':noload_mean_toe_off,'kinematics':utils.smooth(mean_bi_loaded_hip_kinematics,5),
                          'kinematics_std':utils.smooth(std_bi_loaded_hip_kinematics,5),'moment':utils.smooth(mean_bi_loaded_hipmuscles_moment,5),
                          'moment_std':utils.smooth(std_bi_loaded_hipmuscles_moment,5),'color':mycolors['crimson red'],'toe_off_color':mycolors['crimson red'],'label':'bi loaded hip muscles'}
utils.plot_stiffness(plot_dic = hip_stiffness_plot_dic, load_condition='loaded',\
                     kinematics_ticks=[-20,-10,0,10,20,30,40,50],moment_ticks=[-2,-1,0,1,2],ax1=ax1,ax2=ax2,ax3=ax3)
# biarticular hip muscles noload case
hip_stiffness_plot_dic = {'loaded_toe_off':loaded_mean_toe_off,'noload_toe_off':noload_mean_toe_off,'kinematics':utils.smooth(mean_bi_noload_hip_kinematics,5),
                          'kinematics_std':utils.smooth(std_bi_noload_hip_kinematics,5),'moment':utils.smooth(mean_bi_noload_hipmuscles_moment,5),
                          'moment_std':utils.smooth(std_bi_noload_hipmuscles_moment,5),'color':mycolors['french rose'],'toe_off_color':mycolors['french rose'],'label':'bi noload hip muscles'}
utils.plot_stiffness(plot_dic = hip_stiffness_plot_dic, load_condition='noload',
kinematics_ticks=[-20,-10,0,10,20,30,40,50],moment_ticks=[-2,-1,0,1,2],ax1=ax4,ax2=ax5,ax3=ax6)

fig.tight_layout(h_pad=-1, w_pad=-1.5)
fig.subplots_adjust(top=0.98, bottom=0.075, left=0.100, right=0.975,hspace=0.45,wspace=0.35)
plt.show()
fig.savefig('./Figures/Ideal/Biarticular_LoadedVsNoload_HipActuator_Stiffness.pdf',orientation='landscape',bbox_inches='tight')
#************************************************************
# monoarticular loaded vs noload hip joint stiffness
fig = plt.figure(num='Hip Joint Stiffness',figsize=(15, 12))
gridsize = (3, 4)
ax1 = plt.subplot2grid(gridsize, (0, 0), colspan=2, rowspan=2) # stiffness plot
ax2 = plt.subplot2grid(gridsize, (2, 0)) # kinematics plot
ax3 = plt.subplot2grid(gridsize, (2, 1)) # moment plot
ax4 = plt.subplot2grid(gridsize, (0, 2), colspan=2, rowspan=2) # stiffness plot
ax5 = plt.subplot2grid(gridsize, (2, 2)) # kinematics plot
ax6 = plt.subplot2grid(gridsize, (2, 3)) # moment plot
# hip loaded case
hip_stiffness_plot_dic = {'loaded_toe_off':loaded_mean_toe_off,'noload_toe_off':noload_mean_toe_off,'kinematics':utils.smooth(rra_dataset['mean_loaded_hipjoint_kinematics'],5),
                          'kinematics_std':utils.smooth(rra_dataset['std_loaded_hipjoint_kinematics'],5),'moment':utils.smooth(unnormalized_moment_dataset['mean_loaded_hipmuscles_moment'],5),
                          'moment_std':utils.smooth(unnormalized_moment_dataset['std_loaded_hipmuscles_moment'],5),'color':'k','toe_off_color':'grey','label':'hip joint'}
utils.plot_stiffness(plot_dic = hip_stiffness_plot_dic, load_condition='loaded',\
                     kinematics_ticks=[-20,-10,0,10,20,30,40,50],moment_ticks=[-2,-1,0,1,2],ax1=ax1,ax2=ax2,ax3=ax3)
# hip noload case
hip_stiffness_plot_dic = {'loaded_toe_off':loaded_mean_toe_off,'noload_toe_off':noload_mean_toe_off,'kinematics':utils.smooth(rra_dataset['mean_noload_hipjoint_kinematics'],5),
                          'kinematics_std':utils.smooth(rra_dataset['std_noload_hipjoint_kinematics'],5),'moment':utils.smooth(unnormalized_moment_dataset['mean_noload_hipmuscles_moment'],5),
                          'moment_std':utils.smooth(unnormalized_moment_dataset['std_noload_hipmuscles_moment'],5),'color':'xkcd:irish green','toe_off_color':'xkcd:shamrock green','label':'hip joint'}
utils.plot_stiffness(plot_dic = hip_stiffness_plot_dic, load_condition='noload',
kinematics_ticks=[-20,-10,0,10,20,30,40,50],moment_ticks=[-2,-1,0,1,2],ax1=ax4,ax2=ax5,ax3=ax6)
# monoarticular hip loaded case
hip_stiffness_plot_dic = {'loaded_toe_off':loaded_mean_toe_off,'noload_toe_off':noload_mean_toe_off,'kinematics':utils.smooth(mean_mono_loaded_hip_kinematics,5),
                          'kinematics_std':utils.smooth(std_mono_loaded_hip_kinematics,5),'moment':utils.smooth(mean_mono_loaded_hip_torque,5),
                          'moment_std':utils.smooth(std_mono_loaded_hip_torque,5),'color':mycolors['cyan blue'],'toe_off_color':mycolors['cyan blue'],'label':'bi loaded hip actuator'}
utils.plot_stiffness(plot_dic = hip_stiffness_plot_dic, load_condition='loaded',\
                     kinematics_ticks=[-20,-10,0,10,20,30,40,50],moment_ticks=[-2,-1,0,1,2],ax1=ax1,ax2=ax2,ax3=ax3)
# monoarticular hip noload case
hip_stiffness_plot_dic = {'loaded_toe_off':loaded_mean_toe_off,'noload_toe_off':noload_mean_toe_off,'kinematics':utils.smooth(mean_mono_noload_hip_kinematics,5),
                          'kinematics_std':utils.smooth(std_mono_noload_hip_kinematics,5),'moment':utils.smooth(mean_mono_noload_hip_torque,5),
                          'moment_std':utils.smooth(std_mono_noload_hip_torque,5),'color':mycolors['olympic blue'],'toe_off_color':mycolors['olympic blue'],'label':'bi noload hip actuator'}
utils.plot_stiffness(plot_dic = hip_stiffness_plot_dic, load_condition='noload',
kinematics_ticks=[-20,-10,0,10,20,30,40,50],moment_ticks=[-2,-1,0,1,2],ax1=ax4,ax2=ax5,ax3=ax6)
# monoarticular hip muscles loaded case
hip_stiffness_plot_dic = {'loaded_toe_off':loaded_mean_toe_off,'noload_toe_off':noload_mean_toe_off,'kinematics':utils.smooth(mean_mono_loaded_hip_kinematics,5),
                          'kinematics_std':utils.smooth(std_mono_loaded_hip_kinematics,5),'moment':utils.smooth(mean_mono_loaded_hipmuscles_moment,5),
                          'moment_std':utils.smooth(std_mono_loaded_hipmuscles_moment,5),'color':mycolors['crimson red'],'toe_off_color':mycolors['crimson red'],'label':'bi loaded hip muscles'}
utils.plot_stiffness(plot_dic = hip_stiffness_plot_dic, load_condition='loaded',\
                     kinematics_ticks=[-20,-10,0,10,20,30,40,50],moment_ticks=[-2,-1,0,1,2],ax1=ax1,ax2=ax2,ax3=ax3)
# monoarticular hip muscles noload case
hip_stiffness_plot_dic = {'loaded_toe_off':loaded_mean_toe_off,'noload_toe_off':noload_mean_toe_off,'kinematics':utils.smooth(mean_mono_noload_hip_kinematics,5),
                          'kinematics_std':utils.smooth(std_mono_noload_hip_kinematics,5),'moment':utils.smooth(mean_mono_noload_hipmuscles_moment,5),
                          'moment_std':utils.smooth(std_mono_noload_hipmuscles_moment,5),'color':mycolors['french rose'],'toe_off_color':mycolors['french rose'],'label':'bi noload hip muscles'}
utils.plot_stiffness(plot_dic = hip_stiffness_plot_dic, load_condition='noload',
kinematics_ticks=[-20,-10,0,10,20,30,40,50],moment_ticks=[-2,-1,0,1,2],ax1=ax4,ax2=ax5,ax3=ax6)

fig.tight_layout(h_pad=-1, w_pad=-1.5)
fig.subplots_adjust(top=0.98, bottom=0.075, left=0.100, right=0.975,hspace=0.45,wspace=0.35)
plt.show()
fig.savefig('./Figures/Ideal/Monoarticular_LoadedVsNoload_HipActuator_Stiffness.pdf',orientation='landscape',bbox_inches='tight')
#************************************************************
#************************************************************
# biarticular loaded vs noload knee joint stiffness
fig = plt.figure(num='Knee Joint Stiffness',figsize=(15, 12))
gridsize = (3, 4)
ax1 = plt.subplot2grid(gridsize, (0, 0), colspan=2, rowspan=2) # stiffness plot
ax2 = plt.subplot2grid(gridsize, (2, 0)) # kinematics plot
ax3 = plt.subplot2grid(gridsize, (2, 1)) # moment plot
ax4 = plt.subplot2grid(gridsize, (0, 2), colspan=2, rowspan=2) # stiffness plot
ax5 = plt.subplot2grid(gridsize, (2, 2)) # kinematics plot
ax6 = plt.subplot2grid(gridsize, (2, 3)) # moment plot
# knee loaded case
knee_stiffness_plot_dic = {'loaded_toe_off':loaded_mean_toe_off,'noload_toe_off':noload_mean_toe_off,'kinematics':utils.smooth(rra_dataset['mean_loaded_kneejoint_kinematics'],5),
                          'kinematics_std':utils.smooth(rra_dataset['std_loaded_kneejoint_kinematics'],5),'moment':utils.smooth(unnormalized_moment_dataset['mean_loaded_kneemuscles_moment'],5),
                          'moment_std':utils.smooth(unnormalized_moment_dataset['std_loaded_kneemuscles_moment'],5),'color':'k','toe_off_color':'grey','label':'knee joint'}
utils.plot_stiffness(plot_dic = knee_stiffness_plot_dic, load_condition='loaded',\
                     kinematics_ticks=[-40,-20,0,20,40,60,80],moment_ticks=[-2,-1,0,1,2],ax1=ax1,ax2=ax2,ax3=ax3)
# knee noload case
knee_stiffness_plot_dic = {'loaded_toe_off':loaded_mean_toe_off,'noload_toe_off':noload_mean_toe_off,'kinematics':utils.smooth(rra_dataset['mean_noload_kneejoint_kinematics'],5),
                          'kinematics_std':utils.smooth(rra_dataset['std_noload_kneejoint_kinematics'],5),'moment':utils.smooth(unnormalized_moment_dataset['mean_noload_kneemuscles_moment'],5),
                          'moment_std':utils.smooth(unnormalized_moment_dataset['std_noload_kneemuscles_moment'],5),'color':'xkcd:irish green','toe_off_color':'xkcd:shamrock green','label':'knee joint'}
utils.plot_stiffness(plot_dic = knee_stiffness_plot_dic, load_condition='noload',
kinematics_ticks=[-40,-20,0,20,40,60,80],moment_ticks=[-2,-1,0,1,2],ax1=ax4,ax2=ax5,ax3=ax6)
# biarticular knee loaded case
knee_stiffness_plot_dic = {'loaded_toe_off':loaded_mean_toe_off,'noload_toe_off':noload_mean_toe_off,'kinematics':utils.smooth(mean_bi_loaded_knee_kinematics,5),
                          'kinematics_std':utils.smooth(std_bi_loaded_knee_kinematics,5),'moment':utils.smooth(mean_bi_loaded_knee_torque,5),
                          'moment_std':utils.smooth(std_bi_loaded_knee_torque,5),'color':mycolors['cyan blue'],'toe_off_color':mycolors['cyan blue'],'label':'bi loaded knee actuator'}
utils.plot_stiffness(plot_dic = knee_stiffness_plot_dic, load_condition='loaded',\
                     kinematics_ticks=[-40,-20,0,20,40,60,80],moment_ticks=[-2,-1,0,1,2],ax1=ax1,ax2=ax2,ax3=ax3)
# biarticular knee noload case
knee_stiffness_plot_dic = {'loaded_toe_off':loaded_mean_toe_off,'noload_toe_off':noload_mean_toe_off,'kinematics':utils.smooth(mean_bi_noload_knee_kinematics,5),
                          'kinematics_std':utils.smooth(std_bi_noload_knee_kinematics,5),'moment':utils.smooth(mean_bi_noload_knee_torque,5),
                          'moment_std':utils.smooth(std_bi_noload_knee_torque,5),'color':mycolors['olympic blue'],'toe_off_color':mycolors['olympic blue'],'label':'bi noload knee actuator'}
utils.plot_stiffness(plot_dic = knee_stiffness_plot_dic, load_condition='noload',
kinematics_ticks=[-40,-20,0,20,40,60,80],moment_ticks=[-2,-1,0,1,2],ax1=ax4,ax2=ax5,ax3=ax6)
# biarticular knee muscles loaded case
knee_stiffness_plot_dic = {'loaded_toe_off':loaded_mean_toe_off,'noload_toe_off':noload_mean_toe_off,'kinematics':utils.smooth(mean_mono_loaded_knee_kinematics,5),
                          'kinematics_std':utils.smooth(std_mono_loaded_knee_kinematics,5),'moment':utils.smooth(mean_bi_loaded_kneemuscles_moment,5),
                          'moment_std':utils.smooth(std_bi_loaded_kneemuscles_moment,5),'color':mycolors['crimson red'],'toe_off_color':mycolors['crimson red'],'label':'bi loaded knee muscles'}
utils.plot_stiffness(plot_dic = knee_stiffness_plot_dic, load_condition='loaded',\
                     kinematics_ticks=[-40,-20,0,20,40,60,80],moment_ticks=[-2,-1,0,1,2],ax1=ax1,ax2=ax2,ax3=ax3)
# biarticular knee muscles noload case
knee_stiffness_plot_dic = {'loaded_toe_off':loaded_mean_toe_off,'noload_toe_off':noload_mean_toe_off,'kinematics':utils.smooth(mean_mono_noload_knee_kinematics,5),
                          'kinematics_std':utils.smooth(std_mono_noload_knee_kinematics,5),'moment':utils.smooth(mean_bi_noload_kneemuscles_moment,5),
                          'moment_std':utils.smooth(std_bi_noload_kneemuscles_moment,5),'color':mycolors['french rose'],'toe_off_color':mycolors['french rose'],'label':'bi noload knee muscles'}
utils.plot_stiffness(plot_dic = knee_stiffness_plot_dic, load_condition='noload',
kinematics_ticks=[-40,-20,0,20,40,60,80],moment_ticks=[-2,-1,0,1,2],ax1=ax4,ax2=ax5,ax3=ax6)

fig.tight_layout(h_pad=-1, w_pad=-1.5)
fig.subplots_adjust(top=0.98, bottom=0.075, left=0.100, right=0.975,hspace=0.45,wspace=0.35)
plt.show()
fig.savefig('./Figures/Ideal/Biarticular_LoadedVsNoload_KneeActuator_Stiffness.pdf',orientation='landscape',bbox_inches='tight')
#************************************************************
# monoarticular loaded vs noload knee joint stiffness
fig = plt.figure(num='Knee Joint Stiffness',figsize=(15, 12))
gridsize = (3, 4)
ax1 = plt.subplot2grid(gridsize, (0, 0), colspan=2, rowspan=2) # stiffness plot
ax2 = plt.subplot2grid(gridsize, (2, 0)) # kinematics plot
ax3 = plt.subplot2grid(gridsize, (2, 1)) # moment plot
ax4 = plt.subplot2grid(gridsize, (0, 2), colspan=2, rowspan=2) # stiffness plot
ax5 = plt.subplot2grid(gridsize, (2, 2)) # kinematics plot
ax6 = plt.subplot2grid(gridsize, (2, 3)) # moment plot
# knee loaded case
knee_stiffness_plot_dic = {'loaded_toe_off':loaded_mean_toe_off,'noload_toe_off':noload_mean_toe_off,'kinematics':utils.smooth(rra_dataset['mean_loaded_kneejoint_kinematics'],5),
                          'kinematics_std':utils.smooth(rra_dataset['std_loaded_kneejoint_kinematics'],5),'moment':utils.smooth(unnormalized_moment_dataset['mean_loaded_kneemuscles_moment'],5),
                          'moment_std':utils.smooth(unnormalized_moment_dataset['std_loaded_kneemuscles_moment'],5),'color':'k','toe_off_color':'grey','label':'knee joint'}
utils.plot_stiffness(plot_dic = knee_stiffness_plot_dic, load_condition='loaded',\
                     kinematics_ticks=[0,10,20,30,40,50,60,70,80],moment_ticks=[-2,-1,0,1,2],ax1=ax1,ax2=ax2,ax3=ax3)
# knee noload case
knee_stiffness_plot_dic = {'loaded_toe_off':loaded_mean_toe_off,'noload_toe_off':noload_mean_toe_off,'kinematics':utils.smooth(rra_dataset['mean_noload_kneejoint_kinematics'],5),
                          'kinematics_std':utils.smooth(rra_dataset['std_noload_kneejoint_kinematics'],5),'moment':utils.smooth(unnormalized_moment_dataset['mean_noload_kneemuscles_moment'],5),
                          'moment_std':utils.smooth(unnormalized_moment_dataset['std_noload_kneemuscles_moment'],5),'color':'xkcd:irish green','toe_off_color':'xkcd:shamrock green','label':'knee joint'}
utils.plot_stiffness(plot_dic = knee_stiffness_plot_dic, load_condition='noload',
kinematics_ticks=[0,10,20,30,40,50,60,70,80],moment_ticks=[-2,-1,0,1,2],ax1=ax4,ax2=ax5,ax3=ax6)
# monoarticular knee loaded case
knee_stiffness_plot_dic = {'loaded_toe_off':loaded_mean_toe_off,'noload_toe_off':noload_mean_toe_off,'kinematics':utils.smooth(mean_mono_loaded_knee_kinematics,5),
                          'kinematics_std':utils.smooth(std_mono_loaded_knee_kinematics,5),'moment':utils.smooth(mean_mono_loaded_knee_torque,5),
                          'moment_std':utils.smooth(std_mono_loaded_knee_torque,5),'color':mycolors['cyan blue'],'toe_off_color':mycolors['cyan blue'],'label':'bi loaded knee actuator'}
utils.plot_stiffness(plot_dic = knee_stiffness_plot_dic, load_condition='loaded',\
                     kinematics_ticks=[0,10,20,30,40,50,60,70,80],moment_ticks=[-2,-1,0,1,2],ax1=ax1,ax2=ax2,ax3=ax3)
# monoarticular knee noload case
knee_stiffness_plot_dic = {'loaded_toe_off':loaded_mean_toe_off,'noload_toe_off':noload_mean_toe_off,'kinematics':utils.smooth(mean_mono_noload_knee_kinematics,5),
                          'kinematics_std':utils.smooth(std_mono_noload_knee_kinematics,5),'moment':utils.smooth(mean_mono_noload_knee_torque,5),
                          'moment_std':utils.smooth(std_mono_noload_knee_torque,5),'color':mycolors['olympic blue'],'toe_off_color':mycolors['olympic blue'],'label':'bi noload knee actuator'}
utils.plot_stiffness(plot_dic = knee_stiffness_plot_dic, load_condition='noload',
kinematics_ticks=[0,10,20,30,40,50,60,70,80],moment_ticks=[-2,-1,0,1,2],ax1=ax4,ax2=ax5,ax3=ax6)
# monoarticular knee muscles loaded case
knee_stiffness_plot_dic = {'loaded_toe_off':loaded_mean_toe_off,'noload_toe_off':noload_mean_toe_off,'kinematics':utils.smooth(mean_mono_loaded_knee_kinematics,5),
                          'kinematics_std':utils.smooth(std_mono_loaded_knee_kinematics,5),'moment':utils.smooth(mean_mono_loaded_kneemuscles_moment,5),
                          'moment_std':utils.smooth(std_mono_loaded_kneemuscles_moment,5),'color':mycolors['crimson red'],'toe_off_color':mycolors['crimson red'],'label':'bi loaded knee muscles'}
utils.plot_stiffness(plot_dic = knee_stiffness_plot_dic, load_condition='loaded',\
                     kinematics_ticks=[0,10,20,30,40,50,60,70,80],moment_ticks=[-2,-1,0,1,2],ax1=ax1,ax2=ax2,ax3=ax3)
# monoarticular knee muscles noload case
knee_stiffness_plot_dic = {'loaded_toe_off':loaded_mean_toe_off,'noload_toe_off':noload_mean_toe_off,'kinematics':utils.smooth(mean_mono_noload_knee_kinematics,5),
                          'kinematics_std':utils.smooth(std_mono_noload_knee_kinematics,5),'moment':utils.smooth(mean_mono_noload_kneemuscles_moment,5),
                          'moment_std':utils.smooth(std_mono_noload_kneemuscles_moment,5),'color':mycolors['french rose'],'toe_off_color':mycolors['french rose'],'label':'bi noload knee muscles'}
utils.plot_stiffness(plot_dic = knee_stiffness_plot_dic, load_condition='noload',
kinematics_ticks=[0,10,20,30,40,50,60,70,80],moment_ticks=[-2,-1,0,1,2],ax1=ax4,ax2=ax5,ax3=ax6)

fig.tight_layout(h_pad=-1, w_pad=-1.5)
fig.subplots_adjust(top=0.98, bottom=0.075, left=0.100, right=0.975,hspace=0.45,wspace=0.35)
plt.show()
fig.savefig('./Figures/Ideal/Monoarticular_LoadedVsNoload_KneeActuator_Stiffness.pdf',orientation='landscape',bbox_inches='tight')

##########################################################################################################################################################
# required dictionary
plot_dic={
'kinematic_1_list' : [rra_dataset['mean_loaded_hipjoint_kinematics'],rra_dataset['mean_loaded_hipjoint_kinematics'],\
                      rra_dataset['mean_noload_hipjoint_kinematics'],rra_dataset['mean_noload_hipjoint_kinematics'],\
                      rra_dataset['mean_loaded_hipjoint_kinematics'],rra_dataset['mean_loaded_hipjoint_kinematics'],\
                      rra_dataset['mean_noload_hipjoint_kinematics'],rra_dataset['mean_noload_hipjoint_kinematics']],
'moment_1_list' : [unnormalized_moment_dataset['mean_loaded_hipmuscles_moment'],unnormalized_moment_dataset['mean_loaded_hipmuscles_moment'],\
                   unnormalized_moment_dataset['mean_noload_hipmuscles_moment'],unnormalized_moment_dataset['mean_noload_hipmuscles_moment'],\
                   unnormalized_moment_dataset['mean_loaded_hipmuscles_moment'],unnormalized_moment_dataset['mean_loaded_hipmuscles_moment'],\
                   unnormalized_moment_dataset['mean_noload_hipmuscles_moment'],unnormalized_moment_dataset['mean_noload_hipmuscles_moment']],
'label_1':'hip joint',
'kinematic_2_list' : [mean_bi_loaded_hip_kinematics,mean_mono_loaded_hip_kinematics,mean_bi_noload_hip_kinematics,mean_mono_noload_hip_kinematics,\
                      mean_bi_loaded_hip_kinematics,mean_mono_loaded_hip_kinematics,mean_bi_noload_hip_kinematics,mean_mono_noload_hip_kinematics],
'moment_2_list' : [mean_bi_loaded_hip_torque,mean_mono_loaded_hip_torque,mean_bi_noload_hip_torque,mean_mono_noload_hip_torque,\
                   mean_bi_loaded_hipmuscles_moment,mean_mono_loaded_hipmuscles_moment,mean_bi_noload_hipmuscles_moment,mean_mono_noload_hipmuscles_moment],
'label_2':['hip actuator','hip actuator','hip actuator','hip actuator','hip muscles','hip muscles','hip muscles','hip muscles'],
# joint stiffness part
'linear_kinematics_1_list':[hip_loaded_linear_angle_dict,hip_loaded_linear_angle_dict,hip_noload_linear_angle_dict,hip_noload_linear_angle_dict,\
                            hip_loaded_linear_angle_dict,hip_loaded_linear_angle_dict,hip_noload_linear_angle_dict,hip_noload_linear_angle_dict],
'linear_moment_1_list':[hip_loaded_linear_moment_dict,hip_loaded_linear_moment_dict,hip_noload_linear_moment_dict,hip_noload_linear_moment_dict,\
                            hip_loaded_linear_moment_dict,hip_loaded_linear_moment_dict,hip_noload_linear_moment_dict,hip_noload_linear_moment_dict],
'fitted_line_1_list':[hip_loaded_fitted_line,hip_loaded_fitted_line,hip_noload_fitted_line,hip_noload_fitted_line,\
                      hip_loaded_fitted_line,hip_loaded_fitted_line,hip_noload_fitted_line,hip_noload_fitted_line],
# device/muscles part
'linear_kinematics_2_list':[bi_loaded_hip_actuator_linear_angle_dict,mono_loaded_hip_actuator_linear_angle_dict,\
                            bi_noload_hip_actuator_linear_angle_dict,mono_noload_hip_actuator_linear_angle_dict,\
                            bi_loaded_hip_musclesmoment_linear_angle_dict,mono_loaded_hip_musclesmoment_linear_angle_dict,\
                            bi_noload_hip_musclesmoment_linear_angle_dict,mono_noload_hip_musclesmoment_linear_angle_dict],
'linear_moment_2_list':[bi_loaded_hip_actuator_linear_moment_dict,mono_loaded_hip_actuator_linear_moment_dict,\
                            bi_noload_hip_actuator_linear_moment_dict,mono_noload_hip_actuator_linear_moment_dict,\
                            bi_loaded_hip_musclesmoment_linear_moment_dict,mono_loaded_hip_musclesmoment_linear_moment_dict,\
                            bi_noload_hip_musclesmoment_linear_moment_dict,mono_noload_hip_musclesmoment_linear_moment_dict],
'fitted_line_2_list':[bi_loaded_hip_actuator_fitted_line,mono_loaded_hip_actuator_fitted_line,bi_noload_hip_actuator_fitted_line,mono_noload_hip_actuator_fitted_line,\
                      bi_loaded_hip_musclesmoment_fitted_line,mono_loaded_hip_musclesmoment_fitted_line,bi_noload_hip_musclesmoment_fitted_line,mono_noload_hip_musclesmoment_fitted_line],                     
'plot_titles' : ['loaded biarticular\nhip actuator','loaded monoarticular\nhip actuator','noload biarticular\nhip actuator','noload monoarticular\nhip actuator',\
                'loaded biarticular\nhip muscles','loaded monoarticular\nhip muscles','noload biarticular\nhip muscles','noload monoarticular\nhip muscles']
}
default_color_dic = {
'color_1_list' : ['k','k','xkcd:irish green','xkcd:irish green','k','k','xkcd:irish green','xkcd:irish green'],
'color_2_list' : [mycolors['cyan blue'],mycolors['cyan blue'],mycolors['olympic blue'],mycolors['olympic blue'],\
                  mycolors['crimson red'],mycolors['crimson red'],mycolors['french rose'],mycolors['french rose']],
'color_3_list' : [mycolors['cyan blue'],mycolors['cyan blue'],mycolors['olympic blue'],mycolors['olympic blue'],\
                  mycolors['crimson red'],mycolors['crimson red'],mycolors['french rose'],mycolors['french rose']]
}


fig = plt.figure(num='hip joint stiffness',figsize=(14.8, 9.6))
utils.plot_joint_exo_muscles_stiffness(nrows=2,ncols=4,plot_dic=plot_dic,color_dic=default_color_dic,
                                ylabel='moment (N-m/kg)',thirdplot=False,joint_phases=True,joint='hip',
                                moment_ticks = [-2,-1,0,1,2],kinematics_ticks =[-0.5,0,0.5,1],
                                xlabel_loc=[4,5,6,7],ylabel_loc=[0,4])
fig.tight_layout(h_pad=-1.5, w_pad=-1.5)
fig.subplots_adjust(top=0.99, bottom=0.075, left=0.100, right=0.975,hspace=0.2,wspace=0.15)
fig.savefig('./Figures/Ideal/PaperFigure_Hip_Device_Muscles_Stiffness.pdf',orientation='landscape',bbox_inches='tight')
plt.show()

#*****************************************************************************************************************************
# required dictionary
plot_dic={
'kinematic_1_list' : [rra_dataset['mean_loaded_kneejoint_kinematics'],rra_dataset['mean_loaded_kneejoint_kinematics'],\
                      rra_dataset['mean_noload_kneejoint_kinematics'],rra_dataset['mean_noload_kneejoint_kinematics'],\
                      rra_dataset['mean_loaded_kneejoint_kinematics'],rra_dataset['mean_loaded_kneejoint_kinematics'],\
                      rra_dataset['mean_noload_kneejoint_kinematics'],rra_dataset['mean_noload_kneejoint_kinematics']],
'moment_1_list' : [unnormalized_moment_dataset['mean_loaded_kneemuscles_moment'],unnormalized_moment_dataset['mean_loaded_kneemuscles_moment'],\
                   unnormalized_moment_dataset['mean_noload_kneemuscles_moment'],unnormalized_moment_dataset['mean_noload_kneemuscles_moment'],\
                   unnormalized_moment_dataset['mean_loaded_kneemuscles_moment'],unnormalized_moment_dataset['mean_loaded_kneemuscles_moment'],\
                   unnormalized_moment_dataset['mean_noload_kneemuscles_moment'],unnormalized_moment_dataset['mean_noload_kneemuscles_moment']],
'label_1':'knee joint',
'kinematic_2_list' : [mean_bi_loaded_knee_kinematics,mean_mono_loaded_knee_kinematics,mean_bi_noload_knee_kinematics,mean_mono_noload_knee_kinematics,\
                      mean_mono_loaded_knee_kinematics,mean_mono_loaded_knee_kinematics,mean_mono_noload_knee_kinematics,mean_mono_noload_knee_kinematics],
'moment_2_list' : [mean_bi_loaded_knee_torque,mean_mono_loaded_knee_torque,mean_bi_noload_knee_torque,mean_mono_noload_knee_torque,\
                   mean_bi_loaded_kneemuscles_moment,mean_mono_loaded_kneemuscles_moment,mean_bi_noload_kneemuscles_moment,mean_mono_noload_kneemuscles_moment],
'label_2':['knee actuator','knee actuator','knee actuator','knee actuator','knee muscles','knee muscles','knee muscles','knee muscles'],
# joint stiffness part
'linear_kinematics_1_list':[knee_loaded_linear_angle_dict,knee_loaded_linear_angle_dict,knee_noload_linear_angle_dict,knee_noload_linear_angle_dict,\
                            knee_loaded_linear_angle_dict,knee_loaded_linear_angle_dict,knee_noload_linear_angle_dict,knee_noload_linear_angle_dict],
'linear_moment_1_list':[knee_loaded_linear_moment_dict,knee_loaded_linear_moment_dict,knee_noload_linear_moment_dict,knee_noload_linear_moment_dict,\
                            knee_loaded_linear_moment_dict,knee_loaded_linear_moment_dict,knee_noload_linear_moment_dict,knee_noload_linear_moment_dict],
'fitted_line_1_list':[knee_loaded_fitted_line,knee_loaded_fitted_line,knee_noload_fitted_line,knee_noload_fitted_line,\
                      knee_loaded_fitted_line,knee_loaded_fitted_line,knee_noload_fitted_line,knee_noload_fitted_line],
# device/muscles part
'linear_kinematics_2_list':[bi_loaded_knee_actuator_linear_angle_dict,mono_loaded_knee_actuator_linear_angle_dict,\
                            bi_noload_knee_actuator_linear_angle_dict,mono_noload_knee_actuator_linear_angle_dict,\
                            bi_loaded_knee_musclesmoment_linear_angle_dict,mono_loaded_knee_musclesmoment_linear_angle_dict,\
                            bi_noload_knee_musclesmoment_linear_angle_dict,mono_noload_knee_musclesmoment_linear_angle_dict],
'linear_moment_2_list':[bi_loaded_knee_actuator_linear_moment_dict,mono_loaded_knee_actuator_linear_moment_dict,\
                            bi_noload_knee_actuator_linear_moment_dict,mono_noload_knee_actuator_linear_moment_dict,\
                            bi_loaded_knee_musclesmoment_linear_moment_dict,mono_loaded_knee_musclesmoment_linear_moment_dict,\
                            bi_noload_knee_musclesmoment_linear_moment_dict,mono_noload_knee_musclesmoment_linear_moment_dict],
'fitted_line_2_list':[bi_loaded_knee_actuator_fitted_line,mono_loaded_knee_actuator_fitted_line,bi_noload_knee_actuator_fitted_line,mono_noload_knee_actuator_fitted_line,\
                      bi_loaded_knee_musclesmoment_fitted_line,mono_loaded_knee_musclesmoment_fitted_line,bi_noload_knee_musclesmoment_fitted_line,mono_noload_knee_musclesmoment_fitted_line],                     
'plot_titles' : ['loaded biarticular\nknee actuator','loaded monoarticular\nknee actuator','noload biarticular\nknee actuator','noload monoarticular\nknee actuator',\
                'loaded biarticular\nknee muscles','loaded monoarticular\nknee muscles','noload biarticular\nknee muscles','noload monoarticular\nknee muscles']
}
default_color_dic = {
'color_1_list' : ['k','k','xkcd:irish green','xkcd:irish green','k','k','xkcd:irish green','xkcd:irish green'],
'color_2_list' : [mycolors['cyan blue'],mycolors['cyan blue'],mycolors['olympic blue'],mycolors['olympic blue'],\
                  mycolors['crimson red'],mycolors['crimson red'],mycolors['french rose'],mycolors['french rose']],
'color_3_list' : [mycolors['cyan blue'],mycolors['cyan blue'],mycolors['olympic blue'],mycolors['olympic blue'],\
                  mycolors['crimson red'],mycolors['crimson red'],mycolors['french rose'],mycolors['french rose']]
}
fig = plt.figure(num='knee joint stiffness',figsize=(14.8, 9.6))
utils.plot_joint_exo_muscles_stiffness(nrows=2,ncols=4,plot_dic=plot_dic,color_dic=default_color_dic,
                                ylabel='moment (N-m/kg)',thirdplot=False,joint_phases=True,joint='knee',
                                moment_ticks = [-2,-1,0,1,2],kinematics_ticks =[-1,-0.5,0,0.5,1,1.5,2],
                                xlabel_loc=[4,5,6,7],ylabel_loc=[0,4])
fig.tight_layout(h_pad=-1.5, w_pad=-1.5)
fig.subplots_adjust(top=0.99, bottom=0.075, left=0.100, right=0.975,hspace=0.2,wspace=0.15)
fig.savefig('./Figures/Ideal/PaperFigure_Knee_Device_Muscles_Stiffness.pdf',orientation='landscape',bbox_inches='tight')
plt.show()

##########################################################################################################################################################
actuators_mean_cellText = np.transpose(np.array((# biarticular loaded
                         [bi_loaded_hip_actuator_stiffness_dict['mean_hip_extension_stiffness'],bi_loaded_hip_actuator_stiffness_dict['mean_hip_flexion_stiffness'],bi_loaded_hip_actuator_stiffness_dict['mean_hip_total_stiffness']],\
                         [bi_loaded_hip_actuator_stiffness_dict['std_hip_extension_stiffness'],bi_loaded_hip_actuator_stiffness_dict['std_hip_flexion_stiffness'],bi_loaded_hip_actuator_stiffness_dict['std_hip_total_stiffness']],\
                         [bi_loaded_knee_actuator_stiffness_dict['mean_knee_extension_stiffness'],bi_loaded_knee_actuator_stiffness_dict['mean_knee_flexion_stiffness'],bi_loaded_knee_actuator_stiffness_dict['mean_knee_total_stiffness']],\
                         [bi_loaded_knee_actuator_stiffness_dict['std_knee_extension_stiffness'],bi_loaded_knee_actuator_stiffness_dict['std_knee_flexion_stiffness'],bi_loaded_knee_actuator_stiffness_dict['std_knee_total_stiffness']],\
                         # monoarticular loaded
                         [mono_loaded_hip_actuator_stiffness_dict['mean_hip_extension_stiffness'],mono_loaded_hip_actuator_stiffness_dict['mean_hip_flexion_stiffness'],mono_loaded_hip_actuator_stiffness_dict['mean_hip_total_stiffness']],\
                         [mono_loaded_hip_actuator_stiffness_dict['std_hip_extension_stiffness'],mono_loaded_hip_actuator_stiffness_dict['std_hip_flexion_stiffness'],mono_loaded_hip_actuator_stiffness_dict['std_hip_total_stiffness']],\
                         [mono_loaded_knee_actuator_stiffness_dict['mean_knee_extension_stiffness'],mono_loaded_knee_actuator_stiffness_dict['mean_knee_flexion_stiffness'],mono_loaded_knee_actuator_stiffness_dict['mean_knee_total_stiffness']],\
                         [mono_loaded_knee_actuator_stiffness_dict['std_knee_extension_stiffness'],mono_loaded_knee_actuator_stiffness_dict['std_knee_flexion_stiffness'],mono_loaded_knee_actuator_stiffness_dict['std_knee_total_stiffness']],\
                         # biarticular noload
                         [bi_noload_hip_actuator_stiffness_dict['mean_hip_extension_stiffness'],bi_noload_hip_actuator_stiffness_dict['mean_hip_flexion_stiffness'],bi_noload_hip_actuator_stiffness_dict['mean_hip_total_stiffness']],\
                         [bi_noload_hip_actuator_stiffness_dict['std_hip_extension_stiffness'],bi_noload_hip_actuator_stiffness_dict['std_hip_flexion_stiffness'],bi_noload_hip_actuator_stiffness_dict['std_hip_total_stiffness']],\
                         [bi_noload_knee_actuator_stiffness_dict['mean_knee_extension_stiffness'],bi_noload_knee_actuator_stiffness_dict['mean_knee_flexion_stiffness'],bi_noload_knee_actuator_stiffness_dict['mean_knee_total_stiffness']],\
                         [bi_noload_knee_actuator_stiffness_dict['std_knee_extension_stiffness'],bi_noload_knee_actuator_stiffness_dict['std_knee_flexion_stiffness'],bi_noload_knee_actuator_stiffness_dict['std_knee_total_stiffness']],\
                         # monoarticular noload
                         [mono_noload_hip_actuator_stiffness_dict['mean_hip_extension_stiffness'],mono_noload_hip_actuator_stiffness_dict['mean_hip_flexion_stiffness'],mono_noload_hip_actuator_stiffness_dict['mean_hip_total_stiffness']],\
                         [mono_noload_hip_actuator_stiffness_dict['std_hip_extension_stiffness'],mono_noload_hip_actuator_stiffness_dict['std_hip_flexion_stiffness'],mono_noload_hip_actuator_stiffness_dict['std_hip_total_stiffness']],\
                         [mono_noload_knee_actuator_stiffness_dict['mean_knee_extension_stiffness'],mono_noload_knee_actuator_stiffness_dict['mean_knee_flexion_stiffness'],mono_noload_knee_actuator_stiffness_dict['mean_knee_total_stiffness']],\
                         [mono_noload_knee_actuator_stiffness_dict['std_knee_extension_stiffness'],mono_noload_knee_actuator_stiffness_dict['std_knee_flexion_stiffness'],mono_noload_knee_actuator_stiffness_dict['std_knee_total_stiffness']])))
actuators_Rsquare_cellText = np.transpose(np.array((# biarticular loaded
                         [bi_loaded_hip_actuator_Rsquare_dict['mean_hip_extension_R_square'],bi_loaded_hip_actuator_Rsquare_dict['mean_hip_flexion_R_square'],bi_loaded_hip_actuator_Rsquare_dict['mean_hip_total_R_square']],\
                         [bi_loaded_hip_actuator_Rsquare_dict['std_hip_extension_R_square'],bi_loaded_hip_actuator_Rsquare_dict['std_hip_flexion_R_square'],bi_loaded_hip_actuator_Rsquare_dict['std_hip_total_R_square']],\
                         [bi_loaded_knee_actuator_Rsquare_dict['mean_knee_extension_R_square'],bi_loaded_knee_actuator_Rsquare_dict['mean_knee_flexion_R_square'],bi_loaded_knee_actuator_Rsquare_dict['mean_knee_total_R_square']],\
                         [bi_loaded_knee_actuator_Rsquare_dict['std_knee_extension_R_square'],bi_loaded_knee_actuator_Rsquare_dict['std_knee_flexion_R_square'],bi_loaded_knee_actuator_Rsquare_dict['std_knee_total_R_square']],\
                         # monoarticular loaded
                         [mono_loaded_hip_actuator_Rsquare_dict['mean_hip_extension_R_square'],mono_loaded_hip_actuator_Rsquare_dict['mean_hip_flexion_R_square'],mono_loaded_hip_actuator_Rsquare_dict['mean_hip_total_R_square']],\
                         [mono_loaded_hip_actuator_Rsquare_dict['std_hip_extension_R_square'],mono_loaded_hip_actuator_Rsquare_dict['std_hip_flexion_R_square'],mono_loaded_hip_actuator_Rsquare_dict['std_hip_total_R_square']],\
                         [mono_loaded_knee_actuator_Rsquare_dict['mean_knee_extension_R_square'],mono_loaded_knee_actuator_Rsquare_dict['mean_knee_flexion_R_square'],mono_loaded_knee_actuator_Rsquare_dict['mean_knee_total_R_square']],\
                         [mono_loaded_knee_actuator_Rsquare_dict['std_knee_extension_R_square'],mono_loaded_knee_actuator_Rsquare_dict['std_knee_flexion_R_square'],mono_loaded_knee_actuator_Rsquare_dict['std_knee_total_R_square']],\
                         # biarticular noload
                         [bi_noload_hip_actuator_Rsquare_dict['mean_hip_extension_R_square'],bi_noload_hip_actuator_Rsquare_dict['mean_hip_flexion_R_square'],bi_noload_hip_actuator_Rsquare_dict['mean_hip_total_R_square']],\
                         [bi_noload_hip_actuator_Rsquare_dict['std_hip_extension_R_square'],bi_noload_hip_actuator_Rsquare_dict['std_hip_flexion_R_square'],bi_noload_hip_actuator_Rsquare_dict['std_hip_total_R_square']],\
                         [bi_noload_knee_actuator_Rsquare_dict['mean_knee_extension_R_square'],bi_noload_knee_actuator_Rsquare_dict['mean_knee_flexion_R_square'],bi_noload_knee_actuator_Rsquare_dict['mean_knee_total_R_square']],\
                         [bi_noload_knee_actuator_Rsquare_dict['std_knee_extension_R_square'],bi_noload_knee_actuator_Rsquare_dict['std_knee_flexion_R_square'],bi_noload_knee_actuator_Rsquare_dict['std_knee_total_R_square']],\
                         # monoarticular noload
                         [mono_noload_hip_actuator_Rsquare_dict['mean_hip_extension_R_square'],mono_noload_hip_actuator_Rsquare_dict['mean_hip_flexion_R_square'],mono_noload_hip_actuator_Rsquare_dict['mean_hip_total_R_square']],\
                         [mono_noload_hip_actuator_Rsquare_dict['std_hip_extension_R_square'],mono_noload_hip_actuator_Rsquare_dict['std_hip_flexion_R_square'],mono_noload_hip_actuator_Rsquare_dict['std_hip_total_R_square']],\
                         [mono_noload_knee_actuator_Rsquare_dict['mean_knee_extension_R_square'],mono_noload_knee_actuator_Rsquare_dict['mean_knee_flexion_R_square'],mono_noload_knee_actuator_Rsquare_dict['mean_knee_total_R_square']],\
                         [mono_noload_knee_actuator_Rsquare_dict['std_knee_extension_R_square'],mono_noload_knee_actuator_Rsquare_dict['std_knee_flexion_R_square'],mono_noload_knee_actuator_Rsquare_dict['std_knee_total_R_square']])))
# muscles dataset
muscles_mean_cellText = np.transpose(np.array((# biarticular loaded
                         [bi_loaded_hip_musclesmoment_stiffness_dict['mean_hip_extension_stiffness'],bi_loaded_hip_musclesmoment_stiffness_dict['mean_hip_flexion_stiffness'],bi_loaded_hip_musclesmoment_stiffness_dict['mean_hip_total_stiffness']],\
                         [bi_loaded_hip_musclesmoment_stiffness_dict['std_hip_extension_stiffness'],bi_loaded_hip_musclesmoment_stiffness_dict['std_hip_flexion_stiffness'],bi_loaded_hip_musclesmoment_stiffness_dict['std_hip_total_stiffness']],\
                         [bi_loaded_knee_musclesmoment_stiffness_dict['mean_knee_extension_stiffness'],bi_loaded_knee_musclesmoment_stiffness_dict['mean_knee_flexion_stiffness'],bi_loaded_knee_musclesmoment_stiffness_dict['mean_knee_total_stiffness']],\
                         [bi_loaded_knee_musclesmoment_stiffness_dict['std_knee_extension_stiffness'],bi_loaded_knee_musclesmoment_stiffness_dict['std_knee_flexion_stiffness'],bi_loaded_knee_musclesmoment_stiffness_dict['std_knee_total_stiffness']],\
                         # monoarticular loaded
                         [mono_loaded_hip_musclesmoment_stiffness_dict['mean_hip_extension_stiffness'],mono_loaded_hip_musclesmoment_stiffness_dict['mean_hip_flexion_stiffness'],mono_loaded_hip_musclesmoment_stiffness_dict['mean_hip_total_stiffness']],\
                         [mono_loaded_hip_musclesmoment_stiffness_dict['std_hip_extension_stiffness'],mono_loaded_hip_musclesmoment_stiffness_dict['std_hip_flexion_stiffness'],mono_loaded_hip_musclesmoment_stiffness_dict['std_hip_total_stiffness']],\
                         [mono_loaded_knee_musclesmoment_stiffness_dict['mean_knee_extension_stiffness'],mono_loaded_knee_musclesmoment_stiffness_dict['mean_knee_flexion_stiffness'],mono_loaded_knee_musclesmoment_stiffness_dict['mean_knee_total_stiffness']],\
                         [mono_loaded_knee_musclesmoment_stiffness_dict['std_knee_extension_stiffness'],mono_loaded_knee_musclesmoment_stiffness_dict['std_knee_flexion_stiffness'],mono_loaded_knee_musclesmoment_stiffness_dict['std_knee_total_stiffness']],\
                         # biarticular noload
                         [bi_noload_hip_musclesmoment_stiffness_dict['mean_hip_extension_stiffness'],bi_noload_hip_musclesmoment_stiffness_dict['mean_hip_flexion_stiffness'],bi_noload_hip_musclesmoment_stiffness_dict['mean_hip_total_stiffness']],\
                         [bi_noload_hip_musclesmoment_stiffness_dict['std_hip_extension_stiffness'],bi_noload_hip_musclesmoment_stiffness_dict['std_hip_flexion_stiffness'],bi_noload_hip_musclesmoment_stiffness_dict['std_hip_total_stiffness']],\
                         [bi_noload_knee_musclesmoment_stiffness_dict['mean_knee_extension_stiffness'],bi_noload_knee_musclesmoment_stiffness_dict['mean_knee_flexion_stiffness'],bi_noload_knee_musclesmoment_stiffness_dict['mean_knee_total_stiffness']],\
                         [bi_noload_knee_musclesmoment_stiffness_dict['std_knee_extension_stiffness'],bi_noload_knee_musclesmoment_stiffness_dict['std_knee_flexion_stiffness'],bi_noload_knee_musclesmoment_stiffness_dict['std_knee_total_stiffness']],\
                         # monoarticular noload
                         [mono_noload_hip_musclesmoment_stiffness_dict['mean_hip_extension_stiffness'],mono_noload_hip_musclesmoment_stiffness_dict['mean_hip_flexion_stiffness'],mono_noload_hip_musclesmoment_stiffness_dict['mean_hip_total_stiffness']],\
                         [mono_noload_hip_musclesmoment_stiffness_dict['std_hip_extension_stiffness'],mono_noload_hip_musclesmoment_stiffness_dict['std_hip_flexion_stiffness'],mono_noload_hip_musclesmoment_stiffness_dict['std_hip_total_stiffness']],\
                         [mono_noload_knee_musclesmoment_stiffness_dict['mean_knee_extension_stiffness'],mono_noload_knee_musclesmoment_stiffness_dict['mean_knee_flexion_stiffness'],mono_noload_knee_musclesmoment_stiffness_dict['mean_knee_total_stiffness']],\
                         [mono_noload_knee_musclesmoment_stiffness_dict['std_knee_extension_stiffness'],mono_noload_knee_musclesmoment_stiffness_dict['std_knee_flexion_stiffness'],mono_noload_knee_musclesmoment_stiffness_dict['std_knee_total_stiffness']])))
muscles_Rsquare_cellText = np.transpose(np.array((# biarticular loaded
                         [bi_loaded_hip_musclesmoment_Rsquare_dict['mean_hip_extension_R_square'],bi_loaded_hip_musclesmoment_Rsquare_dict['mean_hip_flexion_R_square'],bi_loaded_hip_musclesmoment_Rsquare_dict['mean_hip_total_R_square']],\
                         [bi_loaded_hip_musclesmoment_Rsquare_dict['std_hip_extension_R_square'],bi_loaded_hip_musclesmoment_Rsquare_dict['std_hip_flexion_R_square'],bi_loaded_hip_musclesmoment_Rsquare_dict['std_hip_total_R_square']],\
                         [bi_loaded_knee_musclesmoment_Rsquare_dict['mean_knee_extension_R_square'],bi_loaded_knee_musclesmoment_Rsquare_dict['mean_knee_flexion_R_square'],bi_loaded_knee_musclesmoment_Rsquare_dict['mean_knee_total_R_square']],\
                         [bi_loaded_knee_musclesmoment_Rsquare_dict['std_knee_extension_R_square'],bi_loaded_knee_musclesmoment_Rsquare_dict['std_knee_flexion_R_square'],bi_loaded_knee_musclesmoment_Rsquare_dict['std_knee_total_R_square']],\
                         # monoarticular loaded
                         [mono_loaded_hip_musclesmoment_Rsquare_dict['mean_hip_extension_R_square'],mono_loaded_hip_musclesmoment_Rsquare_dict['mean_hip_flexion_R_square'],mono_loaded_hip_musclesmoment_Rsquare_dict['mean_hip_total_R_square']],\
                         [mono_loaded_hip_musclesmoment_Rsquare_dict['std_hip_extension_R_square'],mono_loaded_hip_musclesmoment_Rsquare_dict['std_hip_flexion_R_square'],mono_loaded_hip_musclesmoment_Rsquare_dict['std_hip_total_R_square']],\
                         [mono_loaded_knee_musclesmoment_Rsquare_dict['mean_knee_extension_R_square'],mono_loaded_knee_musclesmoment_Rsquare_dict['mean_knee_flexion_R_square'],mono_loaded_knee_musclesmoment_Rsquare_dict['mean_knee_total_R_square']],\
                         [mono_loaded_knee_musclesmoment_Rsquare_dict['std_knee_extension_R_square'],mono_loaded_knee_musclesmoment_Rsquare_dict['std_knee_flexion_R_square'],mono_loaded_knee_musclesmoment_Rsquare_dict['std_knee_total_R_square']],\
                         # biarticular noload
                         [bi_noload_hip_musclesmoment_Rsquare_dict['mean_hip_extension_R_square'],bi_noload_hip_musclesmoment_Rsquare_dict['mean_hip_flexion_R_square'],bi_noload_hip_musclesmoment_Rsquare_dict['mean_hip_total_R_square']],\
                         [bi_noload_hip_musclesmoment_Rsquare_dict['std_hip_extension_R_square'],bi_noload_hip_musclesmoment_Rsquare_dict['std_hip_flexion_R_square'],bi_noload_hip_musclesmoment_Rsquare_dict['std_hip_total_R_square']],\
                         [bi_noload_knee_musclesmoment_Rsquare_dict['mean_knee_extension_R_square'],bi_noload_knee_musclesmoment_Rsquare_dict['mean_knee_flexion_R_square'],bi_noload_knee_musclesmoment_Rsquare_dict['mean_knee_total_R_square']],\
                         [bi_noload_knee_musclesmoment_Rsquare_dict['std_knee_extension_R_square'],bi_noload_knee_musclesmoment_Rsquare_dict['std_knee_flexion_R_square'],bi_noload_knee_musclesmoment_Rsquare_dict['std_knee_total_R_square']],\
                         # monoarticular noload
                         [mono_noload_hip_musclesmoment_Rsquare_dict['mean_hip_extension_R_square'],mono_noload_hip_musclesmoment_Rsquare_dict['mean_hip_flexion_R_square'],mono_noload_hip_musclesmoment_Rsquare_dict['mean_hip_total_R_square']],\
                         [mono_noload_hip_musclesmoment_Rsquare_dict['std_hip_extension_R_square'],mono_noload_hip_musclesmoment_Rsquare_dict['std_hip_flexion_R_square'],mono_noload_hip_musclesmoment_Rsquare_dict['std_hip_total_R_square']],\
                         [mono_noload_knee_musclesmoment_Rsquare_dict['mean_knee_extension_R_square'],mono_noload_knee_musclesmoment_Rsquare_dict['mean_knee_flexion_R_square'],mono_noload_knee_musclesmoment_Rsquare_dict['mean_knee_total_R_square']],\
                         [mono_noload_knee_musclesmoment_Rsquare_dict['std_knee_extension_R_square'],mono_noload_knee_musclesmoment_Rsquare_dict['std_knee_flexion_R_square'],mono_noload_knee_musclesmoment_Rsquare_dict['std_knee_total_R_square']])))
actuator_main_cellText = np.concatenate((actuators_mean_cellText,actuators_Rsquare_cellText),axis=0)
muscles_main_cellText = np.concatenate((muscles_mean_cellText,muscles_Rsquare_cellText),axis=0)

rows = ['mean bi loaded hip','std bi loaded hip','mean bi loaded knee','std bi loaded knee',\
        'mean mono loaded hip','std mono loaded hip','mean mono loaded knee','std mono loaded knee',\
        'mean bi noload hip','std bi noload hip','mean bi noload knee','std bi noload knee',\
        'mean mono noload hip','std mono noload hip','mean mono noload knee','std mono noload knee']
columns = ['extension','flexion','total','extension R^2','flexion R^2','total R^2']
fig, ax = plt.subplots(figsize=(15,9))
table = ax.table(cellText=np.transpose(actuator_main_cellText.round(3)),rowLabels=rows,colLabels=columns,loc='center')
table.scale(1,2)
table.set_fontsize(18)
ax.axis('off')
fig.subplots_adjust(top=0.99, bottom=0.075, left=0.23, right=0.975,hspace=0.2,wspace=0.15)
fig.savefig('./Figures/Ideal/Device_Stiffness_Table.pdf',orientation='landscape',bbox_inches='tight')
plt.show()
fig, ax = plt.subplots(figsize=(15,9))
table = ax.table(cellText=np.transpose(muscles_main_cellText.round(3)),rowLabels=rows,colLabels=columns,loc='center')
table.scale(1,2)
table.set_fontsize(18)
ax.axis('off')
fig.subplots_adjust(top=0.99, bottom=0.075, left=0.23, right=0.975,hspace=0.2,wspace=0.15)
fig.savefig('./Figures/Ideal/Muscles_Stiffness_Table.pdf',orientation='landscape',bbox_inches='tight')
plt.show()
