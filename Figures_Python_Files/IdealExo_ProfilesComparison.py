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
# ideal exo torque dataset
directory = 'D:/Ali.K.M.Bonab/Walking_Mass_Inertia_Effect/Data/Data/Ideal/*_torque.csv'
files = enumerate(glob.iglob(directory), 1)
exo_torque_dataset = {pathlib.PurePath(f[1]).stem: np.loadtxt(f[1], delimiter=',') for f in files}
# ideal exo power dataset
directory = 'D:/Ali.K.M.Bonab/Walking_Mass_Inertia_Effect/Data/Data/Ideal/*_power.csv'
files = enumerate(glob.iglob(directory), 1)
exo_power_dataset = {pathlib.PurePath(f[1]).stem: np.loadtxt(f[1], delimiter=',') for f in files}
# ideal exo speed dataset
directory = 'D:/Ali.K.M.Bonab/Walking_Mass_Inertia_Effect/Data/Data/Ideal/*_speed.csv'
files = enumerate(glob.iglob(directory), 1)
exo_speed_dataset = {pathlib.PurePath(f[1]).stem: np.loadtxt(f[1], delimiter=',') for f in files}
# muscles moment dataset
directory = 'D:/Ali.K.M.Bonab/Walking_Mass_Inertia_Effect/Data/Data/Ideal/*_musclesmoment.csv'
files = enumerate(glob.iglob(directory), 1)
musclesmoment_dataset = {pathlib.PurePath(f[1]).stem: np.loadtxt(f[1], delimiter=',') for f in files}
# gls
gl_noload = {'noload_subject{}_trial{}'.format(i,j): utils.construct_gl_mass_side(subjectno=i,trialno=j,loadcond='noload') for i in subjects for j in trials_num}
gl_loaded = {'loaded_subject{}_trial{}'.format(i,j): utils.construct_gl_mass_side(subjectno=i,trialno=j,loadcond='loaded') for i in subjects for j in trials_num}
#####################################################################################
# Processing Data
# toe-off
noload_mean_toe_off,_,loaded_mean_toe_off,_ = utils.toe_off_avg_std(gl_noload,gl_loaded)
# exoskeleton torque profiles
# biarticular
# hip
bi_loaded_hip_torque = utils.normalize_direction_data(exo_torque_dataset['biarticular_ideal_loaded_hipactuator_torque'],gl_noload,direction=False)
bi_noload_hip_torque = utils.normalize_direction_data(exo_torque_dataset['biarticular_ideal_noload_hipactuator_torque'],gl_noload,direction=False)
mean_bi_loaded_hip_torque,std_bi_loaded_hip_torque = utils.mean_std_over_subjects(bi_loaded_hip_torque)
mean_bi_noload_hip_torque,std_bi_noload_hip_torque = utils.mean_std_over_subjects(bi_noload_hip_torque)
# knee
bi_loaded_knee_torque = utils.normalize_direction_data(exo_torque_dataset['biarticular_ideal_loaded_kneeactuator_torque'],gl_noload,direction=True)
bi_noload_knee_torque = utils.normalize_direction_data(exo_torque_dataset['biarticular_ideal_noload_kneeactuator_torque'],gl_noload,direction=True)
mean_bi_loaded_knee_torque,std_bi_loaded_knee_torque = utils.mean_std_over_subjects(bi_loaded_knee_torque)
mean_bi_noload_knee_torque,std_bi_noload_knee_torque = utils.mean_std_over_subjects(bi_noload_knee_torque)
# monoarticular
# hip
mono_loaded_hip_torque = utils.normalize_direction_data(exo_torque_dataset['monoarticular_ideal_loaded_hipactuator_torque'],gl_noload,direction=False)
mono_noload_hip_torque = utils.normalize_direction_data(exo_torque_dataset['monoarticular_ideal_noload_hipactuator_torque'],gl_noload,direction=False)
mean_mono_loaded_hip_torque,std_mono_loaded_hip_torque = utils.mean_std_over_subjects(mono_loaded_hip_torque)
mean_mono_noload_hip_torque,std_mono_noload_hip_torque = utils.mean_std_over_subjects(mono_noload_hip_torque)
# knee
mono_loaded_knee_torque = utils.normalize_direction_data(exo_torque_dataset['monoarticular_ideal_loaded_kneeactuator_torque'],gl_noload,direction=True)
mono_noload_knee_torque = utils.normalize_direction_data(exo_torque_dataset['monoarticular_ideal_noload_kneeactuator_torque'],gl_noload,direction=True)
mean_mono_loaded_knee_torque,std_mono_loaded_knee_torque = utils.mean_std_over_subjects(mono_loaded_knee_torque)
mean_mono_noload_knee_torque,std_mono_noload_knee_torque = utils.mean_std_over_subjects(mono_noload_knee_torque)
#******************************
# exoskeleton power profiles
# biarticular
# hip
bi_loaded_hip_power = utils.normalize_direction_data(exo_power_dataset['biarticular_ideal_loaded_hipactuator_power'],gl_noload,direction=False)
bi_noload_hip_power = utils.normalize_direction_data(exo_power_dataset['biarticular_ideal_noload_hipactuator_power'],gl_noload,direction=False)
mean_bi_loaded_hip_power,std_bi_loaded_hip_power = utils.mean_std_over_subjects(bi_loaded_hip_power)
mean_bi_noload_hip_power,std_bi_noload_hip_power = utils.mean_std_over_subjects(bi_noload_hip_power)
# knee
bi_loaded_knee_power = utils.normalize_direction_data(exo_power_dataset['biarticular_ideal_loaded_kneeactuator_power'],gl_noload,direction=False)
bi_noload_knee_power = utils.normalize_direction_data(exo_power_dataset['biarticular_ideal_noload_kneeactuator_power'],gl_noload,direction=False)
mean_bi_loaded_knee_power,std_bi_loaded_knee_power = utils.mean_std_over_subjects(bi_loaded_knee_power)
mean_bi_noload_knee_power,std_bi_noload_knee_power = utils.mean_std_over_subjects(bi_noload_knee_power)
# monoarticular
# hip
mono_loaded_hip_power = utils.normalize_direction_data(exo_power_dataset['monoarticular_ideal_loaded_hipactuator_power'],gl_noload,direction=False)
mono_noload_hip_power = utils.normalize_direction_data(exo_power_dataset['monoarticular_ideal_noload_hipactuator_power'],gl_noload,direction=False)
mean_mono_loaded_hip_power,std_mono_loaded_hip_power = utils.mean_std_over_subjects(mono_loaded_hip_power)
mean_mono_noload_hip_power,std_mono_noload_hip_power = utils.mean_std_over_subjects(mono_noload_hip_power)
# knee
mono_loaded_knee_power = utils.normalize_direction_data(exo_power_dataset['monoarticular_ideal_loaded_kneeactuator_power'],gl_noload,direction=False)
mono_noload_knee_power = utils.normalize_direction_data(exo_power_dataset['monoarticular_ideal_noload_kneeactuator_power'],gl_noload,direction=False)
mean_mono_loaded_knee_power,std_mono_loaded_knee_power = utils.mean_std_over_subjects(mono_loaded_knee_power)
mean_mono_noload_knee_power,std_mono_noload_knee_power = utils.mean_std_over_subjects(mono_noload_knee_power)
#******************************
# exoskeleton speed profiles
# biarticular
# hip
bi_loaded_hip_speed = utils.normalize_direction_data(exo_speed_dataset['biarticular_ideal_loaded_hipactuator_speed'],gl_noload,direction=False,normalize=False)
bi_noload_hip_speed = utils.normalize_direction_data(exo_speed_dataset['biarticular_ideal_noload_hipactuator_speed'],gl_noload,direction=False,normalize=False)
mean_bi_loaded_hip_speed,std_bi_loaded_hip_speed = utils.mean_std_over_subjects(bi_loaded_hip_speed)
mean_bi_noload_hip_speed,std_bi_noload_hip_speed = utils.mean_std_over_subjects(bi_noload_hip_speed)
# knee
bi_loaded_knee_speed = utils.normalize_direction_data(exo_speed_dataset['biarticular_ideal_loaded_kneeactuator_speed'],gl_noload,direction=True,normalize=False)
bi_noload_knee_speed = utils.normalize_direction_data(exo_speed_dataset['biarticular_ideal_noload_kneeactuator_speed'],gl_noload,direction=True,normalize=False)
mean_bi_loaded_knee_speed,std_bi_loaded_knee_speed = utils.mean_std_over_subjects(bi_loaded_knee_speed)
mean_bi_noload_knee_speed,std_bi_noload_knee_speed = utils.mean_std_over_subjects(bi_noload_knee_speed)
# monoarticular
# hip
mono_loaded_hip_speed = utils.normalize_direction_data(exo_speed_dataset['monoarticular_ideal_loaded_hipactuator_speed'],gl_noload,direction=False,normalize=False)
mono_noload_hip_speed = utils.normalize_direction_data(exo_speed_dataset['monoarticular_ideal_noload_hipactuator_speed'],gl_noload,direction=False,normalize=False)
mean_mono_loaded_hip_speed,std_mono_loaded_hip_speed = utils.mean_std_over_subjects(mono_loaded_hip_speed)
mean_mono_noload_hip_speed,std_mono_noload_hip_speed = utils.mean_std_over_subjects(mono_noload_hip_speed)
# knee
mono_loaded_knee_speed = utils.normalize_direction_data(exo_speed_dataset['monoarticular_ideal_loaded_kneeactuator_speed'],gl_noload,direction=True,normalize=False)
mono_noload_knee_speed = utils.normalize_direction_data(exo_speed_dataset['monoarticular_ideal_noload_kneeactuator_speed'],gl_noload,direction=True,normalize=False)
mean_mono_loaded_knee_speed,std_mono_loaded_knee_speed = utils.mean_std_over_subjects(mono_loaded_knee_speed)
mean_mono_noload_knee_speed,std_mono_noload_knee_speed = utils.mean_std_over_subjects(mono_noload_knee_speed)
#******************************
# hip muscles moment
# biarticular
bi_loaded_hipmuscles_moment = utils.normalize_direction_data(musclesmoment_dataset['biarticular_ideal_loaded_hip_musclesmoment'],gl_noload,direction=True)
bi_noload_hipmuscles_moment = utils.normalize_direction_data(musclesmoment_dataset['biarticular_ideal_noload_hip_musclesmoment'],gl_noload,direction=True)
mean_bi_loaded_hipmuscles_moment,std_bi_loaded_hipmuscles_moment = utils.mean_std_over_subjects(bi_loaded_hipmuscles_moment)
mean_bi_noload_hipmuscles_moment,std_bi_noload_hipmuscles_moment = utils.mean_std_over_subjects(bi_noload_hipmuscles_moment)
# monoarticular
mono_loaded_hipmuscles_moment = utils.normalize_direction_data(musclesmoment_dataset['monoarticular_ideal_loaded_hip_musclesmoment'],gl_noload,direction=True)
mono_noload_hipmuscles_moment = utils.normalize_direction_data(musclesmoment_dataset['monoarticular_ideal_noload_hip_musclesmoment'],gl_noload,direction=True)
mean_mono_loaded_hipmuscles_moment,std_mono_loaded_hipmuscles_moment = utils.mean_std_over_subjects(mono_loaded_hipmuscles_moment)
mean_mono_noload_hipmuscles_moment,std_mono_noload_hipmuscles_moment = utils.mean_std_over_subjects(mono_noload_hipmuscles_moment)
# knee muscles moment
# biarticular
bi_loaded_kneemuscles_moment = utils.normalize_direction_data(musclesmoment_dataset['biarticular_ideal_loaded_knee_musclesmoment'],gl_noload,direction=True)
bi_noload_kneemuscles_moment = utils.normalize_direction_data(musclesmoment_dataset['biarticular_ideal_noload_knee_musclesmoment'],gl_noload,direction=True)
mean_bi_loaded_kneemuscles_moment,std_bi_loaded_kneemuscles_moment = utils.mean_std_over_subjects(bi_loaded_kneemuscles_moment)
mean_bi_noload_kneemuscles_moment,std_bi_noload_kneemuscles_moment = utils.mean_std_over_subjects(bi_noload_kneemuscles_moment)
# monoarticular
mono_loaded_kneemuscles_moment = utils.normalize_direction_data(musclesmoment_dataset['monoarticular_ideal_loaded_knee_musclesmoment'],gl_noload,direction=True)
mono_noload_kneemuscles_moment = utils.normalize_direction_data(musclesmoment_dataset['monoarticular_ideal_noload_knee_musclesmoment'],gl_noload,direction=True)
mean_mono_loaded_kneemuscles_moment,std_mono_loaded_kneemuscles_moment = utils.mean_std_over_subjects(mono_loaded_kneemuscles_moment)
mean_mono_noload_kneemuscles_moment,std_mono_noload_kneemuscles_moment = utils.mean_std_over_subjects(mono_noload_kneemuscles_moment)

#####################################################################################
# Plots
# hip joint moment plot dictionaries
bi_loaded_hip_musclesmoment_dic = {'pgc':gait_cycle,'avg':utils.smooth(mean_bi_loaded_hipmuscles_moment,3),'label':'bi muscles',
                        'std':utils.smooth(std_bi_loaded_hipmuscles_moment,3),'avg_toeoff':loaded_mean_toe_off}
bi_noload_hip_musclesmoment_dic = {'pgc':gait_cycle,'avg':utils.smooth(mean_bi_noload_hipmuscles_moment,3),'label':'bi muscles',
                        'std':utils.smooth(std_bi_noload_hipmuscles_moment,3),'avg_toeoff':noload_mean_toe_off,'load':'noload'}
mono_loaded_hip_musclesmoment_dic = {'pgc':gait_cycle,'avg':utils.smooth(mean_mono_loaded_hipmuscles_moment,3),'label':'mono muscles',
                        'std':utils.smooth(std_mono_loaded_hipmuscles_moment,3),'avg_toeoff':loaded_mean_toe_off}
mono_noload_hip_musclesmoment_dic = {'pgc':gait_cycle,'avg':utils.smooth(mean_mono_noload_hipmuscles_moment,3),'label':'mono muscles ',
                        'std':utils.smooth(std_mono_noload_hipmuscles_moment,3),'avg_toeoff':noload_mean_toe_off,'load':'noload'}
unassist_loaded_hip_musclesmoment_dic = {'pgc':gait_cycle,'avg':utils.smooth(unassist_dataset['mean_norm_loaded_hipmuscles_moment'],3),'label':'loaded joint',
                        'std':utils.smooth(unassist_dataset['std_norm_loaded_hipmuscles_moment'],3),'avg_toeoff':loaded_mean_toe_off}
unassist_noload_hip_musclesmoment_dic = {'pgc':gait_cycle,'avg':utils.smooth(unassist_dataset['mean_norm_noload_hipmuscles_moment'],3),'label':'noload joint',
                        'std':utils.smooth(unassist_dataset['std_norm_noload_hipmuscles_moment'],3),'avg_toeoff':noload_mean_toe_off,'load':'noload'}

# knee joint moment plot dictionaries
bi_loaded_knee_musclesmoment_dic = {'pgc':gait_cycle,'avg':utils.smooth(mean_bi_loaded_kneemuscles_moment,3),'label':'bi muscles',
                        'std':utils.smooth(std_bi_loaded_kneemuscles_moment,3),'avg_toeoff':loaded_mean_toe_off}
bi_noload_knee_musclesmoment_dic = {'pgc':gait_cycle,'avg':utils.smooth(mean_bi_noload_kneemuscles_moment,3),'label':'bi muscles',
                        'std':utils.smooth(std_bi_noload_kneemuscles_moment,3),'avg_toeoff':noload_mean_toe_off,'load':'noload'}
mono_loaded_knee_musclesmoment_dic = {'pgc':gait_cycle,'avg':utils.smooth(mean_mono_loaded_kneemuscles_moment,3),'label':'mono muscles',
                        'std':utils.smooth(std_mono_loaded_kneemuscles_moment,3),'avg_toeoff':loaded_mean_toe_off}
mono_noload_knee_musclesmoment_dic = {'pgc':gait_cycle,'avg':utils.smooth(mean_mono_noload_kneemuscles_moment,3),'label':'mono muscles',
                        'std':utils.smooth(std_mono_noload_kneemuscles_moment,3),'avg_toeoff':noload_mean_toe_off,'load':'noload'}
unassist_loaded_knee_musclesmoment_dic = {'pgc':gait_cycle,'avg':utils.smooth(unassist_dataset['mean_norm_loaded_kneemuscles_moment'],3),'label':'loaded joint',
                        'std':utils.smooth(unassist_dataset['std_norm_loaded_kneemuscles_moment'],3),'avg_toeoff':loaded_mean_toe_off}
unassist_noload_knee_musclesmoment_dic = {'pgc':gait_cycle,'avg':utils.smooth(unassist_dataset['mean_norm_noload_kneemuscles_moment'],3),'label':'noload joint',
                        'std':utils.smooth(unassist_dataset['std_norm_noload_kneemuscles_moment'],3),'avg_toeoff':noload_mean_toe_off,'load':'noload'}

# hip actuator torque plot dictionaries
bi_loaded_hip_torque_dic= {'pgc':gait_cycle,'avg':utils.smooth(mean_bi_loaded_hip_torque,3),'label':'bi actuator',
                        'std':utils.smooth(std_bi_loaded_hip_torque,3),'avg_toeoff':loaded_mean_toe_off}
bi_noload_hip_torque_dic = {'pgc':gait_cycle,'avg':utils.smooth(mean_bi_noload_hip_torque,3),'label':'bi actuator',
                        'std':utils.smooth(std_bi_noload_hip_torque,3),'avg_toeoff':noload_mean_toe_off,'load':'noload'}
mono_loaded_hip_torque_dic = {'pgc':gait_cycle,'avg':utils.smooth(mean_mono_loaded_hip_torque,3),'label':'mono actuator',
                        'std':utils.smooth(std_mono_loaded_hip_torque,3),'avg_toeoff':loaded_mean_toe_off}
mono_noload_hip_torque_dic = {'pgc':gait_cycle,'avg':utils.smooth(mean_mono_noload_hip_torque,3),'label':'mono actuator',
                        'std':utils.smooth(std_mono_noload_hip_torque,3),'avg_toeoff':noload_mean_toe_off,'load':'noload'}

# knee actuator torque plot dictionaries
bi_loaded_knee_torque_dic= {'pgc':gait_cycle,'avg':utils.smooth(mean_bi_loaded_knee_torque,3),'label':'bi actuator',
                        'std':utils.smooth(std_bi_loaded_knee_torque,3),'avg_toeoff':loaded_mean_toe_off}
bi_noload_knee_torque_dic = {'pgc':gait_cycle,'avg':utils.smooth(mean_bi_noload_knee_torque,3),'label':'bi actuator',
                        'std':utils.smooth(std_bi_noload_knee_torque,3),'avg_toeoff':noload_mean_toe_off,'load':'noload'}
mono_loaded_knee_torque_dic = {'pgc':gait_cycle,'avg':utils.smooth(mean_mono_loaded_knee_torque,3),'label':'mono actuator',
                        'std':utils.smooth(std_mono_loaded_knee_torque,3),'avg_toeoff':loaded_mean_toe_off}
mono_noload_knee_torque_dic = {'pgc':gait_cycle,'avg':utils.smooth(mean_mono_noload_knee_torque,3),'label':'mono actuator',
                        'std':utils.smooth(std_mono_noload_knee_torque,3),'avg_toeoff':noload_mean_toe_off,'load':'noload'}

# joint power plot dictionaries
loaded_hip_power_dic = {'pgc':gait_cycle,'avg':utils.smooth(rra_dataset['mean_norm_loaded_hipjoint_power'],3),'label':'loaded joint',
                        'std':utils.smooth(rra_dataset['std_norm_loaded_hipjoint_power'],3),'avg_toeoff':loaded_mean_toe_off}
noload_hip_power_dic = {'pgc':gait_cycle,'avg':utils.smooth(rra_dataset['mean_norm_noload_hipjoint_power'],3),'label':'noload joint',
                        'std':utils.smooth(rra_dataset['std_norm_noload_hipjoint_power'],3),'avg_toeoff':noload_mean_toe_off,'load':'noload'}
loaded_knee_power_dic = {'pgc':gait_cycle,'avg':utils.smooth(rra_dataset['mean_norm_loaded_kneejoint_power'],3),'label':'loaded joint',
                        'std':utils.smooth(rra_dataset['std_norm_loaded_kneejoint_power'],3),'avg_toeoff':loaded_mean_toe_off}
noload_knee_power_dic = {'pgc':gait_cycle,'avg':utils.smooth(rra_dataset['mean_norm_noload_kneejoint_power'],3),'label':'noload joint',
                        'std':utils.smooth(rra_dataset['std_norm_noload_kneejoint_power'],3),'avg_toeoff':noload_mean_toe_off,'load':'noload'}

# hip actuator power plot dictionaries
bi_loaded_hip_power_dic= {'pgc':gait_cycle,'avg':utils.smooth(mean_bi_loaded_hip_power,3),'label':'bi actuator',
                        'std':utils.smooth(std_bi_loaded_hip_power,3),'avg_toeoff':loaded_mean_toe_off}
bi_noload_hip_power_dic = {'pgc':gait_cycle,'avg':utils.smooth(mean_bi_noload_hip_power,3),'label':'bi actuator',
                        'std':utils.smooth(std_bi_noload_hip_power,3),'avg_toeoff':noload_mean_toe_off,'load':'noload'}
mono_loaded_hip_power_dic = {'pgc':gait_cycle,'avg':utils.smooth(mean_mono_loaded_hip_power,3),'label':'mono actuator',
                        'std':utils.smooth(std_mono_loaded_hip_power,3),'avg_toeoff':loaded_mean_toe_off}
mono_noload_hip_power_dic = {'pgc':gait_cycle,'avg':utils.smooth(mean_mono_noload_hip_power,3),'label':'mono actuator',
                        'std':utils.smooth(std_mono_noload_hip_power,3),'avg_toeoff':noload_mean_toe_off,'load':'noload'}

# knee actuator power plot dictionaries
bi_loaded_knee_power_dic= {'pgc':gait_cycle,'avg':utils.smooth(mean_bi_loaded_knee_power,3),'label':'bi actuator',
                        'std':utils.smooth(std_bi_loaded_knee_power,3),'avg_toeoff':loaded_mean_toe_off}
bi_noload_knee_power_dic = {'pgc':gait_cycle,'avg':utils.smooth(mean_bi_noload_knee_power,3),'label':'bi actuator',
                        'std':utils.smooth(std_bi_noload_knee_power,3),'avg_toeoff':noload_mean_toe_off,'load':'noload'}
mono_loaded_knee_power_dic = {'pgc':gait_cycle,'avg':utils.smooth(mean_mono_loaded_knee_power,3),'label':'mono actuator',
                        'std':utils.smooth(std_mono_loaded_knee_power,3),'avg_toeoff':loaded_mean_toe_off}
mono_noload_knee_power_dic = {'pgc':gait_cycle,'avg':utils.smooth(mean_mono_noload_knee_power,3),'label':'mono actuator',
                        'std':utils.smooth(std_mono_noload_knee_power,3),'avg_toeoff':noload_mean_toe_off,'load':'noload'}

# joint speed plot dictionaries
loaded_hip_speed_dic = {'pgc':gait_cycle,'avg':utils.smooth(rra_dataset['mean_loaded_hipjoint_speed'],3),'label':'loaded joint',
                        'std':utils.smooth(rra_dataset['std_loaded_hipjoint_speed'],3),'avg_toeoff':loaded_mean_toe_off}
noload_hip_speed_dic = {'pgc':gait_cycle,'avg':utils.smooth(rra_dataset['mean_noload_hipjoint_speed'],3),'label':'noload joint',
                        'std':utils.smooth(rra_dataset['std_noload_hipjoint_speed'],3),'avg_toeoff':noload_mean_toe_off,'load':'noload'}
loaded_knee_speed_dic = {'pgc':gait_cycle,'avg':utils.smooth(rra_dataset['mean_loaded_kneejoint_speed'],3),'label':'loaded joint',
                        'std':utils.smooth(rra_dataset['std_loaded_kneejoint_speed'],3),'avg_toeoff':loaded_mean_toe_off}
noload_knee_speed_dic = {'pgc':gait_cycle,'avg':utils.smooth(rra_dataset['mean_noload_kneejoint_speed'],3),'label':'noload joint',
                        'std':utils.smooth(rra_dataset['std_noload_kneejoint_speed'],3),'avg_toeoff':noload_mean_toe_off,'load':'noload'}

# hip actuator speed plot dictionaries
bi_loaded_hip_speed_dic= {'pgc':gait_cycle,'avg':utils.smooth(mean_bi_loaded_hip_speed,3),'label':'bi actuator',
                        'std':utils.smooth(std_bi_loaded_hip_speed,3),'avg_toeoff':loaded_mean_toe_off}
bi_noload_hip_speed_dic = {'pgc':gait_cycle,'avg':utils.smooth(mean_bi_noload_hip_speed,3),'label':'bi actuator',
                        'std':utils.smooth(std_bi_noload_hip_speed,3),'avg_toeoff':noload_mean_toe_off,'load':'noload'}
mono_loaded_hip_speed_dic = {'pgc':gait_cycle,'avg':utils.smooth(mean_mono_loaded_hip_speed,3),'label':'mono actuator',
                        'std':utils.smooth(std_mono_loaded_hip_speed,3),'avg_toeoff':loaded_mean_toe_off}
mono_noload_hip_speed_dic = {'pgc':gait_cycle,'avg':utils.smooth(mean_mono_noload_hip_speed,3),'label':'mono actuator',
                        'std':utils.smooth(std_mono_noload_hip_speed,3),'avg_toeoff':noload_mean_toe_off,'load':'noload'}

# knee actuator speed plot dictionaries
bi_loaded_knee_speed_dic= {'pgc':gait_cycle,'avg':utils.smooth(mean_bi_loaded_knee_speed,3),'label':'bi actuator',
                        'std':utils.smooth(std_bi_loaded_knee_speed,3),'avg_toeoff':loaded_mean_toe_off}
bi_noload_knee_speed_dic = {'pgc':gait_cycle,'avg':utils.smooth(mean_bi_noload_knee_speed,3),'label':'bi actuator',
                        'std':utils.smooth(std_bi_noload_knee_speed,3),'avg_toeoff':noload_mean_toe_off,'load':'noload'}
mono_loaded_knee_speed_dic = {'pgc':gait_cycle,'avg':utils.smooth(mean_mono_loaded_knee_speed,3),'label':'mono actuator',
                        'std':utils.smooth(std_mono_loaded_knee_speed,3),'avg_toeoff':loaded_mean_toe_off}
mono_noload_knee_speed_dic = {'pgc':gait_cycle,'avg':utils.smooth(mean_mono_noload_knee_speed,3),'label':'mono actuator',
                        'std':utils.smooth(std_mono_noload_knee_speed,3),'avg_toeoff':noload_mean_toe_off,'load':'noload'}

#******************************************************************************************************************************
#******************************************************************************************************************************
# defualt color dictionary
monovsbi_color_dic = {
'color_1_list' : ['k','xkcd:irish green','k','xkcd:irish green'],
'color_2_list' : [mycolors['dark purple'],mycolors['lavender purple'],mycolors['dark purple'],mycolors['lavender purple']],
'color_3_list' : [mycolors['crimson red'],mycolors['french rose'],mycolors['crimson red'],mycolors['french rose']]
}

# ***************************
# hip joint moment figure
# required dictionary
plot_dic={
'plot_1_list' : [unassist_loaded_hip_musclesmoment_dic,unassist_noload_hip_musclesmoment_dic,\
                     unassist_loaded_hip_musclesmoment_dic,unassist_noload_hip_musclesmoment_dic],
'plot_2_list' : [mono_loaded_hip_torque_dic,mono_noload_hip_torque_dic,\
                    mono_loaded_hip_musclesmoment_dic,mono_noload_hip_musclesmoment_dic],
'plot_3_list' : [bi_loaded_hip_torque_dic,bi_noload_hip_torque_dic,\
                  bi_loaded_hip_musclesmoment_dic,bi_noload_hip_musclesmoment_dic],
'plot_titles' : ['loaded hip actuators','noload hip actuators','loaded hip muscles moment','noload hip muscles moment']
}
# plot
fig = plt.figure(num='Loaded Hip Torque Comparison',figsize=(9.4, 6.8))
utils.plot_joint_muscle_exo(nrows=2,ncols=2,plot_dic=plot_dic,color_dic=monovsbi_color_dic,ylabel='moment (N-m/kg)')
fig.tight_layout()
fig.savefig('./Figures/Ideal/Exoskeletons_Hip_Torque_Comparison.pdf',orientation='landscape',bbox_inches='tight')
plt.show()

# ***************************
# knee joint moment figure
# required dictionary
plot_dic={
'plot_1_list' : [unassist_loaded_knee_musclesmoment_dic,unassist_noload_knee_musclesmoment_dic,\
                     unassist_loaded_knee_musclesmoment_dic,unassist_noload_knee_musclesmoment_dic],
'plot_2_list' : [mono_loaded_knee_torque_dic,mono_noload_knee_torque_dic,\
                    mono_loaded_knee_musclesmoment_dic,mono_noload_knee_musclesmoment_dic],
'plot_3_list' : [bi_loaded_knee_torque_dic,bi_noload_knee_torque_dic,\
                  bi_loaded_knee_musclesmoment_dic,bi_noload_knee_musclesmoment_dic],
'plot_titles' : ['loaded knee actuators','noload knee actuators','loaded knee muscles moment','noload knee muscles moment']
}
# plot
fig = plt.figure(num='Loaded Knee Torque Comparison',figsize=(9.4, 6.8))
utils.plot_joint_muscle_exo(nrows=2,ncols=2,plot_dic=plot_dic,color_dic=monovsbi_color_dic,ylabel='moment (N-m/kg)')
fig.tight_layout()
fig.savefig('./Figures/Ideal/Exoskeletons_Knee_Torque_Comparison.pdf',orientation='landscape',bbox_inches='tight')
plt.show()

# ***************************
# loaded versus noload moment figure
# required dictionary
monovsbi_color_dic = {
'color_1_list' : [mycolors['dark purple'],mycolors['crimson red'],mycolors['dark purple'],mycolors['crimson red']],
'color_2_list' : [mycolors['french rose'],mycolors['lavender purple'],mycolors['french rose'],mycolors['lavender purple']]
}
plot_dic={
'plot_1_list' : [mono_loaded_hip_torque_dic,bi_loaded_hip_torque_dic,\
                 mono_loaded_knee_torque_dic,bi_loaded_knee_torque_dic],
'plot_2_list' : [mono_noload_hip_torque_dic,bi_noload_hip_torque_dic,\
                 mono_noload_knee_torque_dic,bi_noload_knee_torque_dic],
'plot_titles' : ['monoarticular hip actuator','monoarticular knee actuator','biarticular hip actuator','biarticular knee actuator']
}
# plot
fig = plt.figure(num='Loaded Versus Noload Torque Comparison',figsize=(9.4, 6.8))
utils.plot_joint_muscle_exo(nrows=2,ncols=2,plot_dic=plot_dic,color_dic=monovsbi_color_dic,ylabel='actuator torque (N-m/kg)',thirdplot=False)
fig.tight_layout()
fig.savefig('./Figures/Ideal/Exoskeletons_LoadedVSNoload_Torque_Comparison.pdf',orientation='landscape',bbox_inches='tight')
plt.show()

#******************************************************************************************************************************
#******************************************************************************************************************************
# defualt color dictionary
monovsbi_color_dic = {
'color_1_list' : ['k','xkcd:irish green','k','xkcd:irish green'],
'color_2_list' : [mycolors['dark purple'],mycolors['lavender purple'],mycolors['dark purple'],mycolors['lavender purple']],
'color_3_list' : [mycolors['crimson red'],mycolors['french rose'],mycolors['crimson red'],mycolors['french rose']]
}

# ***************************
# hip joint moment figure
# required dictionary
plot_dic={
'plot_1_list' : [loaded_hip_power_dic,noload_hip_power_dic,\
                 loaded_knee_power_dic,noload_knee_power_dic],
'plot_2_list' : [mono_loaded_hip_power_dic,mono_noload_hip_power_dic,\
                 mono_loaded_knee_power_dic,mono_noload_knee_power_dic],
'plot_3_list' : [bi_loaded_hip_power_dic,bi_noload_hip_power_dic,\
                 bi_loaded_knee_power_dic,bi_noload_knee_power_dic],
'plot_titles' : ['loaded hip actuators','noload hip actuators','loaded knee actuators','noload knee actuators']
}
# plot
fig = plt.figure(num='Power Comparison',figsize=(9.4, 6.8))
utils.plot_joint_muscle_exo(nrows=2,ncols=2,plot_dic=plot_dic,color_dic=monovsbi_color_dic,ylabel=' actuator power (W/kg)',y_ticks=np.arange(-4,6,2))
fig.tight_layout()
fig.savefig('./Figures/Ideal/Exoskeletons_Power_Comparison.pdf',orientation='landscape',bbox_inches='tight')
plt.show()

# ***************************
# loaded versus noload moment figure
# required dictionary
monovsbi_color_dic = {
'color_1_list' : [mycolors['dark purple'],mycolors['crimson red'],mycolors['dark purple'],mycolors['crimson red']],
'color_2_list' : [mycolors['french rose'],mycolors['lavender purple'],mycolors['french rose'],mycolors['lavender purple']]
}
plot_dic={
'plot_1_list' : [mono_loaded_hip_power_dic,bi_loaded_hip_power_dic,\
                 mono_loaded_knee_power_dic,bi_loaded_knee_power_dic],
'plot_2_list' : [mono_noload_hip_power_dic,bi_noload_hip_power_dic,\
                 mono_noload_knee_power_dic,bi_noload_knee_power_dic],
'plot_titles' : [ 'monoarticular hip actuators','biarticular hip actuators','monoarticular knee actuators','biarticular knee actuators']
}
# plot
fig = plt.figure(num='Loaded Versus Noload Power Comparison',figsize=(9.4, 6.8))
utils.plot_joint_muscle_exo(nrows=2,ncols=2,plot_dic=plot_dic,color_dic=monovsbi_color_dic,ylabel='actuator power (W/kg)',y_ticks=np.arange(-4,6,2),thirdplot=False)
fig.tight_layout()
fig.savefig('./Figures/Ideal/Exoskeletons_LoadedVSNoload_Power_Comparison.pdf',orientation='landscape',bbox_inches='tight')
plt.show()
