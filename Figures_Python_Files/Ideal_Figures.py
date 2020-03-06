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
# assisted subjects muscles activation dataset
directory = 'D:/Ali.K.M.Bonab/Walking_Mass_Inertia_Effect/Data/Data/Ideal/*_activation.csv'
files = enumerate(glob.iglob(directory), 1)
musclesactivation_dataset = {pathlib.PurePath(f[1]).stem: np.loadtxt(f[1], delimiter=',') for f in files}
# assisted subjects energy dataset
directory = 'D:/Ali.K.M.Bonab/Walking_Mass_Inertia_Effect/Data/Data/Ideal/*_energy.csv'
files = enumerate(glob.iglob(directory), 1)
assisted_energy_dataset = {pathlib.PurePath(f[1]).stem: np.loadtxt(f[1], delimiter=',') for f in files}
# unassisted subjects energy dataset
directory = 'D:/Ali.K.M.Bonab/Walking_Mass_Inertia_Effect/Data/Data/Unassist/*_energy.csv'
files = enumerate(glob.iglob(directory), 1)
unasisted_energy_dataset = {pathlib.PurePath(f[1]).stem: np.loadtxt(f[1], delimiter=',') for f in files}
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
bi_loaded_hip_torque = utils.normalize_direction_data(exo_torque_dataset['biarticular_ideal_loaded_hipactuator_torque'],gl_noload,direction=True)
bi_noload_hip_torque = utils.normalize_direction_data(exo_torque_dataset['biarticular_ideal_noload_hipactuator_torque'],gl_noload,direction=True)
mean_bi_loaded_hip_torque,std_bi_loaded_hip_torque = utils.mean_std_over_subjects(bi_loaded_hip_torque)
mean_bi_noload_hip_torque,std_bi_noload_hip_torque = utils.mean_std_over_subjects(bi_noload_hip_torque)
# knee
bi_loaded_knee_torque = utils.normalize_direction_data(exo_torque_dataset['biarticular_ideal_loaded_kneeactuator_torque'],gl_noload,direction=True)
bi_noload_knee_torque = utils.normalize_direction_data(exo_torque_dataset['biarticular_ideal_noload_kneeactuator_torque'],gl_noload,direction=True)
mean_bi_loaded_knee_torque,std_bi_loaded_knee_torque = utils.mean_std_over_subjects(bi_loaded_knee_torque)
mean_bi_noload_knee_torque,std_bi_noload_knee_torque = utils.mean_std_over_subjects(bi_noload_knee_torque)
# monoarticular
# hip
mono_loaded_hip_torque = utils.normalize_direction_data(exo_torque_dataset['monoarticular_ideal_loaded_hipactuator_torque'],gl_noload,direction=True)
mono_noload_hip_torque = utils.normalize_direction_data(exo_torque_dataset['monoarticular_ideal_noload_hipactuator_torque'],gl_noload,direction=True)
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
bi_loaded_hip_power = utils.normalize_direction_data(exo_power_dataset['biarticular_ideal_loaded_hipactuator_power'],gl_noload,direction=True)
bi_noload_hip_power = utils.normalize_direction_data(exo_power_dataset['biarticular_ideal_noload_hipactuator_power'],gl_noload,direction=True)
mean_bi_loaded_hip_power,std_bi_loaded_hip_power = utils.mean_std_over_subjects(bi_loaded_hip_power)
mean_bi_noload_hip_power,std_bi_noload_hip_power = utils.mean_std_over_subjects(bi_noload_hip_power)
# knee
bi_loaded_knee_power = utils.normalize_direction_data(exo_power_dataset['biarticular_ideal_loaded_kneeactuator_power'],gl_noload,direction=True)
bi_noload_knee_power = utils.normalize_direction_data(exo_power_dataset['biarticular_ideal_noload_kneeactuator_power'],gl_noload,direction=True)
mean_bi_loaded_knee_power,std_bi_loaded_knee_power = utils.mean_std_over_subjects(bi_loaded_knee_power)
mean_bi_noload_knee_power,std_bi_noload_knee_power = utils.mean_std_over_subjects(bi_noload_knee_power)
# monoarticular
# hip
mono_loaded_hip_power = utils.normalize_direction_data(exo_power_dataset['monoarticular_ideal_loaded_hipactuator_power'],gl_noload,direction=True)
mono_noload_hip_power = utils.normalize_direction_data(exo_power_dataset['monoarticular_ideal_noload_hipactuator_power'],gl_noload,direction=True)
mean_mono_loaded_hip_power,std_mono_loaded_hip_power = utils.mean_std_over_subjects(mono_loaded_hip_power)
mean_mono_noload_hip_power,std_mono_noload_hip_power = utils.mean_std_over_subjects(mono_noload_hip_power)
# knee
mono_loaded_knee_power = utils.normalize_direction_data(exo_power_dataset['monoarticular_ideal_loaded_kneeactuator_power'],gl_noload,direction=True)
mono_noload_knee_power = utils.normalize_direction_data(exo_power_dataset['monoarticular_ideal_noload_kneeactuator_power'],gl_noload,direction=True)
mean_mono_loaded_knee_power,std_mono_loaded_knee_power = utils.mean_std_over_subjects(mono_loaded_knee_power)
mean_mono_noload_knee_power,std_mono_noload_knee_power = utils.mean_std_over_subjects(mono_noload_knee_power)
#******************************
# exoskeleton speed profiles
# biarticular
# hip
bi_loaded_hip_speed = utils.normalize_direction_data(exo_speed_dataset['biarticular_ideal_loaded_hipactuator_speed'],gl_noload,direction=True)
bi_noload_hip_speed = utils.normalize_direction_data(exo_speed_dataset['biarticular_ideal_noload_hipactuator_speed'],gl_noload,direction=True)
mean_bi_loaded_hip_speed,std_bi_loaded_hip_speed = utils.mean_std_over_subjects(bi_loaded_hip_speed)
mean_bi_noload_hip_speed,std_bi_noload_hip_speed = utils.mean_std_over_subjects(bi_noload_hip_speed)
# knee
bi_loaded_knee_speed = utils.normalize_direction_data(exo_speed_dataset['biarticular_ideal_loaded_kneeactuator_speed'],gl_noload,direction=True)
bi_noload_knee_speed = utils.normalize_direction_data(exo_speed_dataset['biarticular_ideal_noload_kneeactuator_speed'],gl_noload,direction=True)
mean_bi_loaded_knee_speed,std_bi_loaded_knee_speed = utils.mean_std_over_subjects(bi_loaded_knee_speed)
mean_bi_noload_knee_speed,std_bi_noload_knee_speed = utils.mean_std_over_subjects(bi_noload_knee_speed)
# monoarticular
# hip
mono_loaded_hip_speed = utils.normalize_direction_data(exo_speed_dataset['monoarticular_ideal_loaded_hipactuator_speed'],gl_noload,direction=True)
mono_noload_hip_speed = utils.normalize_direction_data(exo_speed_dataset['monoarticular_ideal_noload_hipactuator_speed'],gl_noload,direction=True)
mean_mono_loaded_hip_speed,std_mono_loaded_hip_speed = utils.mean_std_over_subjects(mono_loaded_hip_speed)
mean_mono_noload_hip_speed,std_mono_noload_hip_speed = utils.mean_std_over_subjects(mono_noload_hip_speed)
# knee
mono_loaded_knee_speed = utils.normalize_direction_data(exo_speed_dataset['monoarticular_ideal_loaded_kneeactuator_speed'],gl_noload,direction=True)
mono_noload_knee_speed = utils.normalize_direction_data(exo_speed_dataset['monoarticular_ideal_noload_kneeactuator_speed'],gl_noload,direction=True)
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
# muscles activation
# biarticular
bi_loaded_muscles_activation = utils.normalize_direction_data(musclesactivation_dataset['biarticular_ideal_loaded_ninemuscles_activation'],gl_noload,direction=False,normalize=False)
bi_noload_muscles_activation = utils.normalize_direction_data(musclesactivation_dataset['biarticular_ideal_noload_ninemuscles_activation'],gl_noload,direction=False,normalize=False)
mean_bi_loaded_muscles_activation,std_norm_loaded_muscles_activation = utils.mean_std_muscles_subjects(bi_loaded_muscles_activation)
mean_bi_noload_muscles_activation,std_norm_noload_muscles_activation = utils.mean_std_muscles_subjects(bi_noload_muscles_activation)
# monoarticular
mono_loaded_muscles_activation = utils.normalize_direction_data(musclesactivation_dataset['monoarticular_ideal_loaded_ninemuscles_activation'],gl_noload,direction=False,normalize=False)
mono_noload_muscles_activation = utils.normalize_direction_data(musclesactivation_dataset['monoarticular_ideal_noload_ninemuscles_activation'],gl_noload,direction=False,normalize=False)
mean_mono_loaded_muscles_activation,std_norm_loaded_muscles_activation = utils.mean_std_muscles_subjects(mono_loaded_muscles_activation)
mean_mono_noload_muscles_activation,std_norm_noload_muscles_activation = utils.mean_std_muscles_subjects(mono_noload_muscles_activation)

#####################################################################################
# Plots
# hip joint moment plot dictionaries
hip_moment_loaded_plot_dic = {'pgc':gait_cycle,'avg':utils.smooth(mean_norm_loaded_hipmuscles_moment,3),'label':'Loaded',
                        'std':utils.smooth(std_norm_loaded_hipmuscles_moment,3),'avg_toeoff':loaded_mean_toe_off}
hip_moment_noload_plot_dic = {'pgc':gait_cycle,'avg':utils.smooth(mean_norm_noload_hipmuscles_moment,3),'label':'Noload',
                        'std':utils.smooth(std_norm_noload_hipmuscles_moment,3),'avg_toeoff':noload_mean_toe_off}
# knee joint moment plot dictionaries
knee_moment_loaded_plot_dic = {'pgc':gait_cycle,'avg':utils.smooth(mean_norm_loaded_kneemuscles_moment,3),'label':'Loaded',
                        'std':utils.smooth(std_norm_loaded_kneemuscles_moment,3),'avg_toeoff':loaded_mean_toe_off}
knee_moment_noload_plot_dic = {'pgc':gait_cycle,'avg':utils.smooth(mean_norm_noload_kneemuscles_moment,3),'label':'Noload',
                        'std':utils.smooth(std_norm_noload_kneemuscles_moment,3),'avg_toeoff':noload_mean_toe_off}
# muscles moment plot dictionaries
muscles_activation_loaded_plot_dic = {'pgc':gait_cycle,'avg':mean_norm_loaded_muscles_activation,'muscle_group': 'nine_muscles',
                                      'std':std_norm_loaded_muscles_activation,'avg_toeoff':loaded_mean_toe_off}
muscles_activation_noload_plot_dic = {'pgc':gait_cycle,'avg':mean_norm_noload_muscles_activation,'muscle_group': 'nine_muscles',
                                      'std':std_norm_noload_muscles_activation,'avg_toeoff':noload_mean_toe_off}

#*****************************

# hip joint moment figure
fig, ax = plt.subplots(num='Hip Muscles Moment',figsize=(6.4, 4.8))
utils.plot_shaded_avg(plot_dic=hip_moment_loaded_plot_dic,color='k')
utils.plot_shaded_avg(plot_dic=hip_moment_noload_plot_dic,toeoff_color='xkcd:shamrock green',color='xkcd:irish green')
plt.legend(loc='best',frameon=False)
plt.ylim((-1.5,2))
plt.xlabel('gait cycle (%)')
plt.ylabel('hip muscles moment (N-m/kg)')
utils.no_top_right(ax)
plt.show()
fig.tight_layout(h_pad=-4.0, w_pad=-4.0)
fig.savefig('./Figures/Unassist/HipMusclesMoment.pdf',orientation='landscape',bbox_inches='tight')

# knee joint moment figure
fig, ax = plt.subplots(num='Knee Muscles Moment',figsize=(6.4, 4.8))
utils.plot_shaded_avg(plot_dic=knee_moment_loaded_plot_dic,color='k')
utils.plot_shaded_avg(plot_dic=knee_moment_noload_plot_dic,toeoff_color='xkcd:shamrock green',color='xkcd:irish green')
plt.legend(loc='best',frameon=False)
plt.ylim((-1.5,2))
plt.xlabel('gait cycle (%)')
plt.ylabel('knee muscles moment (N-m/kg)')
utils.no_top_right(ax)
plt.show()
fig.tight_layout(h_pad=-4.0, w_pad=-4.0)
fig.savefig('./Figures/Unassist/KneeMusclesMoment.pdf',orientation='landscape',bbox_inches='tight')

# muscles activation figure
fig, ax = plt.subplots(num='Muscles Activation',figsize=(8.4, 6.8))
utils.plot_muscles_avg(plot_dic=muscles_activation_loaded_plot_dic,color='k',is_std=True)
utils.plot_muscles_avg(plot_dic=muscles_activation_noload_plot_dic,toeoff_color='xkcd:shamrock green',color='xkcd:irish green',is_std=True)
plt.show()
fig.tight_layout()
fig.savefig('./Figures/Unassist/NineMusclesActivation.pdf',orientation='landscape',bbox_inches='tight')
