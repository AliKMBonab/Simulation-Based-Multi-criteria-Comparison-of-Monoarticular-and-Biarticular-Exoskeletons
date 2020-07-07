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
# ideal exo torque dataset
directory = './Data/Ideal/*_torque.csv'
files = enumerate(glob.iglob(directory), 1)
exo_torque_dataset = {pathlib.PurePath(f[1]).stem: np.loadtxt(f[1], delimiter=',') for f in files}
# ideal exo power dataset
directory = './Data/Ideal/*_power.csv'
files = enumerate(glob.iglob(directory), 1)
exo_power_dataset = {pathlib.PurePath(f[1]).stem: np.loadtxt(f[1], delimiter=',') for f in files}
# ideal exo speed dataset
directory = './Data/Ideal/*_speed.csv'
files = enumerate(glob.iglob(directory), 1)
exo_speed_dataset = {pathlib.PurePath(f[1]).stem: np.loadtxt(f[1], delimiter=',') for f in files}
# muscles moment dataset
directory = './Data/Ideal/*_musclesmoment.csv'
files = enumerate(glob.iglob(directory), 1)
musclesmoment_dataset = {pathlib.PurePath(f[1]).stem: np.loadtxt(f[1], delimiter=',') for f in files}
# muscles power dataset
directory = './Data/Ideal/*_musclespower.csv'
files = enumerate(glob.iglob(directory), 1)
musclespower_dataset = {pathlib.PurePath(f[1]).stem: np.loadtxt(f[1], delimiter=',') for f in files}
# assisted subjects muscles activation dataset
directory = './Data/Ideal/*_activation.csv'
files = enumerate(glob.iglob(directory), 1)
musclesactivation_dataset = {pathlib.PurePath(f[1]).stem: np.loadtxt(f[1], delimiter=',') for f in files}
# assisted subjects energy dataset
directory = './Data/Ideal/*_energy.csv'
files = enumerate(glob.iglob(directory), 1)
assisted_energy_dataset = {pathlib.PurePath(f[1]).stem: np.loadtxt(f[1], delimiter=',') for f in files}
# unassisted subjects energy dataset
directory = './Data/Unassist/*_energy.csv'
files = enumerate(glob.iglob(directory), 1)
unassisted_energy_dataset = {pathlib.PurePath(f[1]).stem: np.loadtxt(f[1], delimiter=',') for f in files}
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
mean_bi_loaded_hip_torque,std_bi_loaded_hip_torque = utils.mean_std_over_subjects(bi_loaded_hip_torque,avg_trials=False)
mean_bi_noload_hip_torque,std_bi_noload_hip_torque = utils.mean_std_over_subjects(bi_noload_hip_torque,avg_trials=False)
# knee
bi_loaded_knee_torque = utils.normalize_direction_data(exo_torque_dataset['biarticular_ideal_loaded_kneeactuator_torque'],gl_noload,direction=True)
bi_noload_knee_torque = utils.normalize_direction_data(exo_torque_dataset['biarticular_ideal_noload_kneeactuator_torque'],gl_noload,direction=True)
mean_bi_loaded_knee_torque,std_bi_loaded_knee_torque = utils.mean_std_over_subjects(bi_loaded_knee_torque,avg_trials=False)
mean_bi_noload_knee_torque,std_bi_noload_knee_torque = utils.mean_std_over_subjects(bi_noload_knee_torque,avg_trials=False)
# monoarticular
# hip
mono_loaded_hip_torque = utils.normalize_direction_data(exo_torque_dataset['monoarticular_ideal_loaded_hipactuator_torque'],gl_noload,direction=False)
mono_noload_hip_torque = utils.normalize_direction_data(exo_torque_dataset['monoarticular_ideal_noload_hipactuator_torque'],gl_noload,direction=False)
mean_mono_loaded_hip_torque,std_mono_loaded_hip_torque = utils.mean_std_over_subjects(mono_loaded_hip_torque,avg_trials=False)
mean_mono_noload_hip_torque,std_mono_noload_hip_torque = utils.mean_std_over_subjects(mono_noload_hip_torque,avg_trials=False)
# knee
mono_loaded_knee_torque = utils.normalize_direction_data(exo_torque_dataset['monoarticular_ideal_loaded_kneeactuator_torque'],gl_noload,direction=True)
mono_noload_knee_torque = utils.normalize_direction_data(exo_torque_dataset['monoarticular_ideal_noload_kneeactuator_torque'],gl_noload,direction=True)
mean_mono_loaded_knee_torque,std_mono_loaded_knee_torque = utils.mean_std_over_subjects(mono_loaded_knee_torque,avg_trials=False)
mean_mono_noload_knee_torque,std_mono_noload_knee_torque = utils.mean_std_over_subjects(mono_noload_knee_torque,avg_trials=False)
#******************************
# exoskeleton power profiles
# biarticular
# hip
bi_loaded_hip_power = utils.normalize_direction_data(exo_power_dataset['biarticular_ideal_loaded_hipactuator_power'],gl_noload,direction=False)
bi_noload_hip_power = utils.normalize_direction_data(exo_power_dataset['biarticular_ideal_noload_hipactuator_power'],gl_noload,direction=False)
mean_bi_loaded_hip_power,std_bi_loaded_hip_power = utils.mean_std_over_subjects(bi_loaded_hip_power,avg_trials=False)
mean_bi_noload_hip_power,std_bi_noload_hip_power = utils.mean_std_over_subjects(bi_noload_hip_power,avg_trials=False)
# knee
bi_loaded_knee_power = utils.normalize_direction_data(exo_power_dataset['biarticular_ideal_loaded_kneeactuator_power'],gl_noload,direction=False)
bi_noload_knee_power = utils.normalize_direction_data(exo_power_dataset['biarticular_ideal_noload_kneeactuator_power'],gl_noload,direction=False)
mean_bi_loaded_knee_power,std_bi_loaded_knee_power = utils.mean_std_over_subjects(bi_loaded_knee_power,avg_trials=False)
mean_bi_noload_knee_power,std_bi_noload_knee_power = utils.mean_std_over_subjects(bi_noload_knee_power,avg_trials=False)
# monoarticular
# hip
mono_loaded_hip_power = utils.normalize_direction_data(exo_power_dataset['monoarticular_ideal_loaded_hipactuator_power'],gl_noload,direction=False)
mono_noload_hip_power = utils.normalize_direction_data(exo_power_dataset['monoarticular_ideal_noload_hipactuator_power'],gl_noload,direction=False)
mean_mono_loaded_hip_power,std_mono_loaded_hip_power = utils.mean_std_over_subjects(mono_loaded_hip_power,avg_trials=False)
mean_mono_noload_hip_power,std_mono_noload_hip_power = utils.mean_std_over_subjects(mono_noload_hip_power,avg_trials=False)
# knee
mono_loaded_knee_power = utils.normalize_direction_data(exo_power_dataset['monoarticular_ideal_loaded_kneeactuator_power'],gl_noload,direction=False)
mono_noload_knee_power = utils.normalize_direction_data(exo_power_dataset['monoarticular_ideal_noload_kneeactuator_power'],gl_noload,direction=False)
mean_mono_loaded_knee_power,std_mono_loaded_knee_power = utils.mean_std_over_subjects(mono_loaded_knee_power,avg_trials=False)
mean_mono_noload_knee_power,std_mono_noload_knee_power = utils.mean_std_over_subjects(mono_noload_knee_power,avg_trials=False)
#******************************
# exoskeleton speed profiles
# biarticular
# hip
bi_loaded_hip_speed = utils.normalize_direction_data(exo_speed_dataset['biarticular_ideal_loaded_hipactuator_speed'],gl_noload,direction=False,normalize=False)
bi_noload_hip_speed = utils.normalize_direction_data(exo_speed_dataset['biarticular_ideal_noload_hipactuator_speed'],gl_noload,direction=False,normalize=False)
mean_bi_loaded_hip_speed,std_bi_loaded_hip_speed = utils.mean_std_over_subjects(bi_loaded_hip_speed,avg_trials=False)
mean_bi_noload_hip_speed,std_bi_noload_hip_speed = utils.mean_std_over_subjects(bi_noload_hip_speed,avg_trials=False)
# knee
bi_loaded_knee_speed = utils.normalize_direction_data(exo_speed_dataset['biarticular_ideal_loaded_kneeactuator_speed'],gl_noload,direction=True,normalize=False)
bi_noload_knee_speed = utils.normalize_direction_data(exo_speed_dataset['biarticular_ideal_noload_kneeactuator_speed'],gl_noload,direction=True,normalize=False)
mean_bi_loaded_knee_speed,std_bi_loaded_knee_speed = utils.mean_std_over_subjects(bi_loaded_knee_speed,avg_trials=False)
mean_bi_noload_knee_speed,std_bi_noload_knee_speed = utils.mean_std_over_subjects(bi_noload_knee_speed,avg_trials=False)
# monoarticular
# hip
mono_loaded_hip_speed = utils.normalize_direction_data(exo_speed_dataset['monoarticular_ideal_loaded_hipactuator_speed'],gl_noload,direction=False,normalize=False)
mono_noload_hip_speed = utils.normalize_direction_data(exo_speed_dataset['monoarticular_ideal_noload_hipactuator_speed'],gl_noload,direction=False,normalize=False)
mean_mono_loaded_hip_speed,std_mono_loaded_hip_speed = utils.mean_std_over_subjects(mono_loaded_hip_speed,avg_trials=False)
mean_mono_noload_hip_speed,std_mono_noload_hip_speed = utils.mean_std_over_subjects(mono_noload_hip_speed,avg_trials=False)
# knee
mono_loaded_knee_speed = utils.normalize_direction_data(exo_speed_dataset['monoarticular_ideal_loaded_kneeactuator_speed'],gl_noload,direction=True,normalize=False)
mono_noload_knee_speed = utils.normalize_direction_data(exo_speed_dataset['monoarticular_ideal_noload_kneeactuator_speed'],gl_noload,direction=True,normalize=False)
mean_mono_loaded_knee_speed,std_mono_loaded_knee_speed = utils.mean_std_over_subjects(mono_loaded_knee_speed,avg_trials=False)
mean_mono_noload_knee_speed,std_mono_noload_knee_speed = utils.mean_std_over_subjects(mono_noload_knee_speed,avg_trials=False)
#******************************
# hip muscles moment
# biarticular
bi_loaded_hipmuscles_moment = utils.normalize_direction_data(musclesmoment_dataset['biarticular_ideal_loaded_hip_musclesmoment'],gl_noload,direction=True)
bi_noload_hipmuscles_moment = utils.normalize_direction_data(musclesmoment_dataset['biarticular_ideal_noload_hip_musclesmoment'],gl_noload,direction=True)
mean_bi_loaded_hipmuscles_moment,std_bi_loaded_hipmuscles_moment = utils.mean_std_over_subjects(bi_loaded_hipmuscles_moment,avg_trials=False)
mean_bi_noload_hipmuscles_moment,std_bi_noload_hipmuscles_moment = utils.mean_std_over_subjects(bi_noload_hipmuscles_moment,avg_trials=False)
# monoarticular
mono_loaded_hipmuscles_moment = utils.normalize_direction_data(musclesmoment_dataset['monoarticular_ideal_loaded_hip_musclesmoment'],gl_noload,direction=True)
mono_noload_hipmuscles_moment = utils.normalize_direction_data(musclesmoment_dataset['monoarticular_ideal_noload_hip_musclesmoment'],gl_noload,direction=True)
mean_mono_loaded_hipmuscles_moment,std_mono_loaded_hipmuscles_moment = utils.mean_std_over_subjects(mono_loaded_hipmuscles_moment,avg_trials=False)
mean_mono_noload_hipmuscles_moment,std_mono_noload_hipmuscles_moment = utils.mean_std_over_subjects(mono_noload_hipmuscles_moment,avg_trials=False)
# knee muscles moment
# biarticular
bi_loaded_kneemuscles_moment = utils.normalize_direction_data(musclesmoment_dataset['biarticular_ideal_loaded_knee_musclesmoment'],gl_noload,direction=True)
bi_noload_kneemuscles_moment = utils.normalize_direction_data(musclesmoment_dataset['biarticular_ideal_noload_knee_musclesmoment'],gl_noload,direction=True)
mean_bi_loaded_kneemuscles_moment,std_bi_loaded_kneemuscles_moment = utils.mean_std_over_subjects(bi_loaded_kneemuscles_moment,avg_trials=False)
mean_bi_noload_kneemuscles_moment,std_bi_noload_kneemuscles_moment = utils.mean_std_over_subjects(bi_noload_kneemuscles_moment,avg_trials=False)
# monoarticular
mono_loaded_kneemuscles_moment = utils.normalize_direction_data(musclesmoment_dataset['monoarticular_ideal_loaded_knee_musclesmoment'],gl_noload,direction=True)
mono_noload_kneemuscles_moment = utils.normalize_direction_data(musclesmoment_dataset['monoarticular_ideal_noload_knee_musclesmoment'],gl_noload,direction=True)
mean_mono_loaded_kneemuscles_moment,std_mono_loaded_kneemuscles_moment = utils.mean_std_over_subjects(mono_loaded_kneemuscles_moment,avg_trials=False)
mean_mono_noload_kneemuscles_moment,std_mono_noload_kneemuscles_moment = utils.mean_std_over_subjects(mono_noload_kneemuscles_moment,avg_trials=False)

#******************************
# hip muscles power
# biarticular
bi_loaded_hipmuscles_power = utils.normalize_direction_data(musclespower_dataset['biarticular_ideal_loaded_hip_musclespower'],gl_noload,direction=False)
bi_noload_hipmuscles_power = utils.normalize_direction_data(musclespower_dataset['biarticular_ideal_noload_hip_musclespower'],gl_noload,direction=False)
mean_bi_loaded_hipmuscles_power,std_bi_loaded_hipmuscles_power = utils.mean_std_over_subjects(bi_loaded_hipmuscles_power,avg_trials=False)
mean_bi_noload_hipmuscles_power,std_bi_noload_hipmuscles_power = utils.mean_std_over_subjects(bi_noload_hipmuscles_power,avg_trials=False)
# monoarticular
mono_loaded_hipmuscles_power = utils.normalize_direction_data(musclespower_dataset['monoarticular_ideal_loaded_hip_musclespower'],gl_noload,direction=False)
mono_noload_hipmuscles_power = utils.normalize_direction_data(musclespower_dataset['monoarticular_ideal_noload_hip_musclespower'],gl_noload,direction=False)
mean_mono_loaded_hipmuscles_power,std_mono_loaded_hipmuscles_power = utils.mean_std_over_subjects(mono_loaded_hipmuscles_power,avg_trials=False)
mean_mono_noload_hipmuscles_power,std_mono_noload_hipmuscles_power = utils.mean_std_over_subjects(mono_noload_hipmuscles_power,avg_trials=False)
# knee muscles power
# biarticular
bi_loaded_kneemuscles_power = utils.normalize_direction_data(musclespower_dataset['biarticular_ideal_loaded_knee_musclespower'],gl_noload,direction=False)
bi_noload_kneemuscles_power = utils.normalize_direction_data(musclespower_dataset['biarticular_ideal_noload_knee_musclespower'],gl_noload,direction=False)
mean_bi_loaded_kneemuscles_power,std_bi_loaded_kneemuscles_power = utils.mean_std_over_subjects(bi_loaded_kneemuscles_power,avg_trials=False)
mean_bi_noload_kneemuscles_power,std_bi_noload_kneemuscles_power = utils.mean_std_over_subjects(bi_noload_kneemuscles_power,avg_trials=False)
# monoarticular
mono_loaded_kneemuscles_power = utils.normalize_direction_data(musclespower_dataset['monoarticular_ideal_loaded_knee_musclespower'],gl_noload,direction=False)
mono_noload_kneemuscles_power = utils.normalize_direction_data(musclespower_dataset['monoarticular_ideal_noload_knee_musclespower'],gl_noload,direction=False)
mean_mono_loaded_kneemuscles_power,std_mono_loaded_kneemuscles_power = utils.mean_std_over_subjects(mono_loaded_kneemuscles_power,avg_trials=False)
mean_mono_noload_kneemuscles_power,std_mono_noload_kneemuscles_power = utils.mean_std_over_subjects(mono_noload_kneemuscles_power,avg_trials=False)

#******************************
# muscles activation
# biarticular
bi_loaded_muscles_activation = utils.normalize_direction_data(musclesactivation_dataset['biarticular_ideal_loaded_ninemuscles_activation'],gl_noload,direction=False,normalize=False)
bi_noload_muscles_activation = utils.normalize_direction_data(musclesactivation_dataset['biarticular_ideal_noload_ninemuscles_activation'],gl_noload,direction=False,normalize=False)
mean_bi_loaded_muscles_activation,std_bi_loaded_muscles_activation = utils.mean_std_muscles_subjects(bi_loaded_muscles_activation)
mean_bi_noload_muscles_activation,std_bi_noload_muscles_activation = utils.mean_std_muscles_subjects(bi_noload_muscles_activation)
# monoarticular
mono_loaded_muscles_activation = utils.normalize_direction_data(musclesactivation_dataset['monoarticular_ideal_loaded_ninemuscles_activation'],gl_noload,direction=False,normalize=False)
mono_noload_muscles_activation = utils.normalize_direction_data(musclesactivation_dataset['monoarticular_ideal_noload_ninemuscles_activation'],gl_noload,direction=False,normalize=False)
mean_mono_loaded_muscles_activation,std_mono_loaded_muscles_activation = utils.mean_std_muscles_subjects(mono_loaded_muscles_activation)
mean_mono_noload_muscles_activation,std_mono_noload_muscles_activation = utils.mean_std_muscles_subjects(mono_noload_muscles_activation)
#unassist
mean_unassist_loaded_muscles_activation = utils.recover_muscledata(unassist_dataset,'mean_norm_loaded_muscles_activation')
std_unassist_loaded_muscles_activation  = utils.recover_muscledata(unassist_dataset,'std_norm_loaded_muscles_activation')
mean_unassist_noload_muscles_activation = utils.recover_muscledata(unassist_dataset,'mean_norm_noload_muscles_activation')
std_unassist_noload_muscles_activation  = utils.recover_muscledata(unassist_dataset,'std_norm_noload_muscles_activation')

# ******************************
# Energy
unassist_loadedvsnoload_metabolics = utils.reduction_calc(unassisted_energy_dataset['loaded_metabolics_energy'],unassisted_energy_dataset['noload_metabolics_energy'])
loaded_bi_metabolics = utils.reduction_calc(unassisted_energy_dataset['loaded_metabolics_energy'],assisted_energy_dataset['biarticular_ideal_loaded_metabolics_energy'])
noload_bi_metabolics = utils.reduction_calc(unassisted_energy_dataset['noload_metabolics_energy'],assisted_energy_dataset['biarticular_ideal_noload_metabolics_energy'])
loaded_mono_metabolics = utils.reduction_calc(unassisted_energy_dataset['loaded_metabolics_energy'],assisted_energy_dataset['monoarticular_ideal_loaded_metabolics_energy'])
noload_mono_metabolics = utils.reduction_calc(unassisted_energy_dataset['noload_metabolics_energy'],assisted_energy_dataset['monoarticular_ideal_noload_metabolics_energy'])
bi_loadedvsnoload_metabolics = utils.reduction_calc(assisted_energy_dataset['biarticular_ideal_loaded_metabolics_energy'],assisted_energy_dataset['biarticular_ideal_noload_metabolics_energy'])
mono_loadedvsnoload_metabolics = utils.reduction_calc(assisted_energy_dataset['monoarticular_ideal_loaded_metabolics_energy'],assisted_energy_dataset['monoarticular_ideal_noload_metabolics_energy'])

#####################################################################################
# saving profiles dataset
dataset =np.c_[mean_bi_loaded_hip_torque,std_bi_loaded_hip_torque,mean_bi_noload_hip_torque,std_bi_noload_hip_torque,\
           mean_bi_loaded_knee_torque,std_bi_loaded_knee_torque,mean_bi_noload_knee_torque,std_bi_noload_knee_torque,\
           mean_mono_loaded_hip_torque,std_mono_loaded_hip_torque,mean_mono_noload_hip_torque,std_mono_noload_hip_torque,\
           mean_mono_loaded_knee_torque,std_mono_loaded_knee_torque,mean_mono_noload_knee_torque,std_mono_noload_knee_torque,\
           mean_bi_loaded_hip_power,std_bi_loaded_hip_power,mean_bi_noload_hip_power,std_bi_noload_hip_power,\
           mean_bi_loaded_knee_power,std_bi_loaded_knee_power,mean_bi_noload_knee_power,std_bi_noload_knee_power,\
           mean_mono_loaded_hip_power,std_mono_loaded_hip_power,mean_mono_noload_hip_power,std_mono_noload_hip_power,\
           mean_mono_loaded_knee_power,std_mono_loaded_knee_power,mean_mono_noload_knee_power,std_mono_noload_knee_power]
headers = ['mean_bi_loaded_hip_torque','std_bi_loaded_hip_torque','mean_bi_noload_hip_torque','std_bi_noload_hip_torque',\
           'mean_bi_loaded_knee_torque','std_bi_loaded_knee_torque','mean_bi_noload_knee_torque','std_bi_noload_knee_torque',\
           'mean_mono_loaded_hip_torque','std_mono_loaded_hip_torque','mean_mono_noload_hip_torque','std_mono_noload_hip_torque',\
           'mean_mono_loaded_knee_torque','std_mono_loaded_knee_torque','mean_mono_noload_knee_torque','std_mono_noload_knee_torque',\
           'mean_bi_loaded_hip_power','std_bi_loaded_hip_power','mean_bi_noload_hip_power','std_bi_noload_hip_power',\
           'mean_bi_loaded_knee_power','std_bi_loaded_knee_power','mean_bi_noload_knee_power','std_bi_noload_knee_power',\
           'mean_mono_loaded_hip_power','std_mono_loaded_hip_power','mean_mono_noload_hip_power','std_mono_noload_hip_power',\
           'mean_mono_loaded_knee_power','std_mono_loaded_knee_power','mean_mono_noload_knee_power','std_mono_noload_knee_power']
with open(r'.\Data\Ideal\ideal_exos_profiles_dataset.csv', 'wb') as f:
  f.write(bytes(utils.listToString(headers)+'\n','UTF-8'))
  np.savetxt(f, dataset, fmt='%s', delimiter=",")
#####################################################################################
# hip joint moment plot dictionaries
bi_loaded_hip_musclesmoment_dic = {'pgc':gait_cycle,'avg':utils.smooth(mean_bi_loaded_hipmuscles_moment,3),'label':'loaded muscles',
                        'std':utils.smooth(std_bi_loaded_hipmuscles_moment,3),'avg_toeoff':loaded_mean_toe_off}
bi_noload_hip_musclesmoment_dic = {'pgc':gait_cycle,'avg':utils.smooth(mean_bi_noload_hipmuscles_moment,3),'label':'noload muscles',
                        'std':utils.smooth(std_bi_noload_hipmuscles_moment,3),'avg_toeoff':noload_mean_toe_off,'load':'noload'}
mono_loaded_hip_musclesmoment_dic = {'pgc':gait_cycle,'avg':utils.smooth(mean_mono_loaded_hipmuscles_moment,3),'label':'loaded muscles',
                        'std':utils.smooth(std_mono_loaded_hipmuscles_moment,3),'avg_toeoff':loaded_mean_toe_off}
mono_noload_hip_musclesmoment_dic = {'pgc':gait_cycle,'avg':utils.smooth(mean_mono_noload_hipmuscles_moment,3),'label':'noload muscles ',
                        'std':utils.smooth(std_mono_noload_hipmuscles_moment,3),'avg_toeoff':noload_mean_toe_off,'load':'noload'}
bi_loaded_hip_musclespower_dic = {'pgc':gait_cycle,'avg':utils.smooth(mean_bi_loaded_hipmuscles_power,3),'label':'loaded muscles',
                        'std':utils.smooth(std_bi_loaded_hipmuscles_power,3),'avg_toeoff':loaded_mean_toe_off}
bi_noload_hip_musclespower_dic = {'pgc':gait_cycle,'avg':utils.smooth(mean_bi_noload_hipmuscles_power,3),'label':'noload muscles',
                        'std':utils.smooth(std_bi_noload_hipmuscles_power,3),'avg_toeoff':noload_mean_toe_off,'load':'noload'}
mono_loaded_hip_musclespower_dic = {'pgc':gait_cycle,'avg':utils.smooth(mean_mono_loaded_hipmuscles_power,3),'label':'loaded muscles',
                        'std':utils.smooth(std_mono_loaded_hipmuscles_power,3),'avg_toeoff':loaded_mean_toe_off}
mono_noload_hip_musclespower_dic = {'pgc':gait_cycle,'avg':utils.smooth(mean_mono_noload_hipmuscles_power,3),'label':'noload muscles ',
                        'std':utils.smooth(std_mono_noload_hipmuscles_power,3),'avg_toeoff':noload_mean_toe_off,'load':'noload'}
unassist_loaded_hip_musclesmoment_dic = {'pgc':gait_cycle,'avg':utils.smooth(unassist_dataset['mean_norm_loaded_hipmuscles_moment'],3),'label':'loaded joint',
                        'std':utils.smooth(unassist_dataset['std_norm_loaded_hipmuscles_moment'],3),'avg_toeoff':loaded_mean_toe_off}
unassist_noload_hip_musclesmoment_dic = {'pgc':gait_cycle,'avg':utils.smooth(unassist_dataset['mean_norm_noload_hipmuscles_moment'],3),'label':'noload joint',
                        'std':utils.smooth(unassist_dataset['std_norm_noload_hipmuscles_moment'],3),'avg_toeoff':noload_mean_toe_off,'load':'noload'}

# knee joint moment plot dictionaries
bi_loaded_knee_musclesmoment_dic = {'pgc':gait_cycle,'avg':utils.smooth(mean_bi_loaded_kneemuscles_moment,3),'label':'loaded muscles',
                        'std':utils.smooth(std_bi_loaded_kneemuscles_moment,3),'avg_toeoff':loaded_mean_toe_off}
bi_noload_knee_musclesmoment_dic = {'pgc':gait_cycle,'avg':utils.smooth(mean_bi_noload_kneemuscles_moment,3),'label':'noload muscles',
                        'std':utils.smooth(std_bi_noload_kneemuscles_moment,3),'avg_toeoff':noload_mean_toe_off,'load':'noload'}
mono_loaded_knee_musclesmoment_dic = {'pgc':gait_cycle,'avg':utils.smooth(mean_mono_loaded_kneemuscles_moment,3),'label':'loaded muscles',
                        'std':utils.smooth(std_mono_loaded_kneemuscles_moment,3),'avg_toeoff':loaded_mean_toe_off}
mono_noload_knee_musclesmoment_dic = {'pgc':gait_cycle,'avg':utils.smooth(mean_mono_noload_kneemuscles_moment,3),'label':'noload muscles',
                        'std':utils.smooth(std_mono_noload_kneemuscles_moment,3),'avg_toeoff':noload_mean_toe_off,'load':'noload'}
bi_loaded_knee_musclespower_dic = {'pgc':gait_cycle,'avg':utils.smooth(mean_bi_loaded_kneemuscles_power,3),'label':'loaded muscles',
                        'std':utils.smooth(std_bi_loaded_kneemuscles_power,3),'avg_toeoff':loaded_mean_toe_off}
bi_noload_knee_musclespower_dic = {'pgc':gait_cycle,'avg':utils.smooth(mean_bi_noload_kneemuscles_power,3),'label':'noload muscles',
                        'std':utils.smooth(std_bi_noload_kneemuscles_power,3),'avg_toeoff':noload_mean_toe_off,'load':'noload'}
mono_loaded_knee_musclespower_dic = {'pgc':gait_cycle,'avg':utils.smooth(mean_mono_loaded_kneemuscles_power,3),'label':'loaded muscles',
                        'std':utils.smooth(std_mono_loaded_kneemuscles_power,3),'avg_toeoff':loaded_mean_toe_off}
mono_noload_knee_musclespower_dic = {'pgc':gait_cycle,'avg':utils.smooth(mean_mono_noload_kneemuscles_power,3),'label':'noload muscles',
                        'std':utils.smooth(std_mono_noload_kneemuscles_power,3),'avg_toeoff':noload_mean_toe_off,'load':'noload'}
unassist_loaded_knee_musclesmoment_dic = {'pgc':gait_cycle,'avg':utils.smooth(unassist_dataset['mean_norm_loaded_kneemuscles_moment'],3),'label':'loaded joint',
                        'std':utils.smooth(unassist_dataset['std_norm_loaded_kneemuscles_moment'],3),'avg_toeoff':loaded_mean_toe_off}
unassist_noload_knee_musclesmoment_dic = {'pgc':gait_cycle,'avg':utils.smooth(unassist_dataset['mean_norm_noload_kneemuscles_moment'],3),'label':'noload joint',
                        'std':utils.smooth(unassist_dataset['std_norm_noload_kneemuscles_moment'],3),'avg_toeoff':noload_mean_toe_off,'load':'noload'}

# muscles activation plot dictionaries
bi_loaded_muscles_activation_dic = {'pgc':gait_cycle,'avg':mean_bi_loaded_muscles_activation,'label':'Loaded',
                        'std':std_bi_loaded_muscles_activation,'avg_toeoff':loaded_mean_toe_off}
bi_noload_muscles_activation_dic = {'pgc':gait_cycle,'avg':mean_bi_noload_muscles_activation,'label':'Noload',
                        'std':std_bi_noload_muscles_activation,'avg_toeoff':noload_mean_toe_off,'load':'noload'}
mono_loaded_muscles_activation_dic = {'pgc':gait_cycle,'avg':mean_mono_loaded_muscles_activation,'label':'Loaded',
                        'std':std_mono_loaded_muscles_activation,'avg_toeoff':loaded_mean_toe_off}
mono_noload_muscles_activation_dic = {'pgc':gait_cycle,'avg':mean_mono_noload_muscles_activation,'label':'Noload',
                        'std':std_mono_noload_muscles_activation,'avg_toeoff':noload_mean_toe_off,'load':'noload'}
unassist_loaded_muscles_activation_dic = {'pgc':gait_cycle,'avg':mean_unassist_loaded_muscles_activation,'label':'Loaded',
                        'std':std_unassist_loaded_muscles_activation,'avg_toeoff':loaded_mean_toe_off}
unassist_noload_muscles_activation_dic = {'pgc':gait_cycle,'avg':mean_unassist_noload_muscles_activation,'label':'Noload',
                        'std':std_unassist_noload_muscles_activation,'avg_toeoff':noload_mean_toe_off,'load':'noload'}

# hip actuator torque plot dictionaries
bi_loaded_hip_torque_dic= {'pgc':gait_cycle,'avg':utils.smooth(mean_bi_loaded_hip_torque,3),'label':'loaded actuator',
                        'std':utils.smooth(std_bi_loaded_hip_torque,3),'avg_toeoff':loaded_mean_toe_off}
bi_noload_hip_torque_dic = {'pgc':gait_cycle,'avg':utils.smooth(mean_bi_noload_hip_torque,3),'label':'noload actuator',
                        'std':utils.smooth(std_bi_noload_hip_torque,3),'avg_toeoff':noload_mean_toe_off,'load':'noload'}
mono_loaded_hip_torque_dic = {'pgc':gait_cycle,'avg':utils.smooth(mean_mono_loaded_hip_torque,3),'label':'loaded actuator',
                        'std':utils.smooth(std_mono_loaded_hip_torque,3),'avg_toeoff':loaded_mean_toe_off}
mono_noload_hip_torque_dic = {'pgc':gait_cycle,'avg':utils.smooth(mean_mono_noload_hip_torque,3),'label':'noload actuator',
                        'std':utils.smooth(std_mono_noload_hip_torque,3),'avg_toeoff':noload_mean_toe_off,'load':'noload'}

# knee actuator torque plot dictionaries
bi_loaded_knee_torque_dic= {'pgc':gait_cycle,'avg':utils.smooth(mean_bi_loaded_knee_torque,3),'label':'loaded actuator',
                        'std':utils.smooth(std_bi_loaded_knee_torque,3),'avg_toeoff':loaded_mean_toe_off}
bi_noload_knee_torque_dic = {'pgc':gait_cycle,'avg':utils.smooth(mean_bi_noload_knee_torque,3),'label':'noload actuator',
                        'std':utils.smooth(std_bi_noload_knee_torque,3),'avg_toeoff':noload_mean_toe_off,'load':'noload'}
mono_loaded_knee_torque_dic = {'pgc':gait_cycle,'avg':utils.smooth(mean_mono_loaded_knee_torque,3),'label':'loaded actuator',
                        'std':utils.smooth(std_mono_loaded_knee_torque,3),'avg_toeoff':loaded_mean_toe_off}
mono_noload_knee_torque_dic = {'pgc':gait_cycle,'avg':utils.smooth(mean_mono_noload_knee_torque,3),'label':'noload actuator',
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
bi_loaded_hip_power_dic= {'pgc':gait_cycle,'avg':utils.smooth(mean_bi_loaded_hip_power,3),'label':'loaded actuator',
                        'std':utils.smooth(std_bi_loaded_hip_power,3),'avg_toeoff':loaded_mean_toe_off}
bi_noload_hip_power_dic = {'pgc':gait_cycle,'avg':utils.smooth(mean_bi_noload_hip_power,3),'label':'noload actuator',
                        'std':utils.smooth(std_bi_noload_hip_power,3),'avg_toeoff':noload_mean_toe_off,'load':'noload'}
mono_loaded_hip_power_dic = {'pgc':gait_cycle,'avg':utils.smooth(mean_mono_loaded_hip_power,3),'label':'loaded actuator',
                        'std':utils.smooth(std_mono_loaded_hip_power,3),'avg_toeoff':loaded_mean_toe_off}
mono_noload_hip_power_dic = {'pgc':gait_cycle,'avg':utils.smooth(mean_mono_noload_hip_power,3),'label':'noload actuator',
                        'std':utils.smooth(std_mono_noload_hip_power,3),'avg_toeoff':noload_mean_toe_off,'load':'noload'}

# knee actuator power plot dictionaries
bi_loaded_knee_power_dic= {'pgc':gait_cycle,'avg':utils.smooth(mean_bi_loaded_knee_power,3),'label':'loaded actuator',
                        'std':utils.smooth(std_bi_loaded_knee_power,3),'avg_toeoff':loaded_mean_toe_off}
bi_noload_knee_power_dic = {'pgc':gait_cycle,'avg':utils.smooth(mean_bi_noload_knee_power,3),'label':'noload actuator',
                        'std':utils.smooth(std_bi_noload_knee_power,3),'avg_toeoff':noload_mean_toe_off,'load':'noload'}
mono_loaded_knee_power_dic = {'pgc':gait_cycle,'avg':utils.smooth(mean_mono_loaded_knee_power,3),'label':'loaded actuator',
                        'std':utils.smooth(std_mono_loaded_knee_power,3),'avg_toeoff':loaded_mean_toe_off}
mono_noload_knee_power_dic = {'pgc':gait_cycle,'avg':utils.smooth(mean_mono_noload_knee_power,3),'label':'noload actuator',
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
bi_loaded_hip_speed_dic= {'pgc':gait_cycle,'avg':utils.smooth(mean_bi_loaded_hip_speed,3),'label':'loaded actuator',
                        'std':utils.smooth(std_bi_loaded_hip_speed,3),'avg_toeoff':loaded_mean_toe_off}
bi_noload_hip_speed_dic = {'pgc':gait_cycle,'avg':utils.smooth(mean_bi_noload_hip_speed,3),'label':'noload actuator',
                        'std':utils.smooth(std_bi_noload_hip_speed,3),'avg_toeoff':noload_mean_toe_off,'load':'noload'}
mono_loaded_hip_speed_dic = {'pgc':gait_cycle,'avg':utils.smooth(mean_mono_loaded_hip_speed,3),'label':'loaded actuator',
                        'std':utils.smooth(std_mono_loaded_hip_speed,3),'avg_toeoff':loaded_mean_toe_off}
mono_noload_hip_speed_dic = {'pgc':gait_cycle,'avg':utils.smooth(mean_mono_noload_hip_speed,3),'label':'noload actuator',
                        'std':utils.smooth(std_mono_noload_hip_speed,3),'avg_toeoff':noload_mean_toe_off,'load':'noload'}

# knee actuator speed plot dictionaries
bi_loaded_knee_speed_dic= {'pgc':gait_cycle,'avg':utils.smooth(mean_bi_loaded_knee_speed,3),'label':'loaded actuator',
                        'std':utils.smooth(std_bi_loaded_knee_speed,3),'avg_toeoff':loaded_mean_toe_off}
bi_noload_knee_speed_dic = {'pgc':gait_cycle,'avg':utils.smooth(mean_bi_noload_knee_speed,3),'label':'noload actuator',
                        'std':utils.smooth(std_bi_noload_knee_speed,3),'avg_toeoff':noload_mean_toe_off,'load':'noload'}
mono_loaded_knee_speed_dic = {'pgc':gait_cycle,'avg':utils.smooth(mean_mono_loaded_knee_speed,3),'label':'loaded actuator',
                        'std':utils.smooth(std_mono_loaded_knee_speed,3),'avg_toeoff':loaded_mean_toe_off}
mono_noload_knee_speed_dic = {'pgc':gait_cycle,'avg':utils.smooth(mean_mono_noload_knee_speed,3),'label':'noload actuator',
                        'std':utils.smooth(std_mono_noload_knee_speed,3),'avg_toeoff':noload_mean_toe_off,'load':'noload'}

#******************************************************************************************************************************
#******************************************************************************************************************************
# defualt color dictionary
default_color_dic = {
'color_1_list' : ['k','xkcd:irish green','k','xkcd:irish green'],
'color_2_list' : [mycolors['cyan blue'],mycolors['olympic blue'],mycolors['cyan blue'],mycolors['olympic blue']],
'color_3_list' : [mycolors['crimson red'],mycolors['french rose'],mycolors['crimson red'],mycolors['french rose']]
}
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
'plot_3_list' : [bi_loaded_hip_musclesmoment_dic,bi_noload_hip_musclesmoment_dic,\
                      mono_loaded_hip_musclesmoment_dic,mono_noload_hip_musclesmoment_dic],
'plot_2_list' : [bi_loaded_hip_torque_dic,bi_noload_hip_torque_dic,mono_loaded_hip_torque_dic,mono_noload_hip_torque_dic],
'plot_titles' : ['loaded biarticular hip joint','noload biarticular hip joint','loaded monoarticular hip joint','noload monoarticular hip joint']
}
# plot
fig = plt.figure(num='Loaded Hip Torque',figsize=(9.4, 6.8))
utils.plot_joint_muscle_exo(nrows=2,ncols=2,plot_dic=plot_dic,color_dic=default_color_dic,ylabel='hip flexion/extension (N-m/kg)')
fig.tight_layout()
fig.savefig('./Figures/Ideal/Exoskeletons_Hip_Torque.pdf',orientation='landscape',bbox_inches='tight')
plt.show()

# ***************************
# knee joint moment figure
# required dictionary
plot_dic={
'plot_1_list' : [unassist_loaded_knee_musclesmoment_dic,unassist_noload_knee_musclesmoment_dic,\
                     unassist_loaded_knee_musclesmoment_dic,unassist_noload_knee_musclesmoment_dic],
'plot_3_list' : [bi_loaded_knee_musclesmoment_dic,bi_noload_knee_musclesmoment_dic,\
                      mono_loaded_knee_musclesmoment_dic,mono_noload_knee_musclesmoment_dic],
'plot_2_list' : [bi_loaded_knee_torque_dic,bi_noload_knee_torque_dic,mono_loaded_knee_torque_dic,mono_noload_knee_torque_dic],
'plot_titles' : ['loaded biarticular knee joint','noload biarticular knee joint','loaded monoarticular knee joint','noload monoarticular knee joint']
}
# plot
fig = plt.figure(num='Loaded Knee Torque',figsize=(9.4, 6.8))
utils.plot_joint_muscle_exo(nrows=2,ncols=2,plot_dic=plot_dic,color_dic=default_color_dic,ylabel='knee flexion/extension (N-m/kg)')
fig.tight_layout()
plt.show()
fig.savefig('./Figures/Ideal/Exoskeletons_Knee_Torque.pdf',orientation='landscape',bbox_inches='tight')

#******************************************************************************************************************************
# hip joint power figure
# required dictionary
plot_dic={
'plot_1_list' : [loaded_hip_power_dic,noload_hip_power_dic,\
                     loaded_hip_power_dic,noload_hip_power_dic],
'plot_3_list' : [bi_loaded_hip_musclespower_dic,bi_noload_hip_musclespower_dic,\
                      mono_loaded_hip_musclespower_dic,mono_noload_hip_musclespower_dic],
'plot_2_list' : [bi_loaded_hip_power_dic,bi_noload_hip_power_dic,mono_loaded_hip_power_dic,mono_noload_hip_power_dic],
'plot_titles' : ['loaded biarticular hip joint','noload biarticular hip joint','loaded monoarticular hip joint','noload monoarticular hip joint']
}
# plot
fig = plt.figure(num='Loaded Hip Power',figsize=(9.4, 6.8))
utils.plot_joint_muscle_exo(nrows=2,ncols=2,plot_dic=plot_dic,color_dic=default_color_dic,\
                            thirdplot=True,ylabel='hip flexion/extension (W/kg)',y_ticks=np.arange(-4,5,2))
fig.tight_layout()
fig.savefig('./Figures/Ideal/Exoskeletons_Hip_Power.pdf',orientation='landscape',bbox_inches='tight')
plt.show()

# ***************************
# knee joint power figure
# required dictionary
plot_dic={
'plot_1_list' : [loaded_knee_power_dic,noload_knee_power_dic,\
                     loaded_knee_power_dic,noload_knee_power_dic],
'plot_3_list' : [bi_loaded_knee_musclespower_dic,bi_noload_knee_musclespower_dic,\
                      mono_loaded_knee_musclespower_dic,mono_noload_knee_musclespower_dic],
'plot_2_list' : [bi_loaded_knee_power_dic,bi_noload_knee_power_dic,mono_loaded_knee_power_dic,mono_noload_knee_power_dic],
'plot_titles' : ['loaded biarticular knee joint','noload biarticular knee joint','loaded monoarticular knee joint','noload monoarticular knee joint']
}
# 
fig = plt.figure(num='Loaded Knee Power',figsize=(9.4, 6.8))
utils.plot_joint_muscle_exo(nrows=2,ncols=2,plot_dic=plot_dic,color_dic=default_color_dic,\
                            thirdplot=True,ylabel='knee flexion/extension (W/kg)',y_ticks=np.arange(-4,6,2))
fig.tight_layout()
plt.show()
fig.savefig('./Figures/Ideal/Exoskeletons_Knee_Power.pdf',orientation='landscape',bbox_inches='tight')

#******************************************************************************************************************************
# hip joint speed figure
# required dictionary
plot_dic={
'plot_1_list' : [loaded_hip_speed_dic,noload_hip_speed_dic,\
                     loaded_hip_speed_dic,noload_hip_speed_dic],
'plot_2_list' : [bi_loaded_hip_speed_dic,bi_noload_hip_speed_dic,mono_loaded_hip_speed_dic,mono_noload_hip_speed_dic],
'plot_titles' : ['loaded biarticular hip joint','noload biarticular hip joint','loaded monoarticular hip joint','noload monoarticular hip joint']
}
# plot
fig = plt.figure(num='Loaded Hip Speed',figsize=(9.4, 6.8))
utils.plot_joint_muscle_exo(nrows=2,ncols=2,plot_dic=plot_dic,color_dic=default_color_dic,\
                            thirdplot=False,ylabel='hip flexion/extension (rad/s)',y_ticks=np.arange(-6,4,2))
fig.tight_layout()
fig.savefig('./Figures/Ideal/Exoskeletons_Hip_Speed.pdf',orientation='landscape',bbox_inches='tight')
plt.show()

# ***************************
# knee joint speed figure
# required dictionary
plot_dic={
'plot_1_list' : [loaded_knee_speed_dic,noload_knee_speed_dic,\
                     loaded_knee_speed_dic,noload_knee_speed_dic],
'plot_2_list' : [bi_loaded_knee_speed_dic,bi_noload_knee_speed_dic,mono_loaded_knee_speed_dic,mono_noload_knee_speed_dic],
'plot_titles' : ['loaded biarticular knee joint','noload biarticular knee joint','loaded monoarticular knee joint','noload monoarticular knee joint']
}
# plot
fig = plt.figure(num='Loaded Knee Speed',figsize=(9.4, 6.8))
utils.plot_joint_muscle_exo(nrows=2,ncols=2,plot_dic=plot_dic,color_dic=default_color_dic,\
                            thirdplot=False,ylabel='knee flexion/extension (rad/s)',y_ticks=np.arange(-6,11,3))
fig.tight_layout()
plt.show()
fig.savefig('./Figures/Ideal/Exoskeletons_Knee_Speed.pdf',orientation='landscape',bbox_inches='tight')

##################################################################################################
# Paper figure
# ***************************
# moment figure
# required dictionary
plot_dic={
'plot_1_list' : [unassist_loaded_hip_musclesmoment_dic,unassist_loaded_knee_musclesmoment_dic,\
                 unassist_loaded_hip_musclesmoment_dic,unassist_noload_hip_musclesmoment_dic,\
                 unassist_noload_knee_musclesmoment_dic,unassist_noload_hip_musclesmoment_dic,\
                 unassist_loaded_knee_musclesmoment_dic,unassist_noload_knee_musclesmoment_dic],
'plot_3_list' : [bi_loaded_hip_musclesmoment_dic,bi_loaded_knee_musclesmoment_dic,mono_loaded_hip_musclesmoment_dic,\
                 bi_noload_hip_musclesmoment_dic,bi_noload_knee_musclesmoment_dic,mono_noload_hip_musclesmoment_dic,\
                 mono_loaded_knee_musclesmoment_dic,mono_noload_knee_musclesmoment_dic],
'plot_2_list' : [bi_loaded_hip_torque_dic,bi_loaded_knee_torque_dic,mono_loaded_hip_torque_dic,\
                 bi_noload_hip_torque_dic,bi_noload_knee_torque_dic,mono_noload_hip_torque_dic,\
                 mono_loaded_knee_torque_dic,mono_noload_knee_torque_dic],
'plot_titles' : ['loaded biarticular hip joint','loaded biarticular knee joint','loaded monoarticular hip joint',\
                 'noload biarticular hip joint','noload biarticular knee joint','noload monoarticular hip joint',\
                 'loaded monoarticular knee joint','noload monoarticular knee joint']
}
default_color_dic = {
'color_1_list' : ['k','k','k','xkcd:irish green','xkcd:irish green','xkcd:irish green','k','xkcd:irish green'],
'color_2_list' : [mycolors['cyan blue'],mycolors['cyan blue'],mycolors['cyan blue'],mycolors['olympic blue'],\
                  mycolors['olympic blue'],mycolors['olympic blue'],mycolors['cyan blue'],mycolors['olympic blue']],
'color_3_list' : [mycolors['crimson red'],mycolors['crimson red'],mycolors['crimson red'],mycolors['french rose'],\
                  mycolors['french rose'],mycolors['french rose'],mycolors['crimson red'],mycolors['french rose']]
}
# plot
fig = plt.figure(num='Moment Figure',figsize=(12.8, 9.6))
utils.plot_joint_muscle_exo(nrows=3,ncols=3,nplots=8,plot_dic=plot_dic,color_dic=default_color_dic,legend_loc=[0,3],\
                            subplot_legend=True,fig=fig,ylabel='flexion/extension\n moment (N-m/kg)')
fig.tight_layout(h_pad=-1.5, w_pad=-1.5)
fig.subplots_adjust(top=0.99, bottom=0.075, left=0.100, right=0.975,hspace=0.2,wspace=0.15)
fig.savefig('./Figures/Ideal/PaperFigure_Exoskeletons_Torque.pdf',orientation='landscape',bbox_inches='tight')
plt.show()

# ***************************
# power figure
# required dictionary
plot_dic={
'plot_1_list' : [loaded_hip_power_dic,loaded_knee_power_dic,\
                 loaded_hip_power_dic,noload_hip_power_dic,\
                 noload_knee_power_dic,noload_hip_power_dic,\
                 loaded_knee_power_dic,noload_knee_power_dic],
'plot_3_list' : [bi_loaded_hip_musclespower_dic,bi_loaded_knee_musclespower_dic,mono_loaded_hip_musclespower_dic,\
                 bi_noload_hip_musclespower_dic,bi_noload_knee_musclespower_dic,mono_noload_hip_musclespower_dic,\
                 mono_loaded_knee_musclespower_dic,mono_noload_knee_musclespower_dic],
'plot_2_list' : [bi_loaded_hip_power_dic,bi_loaded_knee_power_dic,mono_loaded_hip_power_dic,\
                 bi_noload_hip_power_dic,bi_noload_knee_power_dic,mono_noload_hip_power_dic,\
                 mono_loaded_knee_power_dic,mono_noload_knee_power_dic],
'plot_titles' : ['loaded biarticular hip joint','loaded biarticular knee joint','loaded monoarticular hip joint',\
                 'noload biarticular hip joint','noload biarticular knee joint','noload monoarticular hip joint',\
                 'loaded monoarticular knee joint','noload monoarticular knee joint']
}
default_color_dic = {
'color_1_list' : ['k','k','k','xkcd:irish green','xkcd:irish green','xkcd:irish green','k','xkcd:irish green'],
'color_2_list' : [mycolors['cyan blue'],mycolors['cyan blue'],mycolors['cyan blue'],mycolors['olympic blue'],\
                  mycolors['olympic blue'],mycolors['olympic blue'],mycolors['cyan blue'],mycolors['olympic blue']],
'color_3_list' : [mycolors['crimson red'],mycolors['crimson red'],mycolors['crimson red'],mycolors['french rose'],\
                  mycolors['french rose'],mycolors['french rose'],mycolors['crimson red'],mycolors['french rose']]
}
# plot
fig = plt.figure(num='Power Figure',figsize=(12.8, 9.6))
utils.plot_joint_muscle_exo(nrows=3,ncols=3,nplots=8,plot_dic=plot_dic,color_dic=default_color_dic,legend_loc=[0,3],\
                            subplot_legend=True,fig=fig,ylabel='flexion/extension\n moment (W/kg)')
fig.tight_layout(h_pad=-1.5, w_pad=-1.5)
fig.subplots_adjust(top=0.99, bottom=0.075, left=0.100, right=0.975,hspace=0.2,wspace=0.15)
fig.savefig('./Figures/Ideal/PaperFigure_Exoskeletons_Power.pdf',orientation='landscape',bbox_inches='tight')
plt.show()

# ***************************
# speed figure
# required dictionary
plot_dic={
'plot_1_list' : [loaded_hip_speed_dic,loaded_knee_speed_dic,loaded_hip_speed_dic,loaded_knee_speed_dic,\
                 noload_hip_speed_dic,noload_knee_speed_dic,noload_hip_speed_dic,noload_knee_speed_dic],
'plot_2_list' : [bi_loaded_hip_speed_dic,bi_loaded_knee_speed_dic,mono_loaded_hip_speed_dic,mono_loaded_knee_speed_dic,\
                 bi_noload_hip_speed_dic,bi_noload_knee_speed_dic,mono_noload_hip_speed_dic,mono_noload_knee_speed_dic],
'plot_titles' : ['loaded biarticular\n hip joint','loaded biarticular\n knee joint','loaded monoarticular\n hip joint','loaded monoarticular\n knee joint',\
                 'noload biarticular\n hip joint','noload biarticular\n knee joint','noload monoarticular\n hip joint','noload monoarticular\n knee joint']
}
default_color_dic = {
'color_1_list' : ['k','k','k','k','xkcd:irish green','xkcd:irish green','xkcd:irish green','xkcd:irish green'],
'color_2_list' : [mycolors['cyan blue'],mycolors['cyan blue'],mycolors['cyan blue'],mycolors['cyan blue'],\
                  mycolors['olympic blue'],mycolors['olympic blue'],mycolors['olympic blue'],mycolors['olympic blue']],
'color_3_list' : [mycolors['crimson red'],mycolors['crimson red'],mycolors['crimson red'],mycolors['crimson red'],\
                  mycolors['french rose'],mycolors['french rose'],mycolors['french rose'],mycolors['french rose']]
}
# plot
fig = plt.figure(num='Loaded Hip Torque',figsize=(12.8,6.6))
utils.plot_joint_muscle_exo(nrows=2,ncols=4,nplots=8,plot_dic=plot_dic,color_dic=default_color_dic,legend_loc=[0,4],\
                            thirdplot=False,subplot_legend=False,fig=fig,ylabel='flexion/extension\n speed (rad/s)',y_ticks=[-6,-4,-2,0,2,4,6,8])
fig.tight_layout(h_pad=-1, w_pad=-1.5)
fig.subplots_adjust(top=0.98, bottom=0.075, left=0.100, right=0.975,hspace=0.25,wspace=0.15)
fig.savefig('./Figures/Ideal/PaperFigure_Exoskeletons_Speed.pdf',orientation='landscape',bbox_inches='tight')
plt.show()
