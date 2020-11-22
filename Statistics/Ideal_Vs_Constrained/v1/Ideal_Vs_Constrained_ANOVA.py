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
from mpl_toolkits.mplot3d import Axes3D
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
ideal_dataset = utils.csv2numpy('./Data/Ideal/ideal_exos_dataset.csv') 
rra_dataset = utils.csv2numpy('./Data/RRA/rra_final_data.csv') 
unassist_dataset = utils.csv2numpy('./Data/Unassist/unassist_final_data.csv') 
# pareto exo torque dataset
directory = './Data/Pareto/*_torque.csv'
files = enumerate(glob.iglob(directory), 1)
exo_torque_dataset = {pathlib.PurePath(f[1]).stem: np.loadtxt(f[1], delimiter=',') for f in files}
# pareto exo power dataset
directory = './Data/Pareto/*_power.csv'
files = enumerate(glob.iglob(directory), 1)
exo_power_dataset = {pathlib.PurePath(f[1]).stem: np.loadtxt(f[1], delimiter=',') for f in files}
# pareto exo energy dataset
directory = './Data/Pareto/*_energy.csv'
files = enumerate(glob.iglob(directory), 1)
assisted_energy_dataset = {pathlib.PurePath(f[1]).stem: np.loadtxt(f[1], delimiter=',') for f in files}
# pareto exo power dataset
directory = './Data/Pareto/*_power.csv'
files = enumerate(glob.iglob(directory), 1)
assisted_power_dataset = {pathlib.PurePath(f[1]).stem: np.loadtxt(f[1], delimiter=',') for f in files}
# unassist energy dataset
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
# metabolics cost reduction percents
bi_loaded_metabolics_percent = np.reshape(assisted_energy_dataset['biarticular_pareto_load_metabolics_energy'],(25,21),order='F')
bi_noload_metabolics_percent = np.reshape(assisted_energy_dataset['biarticular_pareto_noload_metabolics_energy'],(25,21),order='F')
mono_loaded_metabolics_percent = np.reshape(assisted_energy_dataset['monoarticular_pareto_load_metabolics_energy'],(25,21),order='F')
mono_noload_metabolics_percent = np.reshape(assisted_energy_dataset['monoarticular_pareto_noload_metabolics_energy'],(25,21),order='F')
# actuators energy
# loaded bi
bi_loaded_hip_energy = np.reshape(assisted_energy_dataset['biarticular_pareto_load_hipactuator_energy'],(25,21),order='F')
bi_loaded_knee_energy = np.reshape(assisted_energy_dataset['biarticular_pareto_load_kneeactuator_energy'],(25,21),order='F')
bi_loaded_hip_max_power = np.reshape(assisted_power_dataset['biarticular_pareto_load_hip_max_power'],(25,21),order='F')
bi_loaded_knee_max_power = np.reshape(assisted_power_dataset['biarticular_pareto_load_knee_max_power'],(25,21),order='F')
bi_loaded_energy = np.reshape(assisted_energy_dataset['biarticular_pareto_load_hipactuator_energy']+assisted_energy_dataset['biarticular_pareto_load_kneeactuator_energy'],(25,21),order='F')
# noload bi
bi_noload_hip_energy = np.reshape(assisted_energy_dataset['biarticular_pareto_noload_hipactuator_energy'],(25,21),order='F')
bi_noload_knee_energy = np.reshape(assisted_energy_dataset['biarticular_pareto_noload_kneeactuator_energy'],(25,21),order='F')
bi_noload_hip_max_power = np.reshape(assisted_power_dataset['biarticular_pareto_noload_hip_max_power'],(25,21),order='F')
bi_noload_knee_max_power = np.reshape(assisted_power_dataset['biarticular_pareto_noload_knee_max_power'],(25,21),order='F')
bi_noload_energy = np.reshape(assisted_energy_dataset['biarticular_pareto_noload_hipactuator_energy']+assisted_energy_dataset['biarticular_pareto_noload_kneeactuator_energy'],(25,21),order='F')
# loaded mono
mono_loaded_hip_energy = np.reshape(assisted_energy_dataset['monoarticular_pareto_load_hipactuator_energy'],(25,21),order='F')
mono_loaded_knee_energy = np.reshape(assisted_energy_dataset['monoarticular_pareto_load_kneeactuator_energy'],(25,21),order='F')
mono_loaded_hip_max_power = np.reshape(assisted_power_dataset['monoarticular_pareto_load_hip_max_power'],(25,21),order='F')
mono_loaded_knee_max_power = np.reshape(assisted_power_dataset['monoarticular_pareto_load_knee_max_power'],(25,21),order='F')
mono_loaded_energy = np.reshape(assisted_energy_dataset['monoarticular_pareto_load_hipactuator_energy']+assisted_energy_dataset['biarticular_pareto_load_kneeactuator_energy'],(25,21),order='F')
# noload mono
mono_noload_hip_energy = np.reshape(assisted_energy_dataset['monoarticular_pareto_noload_hipactuator_energy'],(25,21),order='F')
mono_noload_knee_energy = np.reshape(assisted_energy_dataset['monoarticular_pareto_noload_kneeactuator_energy'],(25,21),order='F')
mono_noload_hip_max_power = np.reshape(assisted_power_dataset['monoarticular_pareto_noload_hip_max_power'],(25,21),order='F')
mono_noload_knee_max_power = np.reshape(assisted_power_dataset['monoarticular_pareto_noload_knee_max_power'],(25,21),order='F')
mono_noload_energy = np.reshape(assisted_energy_dataset['monoarticular_pareto_noload_hipactuator_energy']+assisted_energy_dataset['biarticular_pareto_noload_kneeactuator_energy'],(25,21),order='F')
#####################################################################################
# metabolic rate and energy of "Aa" for all conditions
# metabolics cost reduction percents
bi_loaded_Aa_metabolics_percent = np.reshape(bi_loaded_metabolics_percent[16,:],(7,3))
bi_noload_Aa_metabolics_percent = np.reshape(bi_noload_metabolics_percent[16,:],(7,3))
mono_loaded_Aa_metabolics_percent = np.reshape(mono_loaded_metabolics_percent[9,:],(7,3))
mono_noload_Aa_metabolics_percent = np.reshape(mono_noload_metabolics_percent[12,:],(7,3))
# actuators energy
# loaded bi
bi_loaded_Aa_hip_energy = np.reshape(bi_loaded_hip_energy[16,:],(7,3))

bi_loaded_Aa_knee_energy = np.reshape(bi_loaded_knee_energy[16,:],(7,3))
bi_loaded_Aa_hip_power = np.reshape(bi_loaded_hip_max_power[16,:],(7,3))
bi_loaded_Aa_knee_power = np.reshape(bi_loaded_knee_max_power[16,:],(7,3))
# noload bi
bi_noload_Aa_hip_energy = np.reshape(bi_noload_hip_energy[16,:],(7,3))
bi_noload_Aa_knee_energy = np.reshape(bi_noload_knee_energy[16,:],(7,3))
bi_noload_Aa_hip_power = np.reshape(bi_noload_hip_max_power[16,:],(7,3))
bi_noload_Aa_knee_power = np.reshape(bi_noload_knee_max_power[16,:],(7,3))
# loaded mono
mono_loaded_Aa_hip_energy = np.reshape(mono_loaded_hip_energy[9,:],(7,3))
mono_loaded_Aa_knee_energy = np.reshape(mono_loaded_knee_energy[9,:],(7,3))
mono_loaded_Aa_hip_power = np.reshape(mono_loaded_hip_max_power[9,:],(7,3))
mono_loaded_Aa_knee_power = np.reshape(mono_loaded_knee_max_power[9,:],(7,3))
# noload mono
mono_noload_Aa_hip_energy = np.reshape(mono_noload_hip_energy[12,:],(7,3))
mono_noload_Aa_knee_energy = np.reshape(mono_noload_knee_energy[12,:],(7,3))
mono_noload_Aa_hip_power = np.reshape(mono_noload_hip_max_power[12,:],(7,3))
mono_noload_Aa_knee_power = np.reshape(mono_noload_knee_max_power[12,:],(7,3))
#####################################################################################
# writing data to csv for statistical analyses
# general columns 
subjects = np.array(['subject05','subject07','subject09','subject10','subject11','subject12','subject14'])
loaded_condition_col = np.repeat(np.array('loaded'),7*2)
noload_condition_col = np.repeat(np.array('noload'),7*2)
biarticular_device_col = np.repeat(np.array('biarticular'),7)
monoarticular_device_col = np.repeat(np.array('monoarticular'),7)
constrained_condition_col = np.repeat(np.array('constrained'),7*4)

# establishing dataset for metabolic rate
headers = ['subjects','load condition','constraint type','device type','metabolic rate 01','metabolic rate 02','metabolic rate 03']
subject_col = np.tile(subjects,4)
general_assistance_col = np.concatenate((monoarticular_device_col,biarticular_device_col,monoarticular_device_col,biarticular_device_col),axis=0)
general_load_condition_col = np.concatenate((loaded_condition_col,noload_condition_col),axis=0)
metabolic_rate_data = np.concatenate((mono_loaded_Aa_metabolics_percent,bi_loaded_Aa_metabolics_percent,mono_noload_Aa_metabolics_percent,bi_noload_Aa_metabolics_percent),axis=0)
final_dataset = np.column_stack([general_load_condition_col,constrained_condition_col,general_assistance_col,metabolic_rate_data])
final_dataset = np.column_stack([subject_col,final_dataset])
with open(r'.\Statistics\Ideal_Vs_Constrained\MetabolicRate_Dataset.csv', 'wb') as f:
  f.write(bytes(utils.listToString(headers)+'\n','UTF-8'))
  np.savetxt(f, final_dataset, fmt='%s', delimiter=",")

#****************************************************************
#establishing dataset for assistive actuators maximum power 
headers = ['subjects','load condition','constraints','device type','max power 01','max power 02','max power 03']
loaded_condition_col = np.repeat(np.array('loaded'),7*2)
noload_condition_col = np.repeat(np.array('noload'),7*2)
biarticular_device_col = np.repeat(np.array('biarticular'),7)
monoarticular_device_col = np.repeat(np.array('monoarticular'),7)
constrained_condition_col = np.repeat(np.array('constrained'),7*4)
# hip_actuator_col = np.repeat(np.array('hip actuator'),7)
# knee_actuator_col = np.repeat(np.array('knee actuator'),7)
#general_actuator_col = np.concatenate((hip_actuator_col,knee_actuator_col,hip_actuator_col,knee_actuator_col,hip_actuator_col,knee_actuator_col,hip_actuator_col,knee_actuator_col),axis=0)

subject_col = np.tile(subjects,4)
general_load_condition_col = np.concatenate((loaded_condition_col,noload_condition_col),axis=0)
general_device_col = np.concatenate((biarticular_device_col,monoarticular_device_col,biarticular_device_col,monoarticular_device_col),axis=0)
assistive_actuators_max_power_data = np.concatenate((bi_loaded_Aa_hip_power+bi_loaded_Aa_knee_power,mono_loaded_Aa_hip_power+mono_loaded_Aa_knee_power,\
                                                     bi_noload_Aa_hip_power+bi_noload_Aa_knee_power,mono_noload_Aa_hip_power+mono_noload_Aa_knee_power),axis=0)
final_dataset = np.column_stack([general_load_condition_col,constrained_condition_col,general_device_col,assistive_actuators_max_power_data])
final_dataset = np.column_stack([subject_col,final_dataset])
with open(r'.\Statistics\Ideal_Vs_Constrained\ActuatorsMaxPower_Dataset.csv', 'wb') as f:
  f.write(bytes(utils.listToString(headers)+'\n','UTF-8'))
  np.savetxt(f, final_dataset, fmt='%s', delimiter=",")

#****************************************************************
#establishing dataset for assistive actuators average total power 
headers = ['subjects','load condition','constraints','device type','avg power 01','avg power 02','avg power 03']
loaded_condition_col = np.repeat(np.array('loaded'),7*2)
noload_condition_col = np.repeat(np.array('noload'),7*2)
biarticular_device_col = np.repeat(np.array('biarticular'),7)
monoarticular_device_col = np.repeat(np.array('monoarticular'),7)
constrained_condition_col = np.repeat(np.array('constrained'),7*4)
# hip_actuator_col = np.repeat(np.array('hip actuator'),7)
# knee_actuator_col = np.repeat(np.array('knee actuator'),7)
#general_actuator_col = np.concatenate((hip_actuator_col,knee_actuator_col,hip_actuator_col,knee_actuator_col,hip_actuator_col,knee_actuator_col,hip_actuator_col,knee_actuator_col),axis=0)

subject_col = np.tile(subjects,4)
general_load_condition_col = np.concatenate((loaded_condition_col,noload_condition_col),axis=0)
general_device_col = np.concatenate((biarticular_device_col,monoarticular_device_col,biarticular_device_col,monoarticular_device_col),axis=0)
assistive_actuators_avg_totalpower_data = np.concatenate((bi_loaded_Aa_hip_energy+bi_loaded_Aa_knee_energy,mono_loaded_Aa_hip_energy+mono_loaded_Aa_knee_energy,\
                                                          bi_noload_Aa_hip_energy+bi_noload_Aa_knee_energy,mono_noload_Aa_hip_energy+mono_noload_Aa_knee_energy),axis=0)
final_dataset = np.column_stack([general_load_condition_col,constrained_condition_col,general_device_col,assistive_actuators_avg_totalpower_data])
final_dataset = np.column_stack([subject_col,final_dataset])
with open(r'.\Statistics\Ideal_Vs_Constrained\ActuatorsAvgPower_Dataset.csv', 'wb') as f:
  f.write(bytes(utils.listToString(headers)+'\n','UTF-8'))
  np.savetxt(f, final_dataset, fmt='%s', delimiter=",")
