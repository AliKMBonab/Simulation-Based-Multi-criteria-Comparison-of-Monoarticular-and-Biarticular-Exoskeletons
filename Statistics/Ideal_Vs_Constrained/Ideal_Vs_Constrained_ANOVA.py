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
bi_loaded_energy = np.reshape(assisted_energy_dataset['biarticular_pareto_load_hipactuator_energy']+assisted_energy_dataset['biarticular_pareto_load_kneeactuator_energy'],(25,21),order='F')
# noload bi
bi_noload_hip_energy = np.reshape(assisted_energy_dataset['biarticular_pareto_noload_hipactuator_energy'],(25,21),order='F')
bi_noload_knee_energy = np.reshape(assisted_energy_dataset['biarticular_pareto_noload_kneeactuator_energy'],(25,21),order='F')
bi_noload_energy = np.reshape(assisted_energy_dataset['biarticular_pareto_noload_hipactuator_energy']+assisted_energy_dataset['biarticular_pareto_noload_kneeactuator_energy'],(25,21),order='F')
# loaded mono
mono_loaded_hip_energy = np.reshape(assisted_energy_dataset['monoarticular_pareto_load_hipactuator_energy'],(25,21),order='F')
mono_loaded_knee_energy = np.reshape(assisted_energy_dataset['monoarticular_pareto_load_kneeactuator_energy'],(25,21),order='F')
mono_loaded_energy = np.reshape(assisted_energy_dataset['monoarticular_pareto_load_hipactuator_energy']+assisted_energy_dataset['biarticular_pareto_load_kneeactuator_energy'],(25,21),order='F')
# noload mono
mono_noload_hip_energy = np.reshape(assisted_energy_dataset['monoarticular_pareto_noload_hipactuator_energy'],(25,21),order='F')
mono_noload_knee_energy = np.reshape(assisted_energy_dataset['monoarticular_pareto_noload_kneeactuator_energy'],(25,21),order='F')
mono_noload_energy = np.reshape(assisted_energy_dataset['monoarticular_pareto_noload_hipactuator_energy']+assisted_energy_dataset['biarticular_pareto_noload_kneeactuator_energy'],(25,21),order='F')
#####################################################################################
# metabolic rate and energy of "Aa" for all conditions
# metabolics cost reduction percents
bi_loaded_Aa_metabolics_percent = np.reshape(bi_loaded_metabolics_percent[0,:],(7,3))
bi_noload_Aa_metabolics_percent = np.reshape(bi_noload_metabolics_percent[0,:],(7,3))
mono_loaded_Aa_metabolics_percent = np.reshape(mono_loaded_metabolics_percent[0,:],(7,3))
mono_noload_Aa_metabolics_percent = np.reshape(mono_noload_metabolics_percent[0,:],(7,3))
# actuators energy
# loaded bi
bi_loaded_Aa_hip_energy = np.reshape(bi_loaded_hip_energy[0,:],(7,3))
bi_loaded_Aa_knee_energy = np.reshape(bi_loaded_knee_energy[0,:],(7,3))
bi_loaded_Aa_energy = np.reshape(bi_loaded_energy[0,:],(7,3))
# noload bi
bi_noload_Aa_hip_energy = np.reshape(bi_noload_hip_energy[0,:],(7,3))
bi_noload_Aa_knee_energy = np.reshape(bi_noload_knee_energy[0,:],(7,3))
bi_noload_Aa_energy = np.reshape(bi_noload_energy[0,:],(7,3))
# loaded mono
mono_loaded_Aa_hip_energy = np.reshape(mono_loaded_hip_energy[0,:],(7,3))
mono_loaded_Aa_knee_energy = np.reshape(mono_loaded_knee_energy[0,:],(7,3))
mono_loaded_Aa_energy = np.reshape(mono_loaded_energy[0,:],(7,3))
# noload mono
mono_noload_Aa_hip_energy = np.reshape(mono_noload_hip_energy[0,:],(7,3))
mono_noload_Aa_knee_energy = np.reshape(mono_noload_knee_energy[0,:],(7,3))
mono_noload_Aa_energy = np.reshape(mono_noload_energy[0,:],(7,3))
#####################################################################################
# writing data to csv for statistical analyses
# general columns 
subjects = np.array(['subject05','subject07','subject09','subject10','subject11','subject12','subject14'])
loaded_biarticular_col = np.repeat(np.array('loaded constrained biarticular'),7)
loaded_monoarticular_col = np.repeat(np.array('loaded constrained monoarticular'),7)
noload_biarticular_col = np.repeat(np.array('noload constrained biarticular'),7)
noload_monoarticular_col = np.repeat(np.array('noload constrained monoarticular'),7)

# establishing dataset for metabolic rate
headers = ['subjects','assistance','metabolic rate 01','metabolic rate 02','metabolic rate 03']
subject_col = np.tile(subjects,4)
assistance_col = np.concatenate((loaded_monoarticular_col,loaded_biarticular_col,noload_monoarticular_col,noload_biarticular_col),axis=0)
metabolic_rate_data = np.concatenate((mono_loaded_Aa_metabolics_percent,bi_loaded_Aa_metabolics_percent,mono_noload_Aa_metabolics_percent,bi_noload_Aa_metabolics_percent),axis=0)
final_dataset = np.column_stack([assistance_col,metabolic_rate_data])
final_dataset = np.column_stack([subject_col,final_dataset])
with open(r'.\Statistics\Ideal_Vs_Constrained\MetabolicRate_Dataset.csv', 'wb') as f:
  f.write(bytes(utils.listToString(headers)+'\n','UTF-8'))
  np.savetxt(f, final_dataset, fmt='%s', delimiter=",")

#****************************************************************
#establishing dataset for assistive actuators average total power 
headers = ['subjects','assistive actuator','avg total power 01','avg total power 02','avg total power 03']
loaded_biarticular_hip_col = np.repeat(np.array('loaded constrained biarticular hip actuator'),7)
loaded_monoarticular_hip_col = np.repeat(np.array('loaded constrained monoarticular hip actuator'),7)
loaded_biarticular_knee_col = np.repeat(np.array('loaded constrained biarticular knee actuator'),7)
loaded_monoarticular_knee_col = np.repeat(np.array('loaded constrained monoarticular knee actuator'),7)
noload_biarticular_hip_col = np.repeat(np.array('noload constrained biarticular hip actuator'),7)
noload_monoarticular_hip_col = np.repeat(np.array('noload constrained monoarticular hip actuator'),7)
noload_biarticular_knee_col = np.repeat(np.array('noload constrained biarticular knee actuator'),7)
noload_monoarticular_knee_col = np.repeat(np.array('noload constrained monoarticular knee actuator'),7)
subject_col = np.tile(subjects,8)
assistive_actuators_col = np.concatenate((loaded_biarticular_hip_col,loaded_biarticular_knee_col,loaded_monoarticular_hip_col,loaded_monoarticular_knee_col,
                                          noload_biarticular_hip_col,noload_biarticular_knee_col,noload_monoarticular_hip_col,noload_monoarticular_knee_col),axis=0)
assistive_actuators_avg_totalpower_data = np.concatenate((bi_loaded_Aa_hip_energy,bi_loaded_Aa_knee_energy,mono_loaded_Aa_hip_energy,mono_loaded_Aa_knee_energy,\
                                                          bi_noload_Aa_hip_energy,bi_noload_Aa_knee_energy,mono_noload_Aa_hip_energy,mono_noload_Aa_knee_energy),axis=0)
final_dataset = np.column_stack([assistive_actuators_col,assistive_actuators_avg_totalpower_data])
final_dataset = np.column_stack([subject_col,final_dataset])
with open(r'.\Statistics\Ideal_Vs_Constrained\ActuatorsAvgPower_Dataset.csv', 'wb') as f:
  f.write(bytes(utils.listToString(headers)+'\n','UTF-8'))
  np.savetxt(f, final_dataset, fmt='%s', delimiter=",")
