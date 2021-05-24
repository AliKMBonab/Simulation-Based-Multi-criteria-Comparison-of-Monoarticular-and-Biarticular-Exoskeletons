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
from Colors import colors as mycolors

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
# pareto exo max power dataset
directory = './Data/Pareto/*_max_power.csv'
files = enumerate(glob.iglob(directory), 1)
max_power_dataset = {pathlib.PurePath(f[1]).stem: np.loadtxt(f[1], delimiter=',') for f in files}
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
bi_loaded_metabolics_percent = utils.pareto_metabolics_reduction(assisted_energy_dataset['biarticular_pareto_load_metabolics_energy'],unassisted_energy_dataset['loaded_metabolics_energy'])
bi_noload_metabolics_percent = utils.pareto_metabolics_reduction(assisted_energy_dataset['biarticular_pareto_noload_metabolics_energy'],unassisted_energy_dataset['noload_metabolics_energy'])
mono_loaded_metabolics_percent = utils.pareto_metabolics_reduction(assisted_energy_dataset['monoarticular_pareto_load_metabolics_energy'],unassisted_energy_dataset['loaded_metabolics_energy'])
mono_noload_metabolics_percent = utils.pareto_metabolics_reduction(assisted_energy_dataset['monoarticular_pareto_noload_metabolics_energy'],unassisted_energy_dataset['noload_metabolics_energy'])
# mean & std metabolics cost reduction percents
mean_bi_loaded_metabolics_percent, std_bi_loaded_metabolics_percent = utils.pareto_avg_std_energy(bi_loaded_metabolics_percent,reshape=False)
mean_bi_noload_metabolics_percent, std_bi_noload_metabolics_percent = utils.pareto_avg_std_energy(bi_noload_metabolics_percent,reshape=False)
mean_mono_loaded_metabolics_percent, std_mono_loaded_metabolics_percent = utils.pareto_avg_std_energy(mono_loaded_metabolics_percent,reshape=False)
mean_mono_noload_metabolics_percent, std_mono_noload_metabolics_percent = utils.pareto_avg_std_energy(mono_noload_metabolics_percent,reshape=False)
# mean & std metabolics cost reduction
mean_bi_loaded_metabolics, std_bi_loaded_metabolics = utils.pareto_avg_std_energy(assisted_energy_dataset['biarticular_pareto_load_metabolics_energy'],reshape=True)
mean_bi_noload_metabolics, std_bi_noload_metabolics = utils.pareto_avg_std_energy(assisted_energy_dataset['biarticular_pareto_noload_metabolics_energy'],reshape=True)
mean_mono_loaded_metabolics, std_mono_loaded_metabolics = utils.pareto_avg_std_energy(assisted_energy_dataset['monoarticular_pareto_load_metabolics_energy'],reshape=True)
mean_mono_noload_metabolics, std_mono_noload_metabolics = utils.pareto_avg_std_energy(assisted_energy_dataset['monoarticular_pareto_noload_metabolics_energy'],reshape=True)
# mean & std actuators energy
# loaded bi
mean_bi_loaded_hip_energy, std_bi_loaded_hip_energy = utils.pareto_avg_std_energy(assisted_energy_dataset['biarticular_pareto_load_hipactuator_energy'],reshape=True)
mean_bi_loaded_knee_energy, std_bi_loaded_knee_energy = utils.pareto_avg_std_energy(assisted_energy_dataset['biarticular_pareto_load_kneeactuator_energy'],reshape=True)
mean_bi_loaded_energy, std_bi_loaded_energy = utils.pareto_avg_std_energy(assisted_energy_dataset['biarticular_pareto_load_kneeactuator_energy']+assisted_energy_dataset['biarticular_pareto_load_hipactuator_energy'],reshape=True)
# noload bi
mean_bi_noload_hip_energy, std_bi_noload_hip_energy = utils.pareto_avg_std_energy(assisted_energy_dataset['biarticular_pareto_noload_hipactuator_energy'],reshape=True)
mean_bi_noload_knee_energy, std_bi_noload_knee_energy = utils.pareto_avg_std_energy(assisted_energy_dataset['biarticular_pareto_noload_kneeactuator_energy'],reshape=True)
mean_bi_noload_energy, std_bi_noload_energy = utils.pareto_avg_std_energy(assisted_energy_dataset['biarticular_pareto_noload_kneeactuator_energy']+assisted_energy_dataset['biarticular_pareto_noload_hipactuator_energy'],reshape=True)
# loaded mono
mean_mono_loaded_hip_energy, std_mono_loaded_hip_energy = utils.pareto_avg_std_energy(assisted_energy_dataset['monoarticular_pareto_load_hipactuator_energy'],reshape=True)
mean_mono_loaded_knee_energy, std_mono_loaded_knee_energy = utils.pareto_avg_std_energy(assisted_energy_dataset['monoarticular_pareto_load_kneeactuator_energy'],reshape=True)
mean_mono_loaded_energy, std_mono_loaded_energy = utils.pareto_avg_std_energy(assisted_energy_dataset['monoarticular_pareto_load_kneeactuator_energy']+assisted_energy_dataset['monoarticular_pareto_load_hipactuator_energy'],reshape=True)
# noload mono
mean_mono_noload_hip_energy, std_mono_noload_hip_energy = utils.pareto_avg_std_energy(assisted_energy_dataset['monoarticular_pareto_noload_hipactuator_energy'],reshape=True)
mean_mono_noload_knee_energy, std_mono_noload_knee_energy = utils.pareto_avg_std_energy(assisted_energy_dataset['monoarticular_pareto_noload_kneeactuator_energy'],reshape=True)
mean_mono_noload_energy, std_mono_noload_energy = utils.pareto_avg_std_energy(assisted_energy_dataset['monoarticular_pareto_noload_kneeactuator_energy']+assisted_energy_dataset['monoarticular_pareto_noload_hipactuator_energy'],reshape=True)
# mean & std actuators maximum positive power
# loaded bi
mean_bi_loaded_hip_max_power, std_bi_loaded_hip_max_power = utils.pareto_avg_std_energy(max_power_dataset['biarticular_pareto_load_hip_max_power'],reshape=True)
mean_bi_loaded_knee_max_power, std_bi_loaded_knee_max_power = utils.pareto_avg_std_energy(max_power_dataset['biarticular_pareto_load_knee_max_power'],reshape=True)
mean_bi_loaded_max_power, std_bi_loaded_max_power = utils.pareto_avg_std_energy(max_power_dataset['biarticular_pareto_load_knee_max_power']+max_power_dataset['biarticular_pareto_load_hip_max_power'],reshape=True)
# noload bi
mean_bi_noload_hip_max_power, std_bi_noload_hip_max_power = utils.pareto_avg_std_energy(max_power_dataset['biarticular_pareto_noload_hip_max_power'],reshape=True)
mean_bi_noload_knee_max_power, std_bi_noload_knee_max_power = utils.pareto_avg_std_energy(max_power_dataset['biarticular_pareto_noload_knee_max_power'],reshape=True)
mean_bi_noload_max_power, std_bi_noload_max_power = utils.pareto_avg_std_energy(max_power_dataset['biarticular_pareto_noload_knee_max_power']+max_power_dataset['biarticular_pareto_noload_hip_max_power'],reshape=True)
# loaded mono
mean_mono_loaded_hip_max_power, std_mono_loaded_hip_max_power = utils.pareto_avg_std_energy(max_power_dataset['monoarticular_pareto_load_hip_max_power'],reshape=True)
mean_mono_loaded_knee_max_power, std_mono_loaded_knee_max_power = utils.pareto_avg_std_energy(max_power_dataset['monoarticular_pareto_load_knee_max_power'],reshape=True)
mean_mono_loaded_max_power, std_mono_loaded_max_power = utils.pareto_avg_std_energy(max_power_dataset['monoarticular_pareto_load_knee_max_power']+max_power_dataset['monoarticular_pareto_load_hip_max_power'],reshape=True)
# noload mono
mean_mono_noload_hip_max_power, std_mono_noload_hip_max_power = utils.pareto_avg_std_energy(max_power_dataset['monoarticular_pareto_noload_hip_max_power'],reshape=True)
mean_mono_noload_knee_max_power, std_mono_noload_knee_max_power = utils.pareto_avg_std_energy(max_power_dataset['monoarticular_pareto_noload_knee_max_power'],reshape=True)
mean_mono_noload_max_power, std_mono_noload_max_power = utils.pareto_avg_std_energy(max_power_dataset['monoarticular_pareto_noload_knee_max_power']+max_power_dataset['monoarticular_pareto_noload_hip_max_power'],reshape=True)

# mean & std actuators regenrative energy
# noload bi
mu = 0.65
mean_bi_noload_regen_energy, std_bi_noload_regen_energy = utils.pareto_avg_std_energy(-mu*assisted_energy_dataset['biarticular_pareto_noload_kneeregenrative_energy']-\
                                                                                       mu*assisted_energy_dataset['biarticular_pareto_noload_hipregenrative_energy']+\
                                                                                          assisted_energy_dataset['biarticular_pareto_noload_kneeactuator_energy']+\
                                                                                          assisted_energy_dataset['biarticular_pareto_noload_hipactuator_energy'],reshape=True)
# noload mono
mean_mono_noload_regen_energy, std_mono_noload_regen_energy = utils.pareto_avg_std_energy(-mu*assisted_energy_dataset['monoarticular_pareto_noload_kneeregenrative_energy']-\
                                                                                           mu*assisted_energy_dataset['monoarticular_pareto_noload_hipregenrative_energy']+\
                                                                                          assisted_energy_dataset['monoarticular_pareto_noload_kneeactuator_energy']+\
                                                                                          assisted_energy_dataset['monoarticular_pareto_noload_hipactuator_energy'],reshape=True)
# loaded bi
mean_bi_loaded_regen_energy, std_bi_loaded_regen_energy = utils.pareto_avg_std_energy(-mu*assisted_energy_dataset['biarticular_pareto_load_kneeregenrative_energy']-\
                                                                                       mu*assisted_energy_dataset['biarticular_pareto_load_hipregenrative_energy']+\
                                                                                          assisted_energy_dataset['biarticular_pareto_load_kneeactuator_energy']+\
                                                                                          assisted_energy_dataset['biarticular_pareto_load_hipactuator_energy'],reshape=True)
# loaded mono
mean_mono_loaded_regen_energy, std_mono_loaded_regen_energy = utils.pareto_avg_std_energy(-mu*assisted_energy_dataset['monoarticular_pareto_load_kneeregenrative_energy']-\
                                                                                           mu*assisted_energy_dataset['monoarticular_pareto_load_hipregenrative_energy']+\
                                                                                          assisted_energy_dataset['monoarticular_pareto_load_kneeactuator_energy']+\
                                                                                          assisted_energy_dataset['monoarticular_pareto_load_hipactuator_energy'],reshape=True)

#####################################################################################
# Paretofront data
# indices
bi_loaded_indices = np.array([25,24,23,22,21,17,16,13,12,11,6,1])
mono_loaded_indices = np.array([25,20,15,10,5,4,3,2,1])
bi_noload_indices = np.array([25,24,23,22,21,19,18,17,13,12,11,1])
mono_noload_indices = np.array([25,20,15,14,13,8,7,6,2,1])
# regenerated
bi_loaded_regen_indices = np.array([25,24,23,22,21,17,13,12,11,6,1])
mono_loaded_regen_indices = np.array([25,20,19,15,10,5,9,4,3,2,1])
bi_noload_regen_indices = np.array([25,24,23,22,21,19,18,17,16,13,12,11,6,1])
mono_noload_regen_indices = np.array([25,24,20,15,10,5,4,1])

# loaded biarticular
mean_bi_loaded_paretofront = utils.manual_paretofront(mean_bi_loaded_metabolics_percent,mean_bi_loaded_energy,bi_loaded_indices)
std_bi_loaded_paretofront = utils.manual_paretofront(std_bi_loaded_metabolics_percent,std_bi_loaded_energy,bi_loaded_indices)
#hip
mean_bi_loaded_hip_actuator_paretofront = utils.manual_paretofront(mean_bi_loaded_metabolics_percent,mean_bi_loaded_hip_energy,bi_loaded_indices)
std_bi_loaded_hip_actuator_paretofront = utils.manual_paretofront(std_bi_loaded_metabolics_percent,std_bi_loaded_hip_energy,bi_loaded_indices)
#knee
mean_bi_loaded_knee_actuator_paretofront = utils.manual_paretofront(mean_bi_loaded_metabolics_percent,mean_bi_loaded_knee_energy,bi_loaded_indices)
std_bi_loaded_knee_actuator_paretofront = utils.manual_paretofront(std_bi_loaded_metabolics_percent,std_bi_loaded_knee_energy,bi_loaded_indices)

# loaded monoarticular
mean_mono_loaded_paretofront = utils.manual_paretofront(mean_mono_loaded_metabolics_percent,mean_mono_loaded_energy,mono_loaded_indices)
std_mono_loaded_paretofront = utils.manual_paretofront(std_mono_loaded_metabolics_percent,std_mono_loaded_energy,mono_loaded_indices)
#hip
mean_mono_loaded_hip_actuator_paretofront = utils.manual_paretofront(mean_mono_loaded_metabolics_percent,mean_mono_loaded_hip_energy,mono_loaded_indices)
std_mono_loaded_hip_actuator_paretofront = utils.manual_paretofront(std_mono_loaded_metabolics_percent,std_mono_loaded_hip_energy,mono_loaded_indices)
#knee
mean_mono_loaded_knee_actuator_paretofront = utils.manual_paretofront(mean_mono_loaded_metabolics_percent,mean_mono_loaded_knee_energy,mono_loaded_indices)
std_mono_loaded_knee_actuator_paretofront = utils.manual_paretofront(std_mono_loaded_metabolics_percent,std_mono_loaded_knee_energy,mono_loaded_indices)

# noload biarticular
mean_bi_noload_paretofront = utils.manual_paretofront(mean_bi_noload_metabolics_percent,mean_bi_noload_energy,bi_noload_indices)
std_bi_noload_paretofront = utils.manual_paretofront(std_bi_noload_metabolics_percent,std_bi_noload_energy,bi_noload_indices)
#hip
mean_bi_noload_hip_actuator_paretofront = utils.manual_paretofront(mean_bi_noload_metabolics_percent,mean_bi_noload_hip_energy,bi_noload_indices)
std_bi_noload_hip_actuator_paretofront = utils.manual_paretofront(std_bi_noload_metabolics_percent,std_bi_noload_hip_energy,bi_noload_indices)
#knee
mean_bi_noload_knee_actuator_paretofront = utils.manual_paretofront(mean_bi_noload_metabolics_percent,mean_bi_noload_knee_energy,bi_noload_indices)
std_bi_noload_knee_actuator_paretofront = utils.manual_paretofront(std_bi_noload_metabolics_percent,std_bi_noload_knee_energy,bi_noload_indices)

# noload monoarticular
mean_mono_noload_paretofront = utils.manual_paretofront(mean_mono_noload_metabolics_percent,mean_mono_noload_energy,mono_noload_indices)
std_mono_noload_paretofront = utils.manual_paretofront(std_mono_noload_metabolics_percent,std_mono_noload_energy,mono_noload_indices)
#hip
mean_mono_noload_hip_actuator_paretofront = utils.manual_paretofront(mean_mono_noload_metabolics_percent,mean_mono_noload_hip_energy,mono_noload_indices)
std_mono_noload_hip_actuator_paretofront = utils.manual_paretofront(std_mono_noload_metabolics_percent,std_mono_noload_hip_energy,mono_noload_indices)
#knee
mean_mono_noload_knee_actuator_paretofront = utils.manual_paretofront(mean_mono_noload_metabolics_percent,mean_mono_noload_knee_energy,mono_noload_indices)
std_mono_noload_knee_actuator_paretofront = utils.manual_paretofront(std_mono_noload_metabolics_percent,std_mono_noload_knee_energy,mono_noload_indices)

# loaded biarticular regenerated
mean_bi_loaded_regen_paretofront = utils.manual_paretofront(mean_bi_loaded_metabolics_percent,mean_bi_loaded_regen_energy,bi_loaded_regen_indices)
std_bi_loaded_regen_paretofront = utils.manual_paretofront(std_bi_loaded_metabolics_percent,std_bi_loaded_regen_energy,bi_loaded_regen_indices)
# loaded monoarticular regenerated
mean_mono_loaded_regen_paretofront = utils.manual_paretofront(mean_mono_loaded_metabolics_percent,mean_mono_loaded_regen_energy,mono_loaded_regen_indices)
std_mono_loaded_regen_paretofront = utils.manual_paretofront(std_mono_loaded_metabolics_percent,std_mono_loaded_regen_energy,mono_loaded_regen_indices)
# noload biarticular regenerated
mean_bi_noload_regen_paretofront = utils.manual_paretofront(mean_bi_noload_metabolics_percent,mean_bi_noload_regen_energy,bi_noload_regen_indices)
std_bi_noload_regen_paretofront = utils.manual_paretofront(std_bi_noload_metabolics_percent,std_bi_noload_regen_energy,bi_noload_regen_indices)
# noload monoarticular regenerated
mean_mono_noload_regen_paretofront = utils.manual_paretofront(mean_mono_noload_metabolics_percent,mean_mono_noload_regen_energy,mono_noload_regen_indices)
std_mono_noload_regen_paretofront = utils.manual_paretofront(std_mono_noload_metabolics_percent,std_mono_noload_regen_energy,mono_noload_regen_indices)

#####################################################################################
# loaded biarticular
#hip
mean_bi_loaded_hip_max_power_paretofront = utils.manual_paretofront(mean_bi_loaded_metabolics_percent,mean_bi_loaded_hip_max_power,bi_loaded_indices)
std_bi_loaded_hip_max_power_paretofront = utils.manual_paretofront(std_bi_loaded_metabolics_percent,std_bi_loaded_hip_max_power,bi_loaded_indices)
#knee
mean_bi_loaded_knee_max_power_paretofront = utils.manual_paretofront(mean_bi_loaded_metabolics_percent,mean_bi_loaded_knee_max_power,bi_loaded_indices)
std_bi_loaded_knee_max_power_paretofront = utils.manual_paretofront(std_bi_loaded_metabolics_percent,std_bi_loaded_knee_max_power,bi_loaded_indices)

# loaded monoarticular
#hip
mean_mono_loaded_hip_max_power_paretofront = utils.manual_paretofront(mean_mono_loaded_metabolics_percent,mean_mono_loaded_hip_max_power,mono_loaded_indices)
std_mono_loaded_hip_max_power_paretofront = utils.manual_paretofront(std_mono_loaded_metabolics_percent,std_mono_loaded_hip_max_power,mono_loaded_indices)
#knee
mean_mono_loaded_knee_max_power_paretofront = utils.manual_paretofront(mean_mono_loaded_metabolics_percent,mean_mono_loaded_knee_max_power,mono_loaded_indices)
std_mono_loaded_knee_max_power_paretofront = utils.manual_paretofront(std_mono_loaded_metabolics_percent,std_mono_loaded_knee_max_power,mono_loaded_indices)

# noload biarticular
#hip
mean_bi_noload_hip_max_power_paretofront = utils.manual_paretofront(mean_bi_noload_metabolics_percent,mean_bi_noload_hip_max_power,bi_noload_indices)
std_bi_noload_hip_max_power_paretofront = utils.manual_paretofront(std_bi_noload_metabolics_percent,std_bi_noload_hip_max_power,bi_noload_indices)
#knee
mean_bi_noload_knee_max_power_paretofront = utils.manual_paretofront(mean_bi_noload_metabolics_percent,mean_bi_noload_knee_max_power,bi_noload_indices)
std_bi_noload_knee_max_power_paretofront = utils.manual_paretofront(std_bi_noload_metabolics_percent,std_bi_noload_knee_max_power,bi_noload_indices)

# noload monoarticular
#hip
mean_mono_noload_hip_max_power_paretofront = utils.manual_paretofront(mean_mono_noload_metabolics_percent,mean_mono_noload_hip_max_power,mono_noload_indices)
std_mono_noload_hip_max_power_paretofront = utils.manual_paretofront(std_mono_noload_metabolics_percent,std_mono_noload_hip_max_power,mono_noload_indices)
#knee
mean_mono_noload_knee_max_power_paretofront = utils.manual_paretofront(mean_mono_noload_metabolics_percent,mean_mono_noload_knee_max_power,mono_noload_indices)
std_mono_noload_knee_max_power_paretofront = utils.manual_paretofront(std_mono_noload_metabolics_percent,std_mono_noload_knee_max_power,mono_noload_indices)
'''
#####################################################################################
# Table: Ideal vs maximum constrained
bi_loaded_indices_table = np.subtract(np.flip(bi_loaded_indices),1)
mono_loaded_indices_table = np.subtract(np.flip(mono_loaded_indices),1)
bi_noload_indices_table = np.subtract(np.flip(bi_noload_indices),1)
mono_noload_indices_table = np.subtract(np.flip(mono_noload_indices),1)
# mean power consumption
cellText = {'biarticular, loaded':np.reshape(np.concatenate((np.round(np.take(mean_bi_loaded_hip_actuator_paretofront[:,1],bi_loaded_indices_table),2),np.round(np.take(mean_bi_loaded_knee_actuator_paretofront[:,1],bi_loaded_indices_table),2),np.round(np.take(mean_bi_loaded_hip_actuator_paretofront[:,0],bi_loaded_indices_table),2),\
                                                 np.round(np.take(std_bi_loaded_hip_actuator_paretofront[:,1],bi_loaded_indices_table),2), np.round(np.take(std_bi_loaded_knee_actuator_paretofront[:,1],bi_loaded_indices_table),2), np.round(np.take(std_bi_loaded_hip_actuator_paretofront[:,0],bi_loaded_indices_table),2)),axis=0),(bi_loaded_indices.shape[0],6),order='F'),
                                                 
            'monoarticular, loaded':np.reshape(np.concatenate((np.round(np.take(mean_mono_loaded_hip_actuator_paretofront[:,1],mono_loaded_indices_table),2),np.round(np.take(mean_mono_loaded_knee_actuator_paretofront[:,1],mono_loaded_indices_table),2),np.round(np.take(mean_mono_loaded_hip_actuator_paretofront[:,0],mono_loaded_indices_table),2),\
                                                 np.round(np.take(std_mono_loaded_hip_actuator_paretofront[:,1],mono_loaded_indices_table),2), np.round(np.take(std_mono_loaded_knee_actuator_paretofront[:,1],mono_loaded_indices_table),2), np.round(np.take(std_mono_loaded_hip_actuator_paretofront[:,0],mono_loaded_indices_table),2)),axis=0),(mono_loaded_indices.shape[0],6),order='F'),
                                                 
            'biarticular, noload':np.reshape(np.concatenate((np.round(np.take(mean_bi_noload_hip_actuator_paretofront[:,1],bi_noload_indices_table),2),np.round(np.take(mean_bi_noload_knee_actuator_paretofront[:,1],bi_noload_indices_table),2),np.round(np.take(mean_bi_noload_hip_actuator_paretofront[:,0],bi_noload_indices_table),2),\
                                                 np.round(np.take(std_bi_noload_hip_actuator_paretofront[:,1],bi_noload_indices_table),2), np.round(np.take(std_bi_noload_knee_actuator_paretofront[:,1],bi_noload_indices_table),2), np.round(np.take(std_bi_noload_hip_actuator_paretofront[:,0],bi_noload_indices_table),2)),axis=0),(bi_noload_indices.shape[0],6),order='F'),
                                                 
            'monoarticular, noload':np.reshape(np.concatenate((np.round(np.take(mean_mono_noload_hip_actuator_paretofront[:,1],mono_noload_indices_table),2),np.round(np.take(mean_mono_noload_knee_actuator_paretofront[:,1],mono_noload_indices_table),2),np.round(np.take(mean_mono_noload_hip_actuator_paretofront[:,0],mono_noload_indices_table),2),\
                                                 np.round(np.take(std_mono_noload_hip_actuator_paretofront[:,1],mono_noload_indices_table),2), np.round(np.take(std_mono_noload_knee_actuator_paretofront[:,1],mono_noload_indices_table),2), np.round(np.take(std_mono_noload_hip_actuator_paretofront[:,0],mono_noload_indices_table),2)),axis=0),(mono_noload_indices.shape[0],6),order='F')}
# maximum positive power
cellText_MaxPower = {'biarticular, loaded':np.reshape(np.concatenate((np.round(np.take(mean_bi_loaded_hip_max_power_paretofront[:,1],bi_loaded_indices_table),2),np.round(np.take(mean_bi_loaded_knee_max_power_paretofront[:,1],bi_loaded_indices_table),2),np.round(np.take(mean_bi_loaded_hip_max_power_paretofront[:,0],bi_loaded_indices_table),2),\
                                                 np.round(np.take(std_bi_loaded_hip_max_power_paretofront[:,1],bi_loaded_indices_table),2), np.round(np.take(std_bi_loaded_knee_max_power_paretofront[:,1],bi_loaded_indices_table),2), np.round(np.take(std_bi_loaded_hip_max_power_paretofront[:,0],bi_loaded_indices_table),2)),axis=0),(bi_loaded_indices.shape[0],6),order='F'),
                                                 
            'monoarticular, loaded':np.reshape(np.concatenate((np.round(np.take(mean_mono_loaded_hip_max_power_paretofront[:,1],mono_loaded_indices_table),2),np.round(np.take(mean_mono_loaded_knee_max_power_paretofront[:,1],mono_loaded_indices_table),2),np.round(np.take(mean_mono_loaded_hip_max_power_paretofront[:,0],mono_loaded_indices_table),2),\
                                                 np.round(np.take(std_mono_loaded_hip_max_power_paretofront[:,1],mono_loaded_indices_table),2), np.round(np.take(std_mono_loaded_knee_max_power_paretofront[:,1],mono_loaded_indices_table),2), np.round(np.take(std_mono_loaded_hip_max_power_paretofront[:,0],mono_loaded_indices_table),2)),axis=0),(mono_loaded_indices.shape[0],6),order='F'),
                                                 
            'biarticular, noload':np.reshape(np.concatenate((np.round(np.take(mean_bi_noload_hip_max_power_paretofront[:,1],bi_noload_indices_table),2),np.round(np.take(mean_bi_noload_knee_max_power_paretofront[:,1],bi_noload_indices_table),2),np.round(np.take(mean_bi_noload_hip_max_power_paretofront[:,0],bi_noload_indices_table),2),\
                                                 np.round(np.take(std_bi_noload_hip_max_power_paretofront[:,1],bi_noload_indices_table),2), np.round(np.take(std_bi_noload_knee_max_power_paretofront[:,1],bi_noload_indices_table),2), np.round(np.take(std_bi_noload_hip_max_power_paretofront[:,0],bi_noload_indices_table),2)),axis=0),(bi_noload_indices.shape[0],6),order='F'),
                                                 
            'monoarticular, noload':np.reshape(np.concatenate((np.round(np.take(mean_mono_noload_hip_max_power_paretofront[:,1],mono_noload_indices_table),2),np.round(np.take(mean_mono_noload_knee_max_power_paretofront[:,1],mono_noload_indices_table),2),np.round(np.take(mean_mono_noload_hip_max_power_paretofront[:,0],mono_noload_indices_table),2),\
                                                 np.round(np.take(std_mono_noload_hip_max_power_paretofront[:,1],mono_noload_indices_table),2), np.round(np.take(std_mono_noload_knee_max_power_paretofront[:,1],mono_noload_indices_table),2), np.round(np.take(std_mono_noload_hip_max_power_paretofront[:,0],mono_noload_indices_table),2)),axis=0),(mono_noload_indices.shape[0],6),order='F')}
# rows and columns
rows = {'biarticular, loaded':['biarticular,\n loaded, W-{}'.format(i) for i in np.flip(bi_loaded_indices)],\
        'monoarticular, loaded':['monoarticular,\n loaded, W-{}'.format(i) for i in np.flip(mono_loaded_indices)],\
       'biarticular, noload':['biarticular,\n noload, W-{}'.format(i) for i in np.flip(bi_noload_indices)],\
        'monoarticular, noload':['monoarticular,\n noload, W-{}'.format(i) for i in np.flip(mono_noload_indices)]}
columns = ['mean hip\n actuator energy (W/kg)','mean knee\n actuator energy (W/kg)','mean metabolic\n reduction (%)',\
           'std hip\n actuator energy (W/kg)','std knee\n actuator energy (W/kg)','std metabolic\n reduction (%)']
#####################################################################################
# Tables of power consumption
fig, ax = plt.subplots(figsize=(12,8))
table = ax.table(cellText=cellText['biarticular, loaded'],rowLabels=rows['biarticular, loaded'],colLabels=columns,loc='center')
table.scale(1,2)
table.set_fontsize(14)
ax.axis('off')
fig.savefig('./Figures/Paretofront/Mean_Pareto/Biarticular_Loaded_Energy_Table.pdf',orientation='landscape',bbox_inches='tight')
plt.show()
fig, ax = plt.subplots(figsize=(12,8))
table = ax.table(cellText=cellText['monoarticular, loaded'],rowLabels=rows['monoarticular, loaded'],colLabels=columns,loc='center')
table.scale(1,2)
table.set_fontsize(14)
ax.axis('off')
fig.savefig('./Figures/Paretofront/Mean_Pareto/Monoarticular_Loaded_Energy_Table.pdf',orientation='landscape',bbox_inches='tight')
plt.show()
fig, ax = plt.subplots(figsize=(12,8))
table = ax.table(cellText=cellText['biarticular, noload'],rowLabels=rows['biarticular, noload'],colLabels=columns,loc='center')
table.scale(1,2)
table.set_fontsize(14)
ax.axis('off')
fig.savefig('./Figures/Paretofront/Mean_Pareto/Biarticular_Noload_Energy_Table.pdf',orientation='landscape',bbox_inches='tight')
plt.show()
fig, ax = plt.subplots(figsize=(12,8))
table = ax.table(cellText=cellText['monoarticular, noload'],rowLabels=rows['monoarticular, noload'],colLabels=columns,loc='center')
table.scale(1,2)
table.set_fontsize(14)
ax.axis('off')
fig.savefig('./Figures/Paretofront/Mean_Pareto/Monoarticular_Noload_Energy_Table.pdf',orientation='landscape',bbox_inches='tight')
plt.show()
#####################################################################################
# Tables of maximum positive power consumption
fig, ax = plt.subplots(figsize=(12,8))
table = ax.table(cellText=cellText_MaxPower ['biarticular, loaded'],rowLabels=rows['biarticular, loaded'],colLabels=columns,loc='center')
table.scale(1,2)
table.set_fontsize(14)
ax.axis('off')
fig.savefig('./Figures/Paretofront/Mean_Pareto/Biarticular_Loaded_MaxPower_Table.pdf',orientation='landscape',bbox_inches='tight')
plt.show()
fig, ax = plt.subplots(figsize=(12,8))
table = ax.table(cellText=cellText_MaxPower['monoarticular, loaded'],rowLabels=rows['monoarticular, loaded'],colLabels=columns,loc='center')
table.scale(1,2)
table.set_fontsize(14)
ax.axis('off')
fig.savefig('./Figures/Paretofront/Mean_Pareto/Monoarticular_Loaded_MaxPower_Table.pdf',orientation='landscape',bbox_inches='tight')
plt.show()
fig, ax = plt.subplots(figsize=(12,8))
table = ax.table(cellText=cellText_MaxPower['biarticular, noload'],rowLabels=rows['biarticular, noload'],colLabels=columns,loc='center')
table.scale(1,2)
table.set_fontsize(14)
ax.axis('off')
fig.savefig('./Figures/Paretofront/Mean_Pareto/Biarticular_Noload_MaxPower_Table.pdf',orientation='landscape',bbox_inches='tight')
plt.show()
fig, ax = plt.subplots(figsize=(12,8))
table = ax.table(cellText=cellText_MaxPower['monoarticular, noload'],rowLabels=rows['monoarticular, noload'],colLabels=columns,loc='center')
table.scale(1,2)
table.set_fontsize(14)
ax.axis('off')
fig.savefig('./Figures/Paretofront/Mean_Pareto/Monoarticular_Noload_MaxPower_Table.pdf',orientation='landscape',bbox_inches='tight')
plt.show()
'''
#####################################################################################
# PAPER FIGURE
# plots
# average pareto curve: loaded mono vs biarticular
plt.rcParams.update({'font.size': 14})
plot_dic = {'x1_data':mean_bi_loaded_paretofront[:,0],'x1err_data':std_bi_loaded_paretofront[:,0],
          'x2_data':mean_mono_loaded_paretofront[:,0],'x2err_data':std_mono_loaded_paretofront[:,0],
          'y1_data':mean_bi_loaded_paretofront[:,1],'y1err_data':std_bi_loaded_paretofront[:,1],
          'y2_data':mean_mono_loaded_paretofront[:,1],'y2err_data':std_mono_loaded_paretofront[:,1],
          'color_1':mycolors['crimson red'],'color_2':mycolors['dark purple'],
          'legend_1':'bi-articular, loaded','legend_2':'mono-articular, loaded',
          'x1_ideal': ideal_dataset['mean_loaded_bi_metabolics'],'x1err_ideal': ideal_dataset['std_loaded_bi_metabolics'],
          'x2_ideal': ideal_dataset['mean_loaded_mono_metabolics'],'x2err_ideal': ideal_dataset['std_loaded_mono_metabolics'],
          'y1_ideal': ideal_dataset['mean_loaded_bi_energy'],'y1err_ideal': ideal_dataset['std_loaded_bi_energy'],
          'y2_ideal': ideal_dataset['mean_loaded_mono_energy'],'y2err_ideal': ideal_dataset['std_loaded_mono_energy'],
          }
fig, axes = plt.subplots(nrows=2,ncols=2,num='PaperFigure_Paretofront',figsize=(12.8, 9.6))
plt.subplot(2,2,1)
utils.plot_pareto_avg_curve (plot_dic,loadcond='loaded',line=True,ideal_configs=True)
plt.title('loaded: mono-articular vs bi-articular\n')
plt.ylabel('exoskeleton power\n consumption (W/kg)')
ax = plt.gca()
ax.set_xticks([5, 10, 15, 20, 25, 30])
ax.set_yticks([1, 1.5, 2, 2.5, 3, 3.5])
plt.tick_params(axis='both',direction='in')
utils.no_top_right(ax)
plt.legend(frameon=False,loc='upper left')

# average pareto curve: noload mono vs biarticular

plot_dic = {'x1_data':mean_bi_noload_paretofront[:,0],'x1err_data':std_bi_noload_paretofront[:,0],
          'x2_data':mean_mono_noload_paretofront[:,0],'x2err_data':std_mono_noload_paretofront[:,0],
          'y1_data':mean_bi_noload_paretofront[:,1],'y1err_data':std_bi_noload_paretofront[:,1],
          'y2_data':mean_mono_noload_paretofront[:,1],'y2err_data':std_mono_noload_paretofront[:,1],
          'color_1':mycolors['magenta pink'],'color_2':mycolors['lavender purple'],
          'legend_1':'bi-articular, unloaded','legend_2':'mono-articular, unloaded',
          'x1_ideal': ideal_dataset['mean_noload_bi_metabolics'],'x1err_ideal': ideal_dataset['std_noload_bi_metabolics'],
          'x2_ideal': ideal_dataset['mean_noload_mono_metabolics'],'x2err_ideal': ideal_dataset['std_noload_mono_metabolics'],
          'y1_ideal': ideal_dataset['mean_noload_bi_energy'],'y1err_ideal': ideal_dataset['std_noload_bi_energy'],
          'y2_ideal': ideal_dataset['mean_noload_mono_energy'],'y2err_ideal': ideal_dataset['std_noload_mono_energy']
          }
plt.subplot(2,2,2)
utils.plot_pareto_avg_curve (plot_dic,loadcond='noload',line=True,ideal_configs=True)
plt.title('unloaded: mono-articular vs bi-articular\n')
plt.tick_params(axis='both',direction='in')
ax = plt.gca()
ax.set_xticks([5, 10, 15, 20, 25, 30])
ax.set_yticks([1, 1.5, 2, 2.5, 3, 3.5])
utils.no_top_right(ax)
plt.legend(frameon=False,loc='upper left')

# average pareto curve: biarticular noload vs loaded

plot_dic = {'x1_data':mean_bi_loaded_paretofront[:,0],'x1err_data':std_bi_loaded_paretofront[:,0],
          'x2_data':mean_bi_noload_paretofront[:,0],'x2err_data':std_bi_noload_paretofront[:,0],
          'y1_data':mean_bi_loaded_paretofront[:,1],'y1err_data':std_bi_loaded_paretofront[:,1],
          'y2_data':mean_bi_noload_paretofront[:,1],'y2err_data':std_bi_noload_paretofront[:,1],
          'color_1':mycolors['crimson red'],'color_2':mycolors['olympic blue'],
          'legend_1':'bi-articular, loaded','legend_2':'bi-articular, unloaded',
          'x1_ideal': ideal_dataset['mean_loaded_bi_metabolics'],'x1err_ideal': ideal_dataset['std_loaded_bi_metabolics'],
          'x2_ideal': ideal_dataset['mean_noload_bi_metabolics'],'x2err_ideal': ideal_dataset['std_noload_bi_metabolics'],
          'y1_ideal': ideal_dataset['mean_loaded_bi_energy'],'y1err_ideal': ideal_dataset['std_loaded_bi_energy'],
          'y2_ideal': ideal_dataset['mean_noload_bi_energy'],'y2err_ideal': ideal_dataset['std_noload_bi_energy']
          }
plt.subplot(2,2,3)
utils.plot_pareto_avg_curve (plot_dic,loadcond='loaded',line=True,ideal_configs=True)
plt.title('bi-articular: loaded vs unloaded\n')
plt.ylabel('exoskeleton power\n consumption (W/kg)')
plt.tick_params(axis='both',direction='in')
ax = plt.gca()
ax.set_xticks([5, 10, 15, 20, 25, 30])
ax.set_yticks([1, 1.5, 2, 2.5, 3, 3.5])
ax.set_xlabel('metabolic cost\nreduction (%)')
utils.no_top_right(ax)
plt.legend(frameon=False,loc='upper left')

# average pareto curve: monoarticular noload vs loaded

plot_dic = {'x1_data':mean_mono_loaded_paretofront[:,0],'x1err_data':std_mono_loaded_paretofront[:,0],
          'x2_data':mean_mono_noload_paretofront[:,0],'x2err_data':std_mono_noload_paretofront[:,0],
          'y1_data':mean_mono_loaded_paretofront[:,1],'y1err_data':std_mono_loaded_paretofront[:,1],
          'y2_data':mean_mono_noload_paretofront[:,1],'y2err_data':std_mono_noload_paretofront[:,1],
          'color_1':mycolors['crimson red'],'color_2':mycolors['olympic blue'],
          'legend_1':'mono-articular, loaded','legend_2':'mono-articular, unloaded',
          'x1_ideal': ideal_dataset['mean_loaded_mono_metabolics'],'x1err_ideal': ideal_dataset['std_loaded_mono_metabolics'],
          'x2_ideal': ideal_dataset['mean_noload_mono_metabolics'],'x2err_ideal': ideal_dataset['std_noload_mono_metabolics'],
          'y1_ideal': ideal_dataset['mean_loaded_mono_energy'],'y1err_ideal': ideal_dataset['std_loaded_mono_energy'],
          'y2_ideal': ideal_dataset['mean_noload_mono_energy'],'y2err_ideal': ideal_dataset['std_noload_mono_energy']
          }
plt.subplot(2,2,4)
utils.plot_pareto_avg_curve (plot_dic,loadcond='loaded',line=True,ideal_configs=True)
plt.title('mono-articular: loaded vs unloaded\n')
plt.tick_params(axis='both',direction='in')
ax = plt.gca()
ax.set_xticks([5, 10, 15, 20, 25, 30])
ax.set_yticks([1, 1.5, 2, 2.5, 3, 3.5])
ax.set_xlabel('metabolic cost\nreduction (%)')
utils.no_top_right(ax)
plt.legend(frameon=False,loc='upper left')
fig.tight_layout(h_pad=-1, w_pad=-1.5)
fig.subplots_adjust(top=0.98, bottom=0.075, left=0.100, right=0.975,hspace=0.30,wspace=0.20)
fig.savefig('./Figures/Paretofront/Mean_Pareto/PaperFigure_Main_Pareto.pdf',orientation='landscape',bbox_inches='tight')
plt.show()
#####################################################################################
# PAPER FIGURE
# average pareto curve: loaded mono vs biarticular REGENERATED

plot_dic = {'x1_data':mean_bi_loaded_regen_paretofront[:,0],'x1err_data':std_bi_loaded_regen_paretofront[:,0],
          'x2_data':mean_mono_loaded_regen_paretofront[:,0],'x2err_data':std_mono_loaded_regen_paretofront[:,0],
          'y1_data':mean_bi_loaded_regen_paretofront[:,1],'y1err_data':std_bi_loaded_regen_paretofront[:,1],
          'y2_data':mean_mono_loaded_regen_paretofront[:,1],'y2err_data':std_mono_loaded_regen_paretofront[:,1],
          'color_1':mycolors['crimson red'],'color_2':mycolors['dark purple']
          }
fig, axes = plt.subplots(nrows=2,ncols=2,num='PaperFigure_Paretofront',figsize=(12.8, 9.6))
plt.subplot(2,2,1)
utils.plot_pareto_avg_curve (plot_dic,loadcond='loaded',line=True)
plt.ylabel('exoskeleton power\n consumption (W/kg)')
plt.title('loaded: mono-articular vs bi-articular\nregenerated')
ax = plt.gca()
ax.set_xticks([5, 10, 15, 20, 25, 30])
ax.set_yticks([1, 1.5, 2, 2.5,3])
plt.tick_params(axis='both',direction='in')
plt.legend(frameon=False,loc='upper left')
utils.no_top_right(ax)
plt.legend(loc='best',frameon=False)

# average pareto curve: noload mono vs biarticular

plot_dic = {'x1_data':mean_bi_noload_regen_paretofront[:,0],'x1err_data':std_bi_noload_regen_paretofront[:,0],
          'x2_data':mean_mono_noload_regen_paretofront[:,0],'x2err_data':std_mono_noload_regen_paretofront[:,0],
          'y1_data':mean_bi_noload_regen_paretofront[:,1],'y1err_data':std_bi_noload_regen_paretofront[:,1],
          'y2_data':mean_mono_noload_regen_paretofront[:,1],'y2err_data':std_mono_noload_regen_paretofront[:,1],
          'color_1':mycolors['magenta pink'],'color_2':mycolors['lavender purple']
          }
plt.subplot(2,2,2)
utils.plot_pareto_avg_curve (plot_dic,loadcond='unloaded',line=True)
plt.title('unloaded: mono-articular vs bi-articular\nregenerated')
plt.tick_params(axis='both',direction='in')
ax = plt.gca()
ax.set_xticks([5, 10, 15, 20, 25, 30])
ax.set_yticks([1, 1.5, 2, 2.5,3])
utils.no_top_right(ax)
plt.legend(frameon=False,loc='upper left')

# average pareto curve: biarticular noload vs loaded

plot_dic = {'x1_data':mean_bi_loaded_regen_paretofront[:,0],'x1err_data':std_bi_loaded_regen_paretofront[:,0],
          'x2_data':mean_bi_noload_regen_paretofront[:,0],'x2err_data':std_bi_noload_regen_paretofront[:,0],
          'y1_data':mean_bi_loaded_regen_paretofront[:,1],'y1err_data':std_bi_loaded_regen_paretofront[:,1],
          'y2_data':mean_bi_noload_regen_paretofront[:,1],'y2err_data':std_bi_noload_regen_paretofront[:,1],
          'color_1':mycolors['crimson red'],'color_2':mycolors['olympic blue'],
          'legend_1':'bi-articular, loaded','legend_2':'bi-articular, unloaded'
          }
plt.subplot(2,2,3)
utils.plot_pareto_avg_curve (plot_dic,loadcond='loaded',line=True)
plt.title('biarticular: loaded vs unloaded\nregenerated')
plt.ylabel('exoskeleton power\n consumption (W/kg)')
plt.tick_params(axis='both',direction='in')
ax = plt.gca()
ax.set_xticks([5, 10, 15, 20, 25, 30])
ax.set_yticks([1, 1.5, 2, 2.5,3])
ax.set_xlabel('metabolic cost\nreduction (%)')
utils.no_top_right(ax)
plt.legend(frameon=False,loc='upper left')

# average pareto curve: monoarticular noload vs loaded

plot_dic = {'x1_data':mean_mono_loaded_regen_paretofront[:,0],'x1err_data':std_mono_loaded_regen_paretofront[:,0],
          'x2_data':mean_mono_noload_regen_paretofront[:,0],'x2err_data':std_mono_noload_regen_paretofront[:,0],
          'y1_data':mean_mono_loaded_regen_paretofront[:,1],'y1err_data':std_mono_loaded_regen_paretofront[:,1],
          'y2_data':mean_mono_noload_regen_paretofront[:,1],'y2err_data':std_mono_noload_regen_paretofront[:,1],
          'color_1':mycolors['crimson red'],'color_2':mycolors['olympic blue'],
          'legend_1':'mono-articular, loaded','legend_2':'mono-articular, unloaded'
          }
plt.subplot(2,2,4)
utils.plot_pareto_avg_curve (plot_dic,loadcond='loaded',line=True)
plt.title('mono-articular: loaded vs unloaded\nregenerated')
plt.tick_params(axis='both',direction='in')
ax = plt.gca()
ax.set_xticks([5, 10, 15, 20, 25, 30])
ax.set_yticks([1, 1.5, 2, 2.5,3])
ax.set_xlabel('metabolic cost\nreduction (%)')
utils.no_top_right(ax)
plt.legend(frameon=False,loc='upper left')
fig.tight_layout(h_pad=-1, w_pad=-1.5)
fig.subplots_adjust(top=0.98, bottom=0.075, left=0.100, right=0.975,hspace=0.30,wspace=0.20)
fig.savefig('./Figures/Paretofront/Mean_Pareto/PaperFigure_Main_Pareto_Regenerated.pdf',orientation='landscape',bbox_inches='tight')
plt.show()

#####################################################################################
# PAPER FIGURE
# plots
# average energy: loaded biarticular
plot_dic = {'x1_data':mean_bi_loaded_hip_actuator_paretofront[:,1],'x1err_data':std_bi_loaded_hip_actuator_paretofront[:,1],
            'y1_data':mean_bi_loaded_knee_actuator_paretofront[:,1],'y1err_data':std_bi_loaded_knee_actuator_paretofront[:,1],
          }
fig, axes = plt.subplots(nrows=2,ncols=2,num='PaperFigure_Paretofront_EnergyBarPlot',figsize=(12.8, 9.6))
plt.subplot(2,2,4)
utils.paretofront_barplot (plot_dic,bi_loaded_indices,loadcond='loaded')
plt.title('loaded, bi-articular')
ax = plt.gca()
ax.set_yticks([0, 0.5, 1, 1.5,2])
plt.tick_params(axis='both',direction='in')
utils.no_top_right(ax)
plt.legend(loc='best',frameon=False)

# average energy: loaded monoarticular
plot_dic = {'x1_data':mean_mono_loaded_hip_actuator_paretofront[:,1],'x1err_data':std_mono_loaded_hip_actuator_paretofront[:,1],
            'y1_data':mean_mono_loaded_knee_actuator_paretofront[:,1],'y1err_data':std_mono_loaded_knee_actuator_paretofront[:,1],
          }
plt.subplot(2,2,3)
utils.paretofront_barplot (plot_dic,mono_loaded_indices,loadcond='loaded')
plt.ylabel('power consumption (W/kg)')
plt.title('loaded, mono-articular')
ax = plt.gca()
ax.set_yticks([0, 0.5, 1, 1.5,2])
plt.tick_params(axis='both',direction='in')
utils.no_top_right(ax)
plt.legend(loc='best',frameon=False)

# average energy: noload biarticular

plot_dic = {'x1_data':mean_bi_noload_hip_actuator_paretofront[:,1],'x1err_data':std_bi_noload_hip_actuator_paretofront[:,1],
            'y1_data':mean_bi_noload_knee_actuator_paretofront[:,1],'y1err_data':std_bi_noload_knee_actuator_paretofront[:,1],
          }
plt.subplot(2,2,2)
utils.paretofront_barplot (plot_dic,bi_noload_indices,loadcond='unloaded')
plt.title('unloaded, bi-articular')
ax = plt.gca()
ax.set_yticks([0, 0.5, 1, 1.5,2])
plt.tick_params(axis='both',direction='in')
utils.no_top_right(ax)
plt.legend(loc='best',frameon=False)

# average energy: noload monoarticular

plot_dic = {'x1_data':mean_mono_noload_hip_actuator_paretofront[:,1],'x1err_data':std_mono_noload_hip_actuator_paretofront[:,1],
            'y1_data':mean_mono_noload_knee_actuator_paretofront[:,1],'y1err_data':std_mono_noload_knee_actuator_paretofront[:,1],
          }
plt.subplot(2,2,1)
utils.paretofront_barplot (plot_dic,mono_noload_indices,loadcond='unloaded')
plt.ylabel('power consumption (W/kg)')
plt.title('unloaded, mono-articular')
ax = plt.gca()
ax.set_yticks([0, 0.5, 1, 1.5,2])
plt.tick_params(axis='both',direction='in')
utils.no_top_right(ax)
plt.legend(loc='best',frameon=False)
fig.tight_layout(h_pad=-1, w_pad=-1.5)
fig.subplots_adjust(top=0.98, bottom=0.075, left=0.100, right=0.975,hspace=0.25,wspace=0.15)
fig.savefig('./Figures/Paretofront/Mean_Pareto/PaperFigure_Paretofront_EnergyBarPlot.pdf',orientation='landscape',bbox_inches='tight')
plt.show()
#####################################################################################
'''
# plots
# average pareto curve: loaded mono vs biarticular

plot_dic = {'x1_data':mean_bi_loaded_paretofront[:,0],'x1err_data':std_bi_loaded_paretofront[:,0],
          'x2_data':mean_mono_loaded_paretofront[:,0],'x2err_data':std_mono_loaded_paretofront[:,0],
          'y1_data':mean_bi_loaded_paretofront[:,1],'y1err_data':std_bi_loaded_paretofront[:,1],
          'y2_data':mean_mono_loaded_paretofront[:,1],'y2err_data':std_mono_loaded_paretofront[:,1],
          'color_1':mycolors['crimson red'],'color_2':mycolors['dark purple']
          }
fig = plt.figure(num='Pareto-Front: loaded mono vs bi',figsize=(10.4, 8.8))
utils.plot_pareto_avg_curve (plot_dic,loadcond='loaded',line=True)
plt.xlabel('Metabolic Cost Reduction (%)')
plt.ylabel('Exoskeleton Energy Consumption (W/kg)')
ax = plt.gca()
utils.no_top_right(ax)
plt.legend(loc='best',frameon=False)
fig.tight_layout()
fig.savefig('./Figures/Paretofront/Mean_Pareto/Paretofront_Load_BiVsMono.pdf',orientation='landscape',bbox_inches='tight')
plt.show()

# average pareto curve: noload mono vs biarticular

plot_dic = {'x1_data':mean_bi_noload_paretofront[:,0],'x1err_data':std_bi_noload_paretofront[:,0],
          'x2_data':mean_mono_noload_paretofront[:,0],'x2err_data':std_mono_noload_paretofront[:,0],
          'y1_data':mean_bi_noload_paretofront[:,1],'y1err_data':std_bi_noload_paretofront[:,1],
          'y2_data':mean_mono_noload_paretofront[:,1],'y2err_data':std_mono_noload_paretofront[:,1],
          'color_1':mycolors['magenta pink'],'color_2':mycolors['lavender purple']
          }
fig = plt.figure(num='Pareto-Front: noload mono vs bi',figsize=(10.4, 8.8))
utils.plot_pareto_avg_curve (plot_dic,loadcond='noload',line=True)
plt.xlabel('Metabolic Cost Reduction (%)')
plt.ylabel('Exoskeleton Energy Consumption (W/kg)')
ax = plt.gca()
utils.no_top_right(ax)
plt.legend(loc='best',frameon=False)
fig.tight_layout()
fig.savefig('./Figures/Paretofront/Mean_Pareto/Paretofront_Noload_BiVsMono.pdf',orientation='landscape',bbox_inches='tight')
plt.show()

# average pareto curve: biarticular noload vs loaded

plot_dic = {'x1_data':mean_bi_loaded_paretofront[:,0],'x1err_data':std_bi_loaded_paretofront[:,0],
          'x2_data':mean_bi_noload_paretofront[:,0],'x2err_data':std_bi_noload_paretofront[:,0],
          'y1_data':mean_bi_loaded_paretofront[:,1],'y1err_data':std_bi_loaded_paretofront[:,1],
          'y2_data':mean_bi_noload_paretofront[:,1],'y2err_data':std_bi_noload_paretofront[:,1],
          'color_1':mycolors['crimson red'],'color_2':mycolors['olympic blue'],
          'legend_1':'loaded biarticular','legend_2':'noload biarticular'
          }
fig = plt.figure(num='Pareto Curve: biarticular loaded vs noload',figsize=(10.4, 8.8))
utils.plot_pareto_avg_curve (plot_dic,loadcond='loaded',line=True)
plt.xlabel('Metabolic Cost Reduction (%)')
plt.ylabel('Exoskeleton Energy Consumption (W/kg)')
ax = plt.gca()
utils.no_top_right(ax)
plt.legend(loc='best',frameon=False)
fig.tight_layout()
fig.savefig('./Figures/Paretofront/Mean_Pareto/Pareto_Biarticular_LoadedvsNoload.pdf',orientation='landscape',bbox_inches='tight')
plt.show()

# average pareto curve: monoarticular noload vs loaded

plot_dic = {'x1_data':mean_mono_loaded_paretofront[:,0],'x1err_data':std_mono_loaded_paretofront[:,0],
          'x2_data':mean_mono_noload_paretofront[:,0],'x2err_data':std_mono_noload_paretofront[:,0],
          'y1_data':mean_mono_loaded_paretofront[:,1],'y1err_data':std_mono_loaded_paretofront[:,1],
          'y2_data':mean_mono_noload_paretofront[:,1],'y2err_data':std_mono_noload_paretofront[:,1],
          'color_1':mycolors['crimson red'],'color_2':mycolors['olympic blue'],
          'legend_1':'loaded monoarticular','legend_2':'noload monoarticular'
          }
fig = plt.figure(num='Pareto Curve: monoarticular loaded vs noload',figsize=(10.4, 8.8))
utils.plot_pareto_avg_curve (plot_dic,loadcond='loaded',line=True)
plt.xlabel('Metabolic Cost Reduction (%)')
plt.ylabel('Exoskeleton Energy Consumption (W/kg)')
ax = plt.gca()
utils.no_top_right(ax)
plt.legend(loc='best',frameon=False)
fig.tight_layout()
fig.savefig('./Figures/Paretofront/Mean_Pareto/Pareto_Monoarticular_LoadedvsNoload.pdf',orientation='landscape',bbox_inches='tight')
plt.show()

#####################################################################################
# plots with regeneration
# average pareto curve: loaded mono vs biarticular

plot_dic = {'x1_data':mean_bi_loaded_regen_paretofront[:,0],'x1err_data':std_bi_loaded_regen_paretofront[:,0],
          'x2_data':mean_mono_loaded_regen_paretofront[:,0],'x2err_data':std_mono_loaded_regen_paretofront[:,0],
          'y1_data':mean_bi_loaded_regen_paretofront[:,1],'y1err_data':std_bi_loaded_regen_paretofront[:,1],
          'y2_data':mean_mono_loaded_regen_paretofront[:,1],'y2err_data':std_mono_loaded_regen_paretofront[:,1],
          'color_1':mycolors['crimson red'],'color_2':mycolors['dark purple']
          }
fig = plt.figure(num='Paretofront: loaded mono vs bi',figsize=(10.4, 8.8))
utils.plot_pareto_avg_curve (plot_dic,loadcond='loaded',line=True)
plt.xlabel('Metabolic Cost Reduction (%)')
plt.ylabel('Exoskeleton Energy Consumption (W/kg)')
ax = plt.gca()
utils.no_top_right(ax)
plt.legend(loc='best',frameon=False)
fig.tight_layout()
fig.savefig('./Figures/Paretofront/Mean_Pareto/Pareto_Load_BiVsMono_Regenerated.pdf',orientation='landscape',bbox_inches='tight')
plt.show()

# average pareto curve: loaded mono non-regenerated vs regenerated

plot_dic = {'x1_data':mean_mono_loaded_regen_paretofront[:,0],'x1err_data':std_mono_loaded_regen_paretofront[:,0],
          'x2_data':mean_mono_loaded_paretofront[:,0],'x2err_data':std_mono_loaded_paretofront[:,0],
          'y1_data':mean_mono_loaded_regen_paretofront[:,1],'y1err_data':std_mono_loaded_regen_paretofront[:,1],
          'y2_data':mean_mono_loaded_paretofront[:,1],'y2err_data':std_mono_loaded_paretofront[:,1],
          'color_1':mycolors['lavender purple'],'color_2':mycolors['dark purple'],
          'legend_1':'loaded mono regen','legend_2':'loaded mono non-regen'
          }
fig = plt.figure(num='Pareto Curve: loaded mono',figsize=(10.4, 8.8))
utils.plot_pareto_avg_curve (plot_dic,loadcond='loaded',line=True)
plt.xlabel('Metabolic Cost Reduction (%)')
plt.ylabel('Exoskeleton Energy Consumption (W/kg)')
ax = plt.gca()
utils.no_top_right(ax)
plt.legend(loc='best',frameon=False)
fig.tight_layout()
fig.savefig('./Figures/Paretofront/Mean_Pareto/Pareto_Load_Mono_GenVsNonGen.pdf',orientation='landscape',bbox_inches='tight')
plt.show()

# average pareto curve: loaded bi non-regenerated vs regenerated

plot_dic = {'x1_data':mean_bi_loaded_regen_paretofront[:,0],'x1err_data':std_bi_loaded_regen_paretofront[:,0],
          'x2_data':mean_bi_loaded_paretofront[:,0],'x2err_data':std_bi_loaded_paretofront[:,0],
          'y1_data':mean_bi_loaded_regen_paretofront[:,1],'y1err_data':std_bi_loaded_regen_paretofront[:,1],
          'y2_data':mean_bi_loaded_paretofront[:,1],'y2err_data':std_bi_loaded_paretofront[:,1],
          'color_1':mycolors['crimson red'],'color_2':mycolors['olympic blue'],
          'legend_1':'loaded bi regen','legend_2':'loaded bi non-regen'
          }
fig = plt.figure(num='Pareto Curve: loaded bi',figsize=(10.4, 8.8))
utils.plot_pareto_avg_curve (plot_dic,loadcond='loaded',line=True)
plt.xlabel('Metabolic Cost Reduction (%)')
plt.ylabel('Exoskeleton Energy Consumption (W/kg)')
ax = plt.gca()
utils.no_top_right(ax)
plt.legend(loc='best',frameon=False)
fig.tight_layout()
fig.savefig('./Figures/Paretofront/Mean_Pareto/Pareto_Load_Bi_GenVsNonGen.pdf',orientation='landscape',bbox_inches='tight')
plt.show()

# average pareto curve: noload mono vs biarticular

plot_dic = {'x1_data':mean_bi_noload_regen_paretofront[:,0],'x1err_data':std_bi_noload_regen_paretofront[:,0],
          'x2_data':mean_mono_noload_regen_paretofront[:,0],'x2err_data':std_mono_noload_regen_paretofront[:,0],
          'y1_data':mean_bi_noload_regen_paretofront[:,1],'y1err_data':std_bi_noload_regen_paretofront[:,1],
          'y2_data':mean_mono_noload_regen_paretofront[:,1],'y2err_data':std_mono_noload_regen_paretofront[:,1],
          'color_1':mycolors['magenta pink'],'color_2':mycolors['lavender purple']
          }
fig = plt.figure(num='Pareto Curve: noload mono vs bi',figsize=(10.4, 8.8))
utils.plot_pareto_avg_curve (plot_dic,loadcond='noload',line=True)
plt.xlabel('Metabolic Cost Reduction (%)')
plt.ylabel('Exoskeleton Energy Consumption (W/kg)')
ax = plt.gca()
utils.no_top_right(ax)
plt.legend(loc='best',frameon=False)
fig.tight_layout()
fig.savefig('./Figures/Paretofront/Mean_Pareto/Pareto_Noload_BiVsMono_Regenerated.pdf',orientation='landscape',bbox_inches='tight')
plt.show()

# average pareto curve: noload mono non-regenerated vs regenerated

plot_dic = {'x1_data':mean_mono_noload_regen_paretofront[:,0],'x1err_data':std_mono_noload_regen_paretofront[:,0],
          'x2_data':mean_mono_noload_paretofront[:,0],'x2err_data':std_mono_noload_paretofront[:,0],
          'y1_data':mean_mono_noload_regen_paretofront[:,1],'y1err_data':std_mono_noload_regen_paretofront[:,1],
          'y2_data':mean_mono_noload_paretofront[:,1],'y2err_data':std_mono_noload_paretofront[:,1],
          'color_1':mycolors['lavender purple'],'color_2':mycolors['dark purple'],
          'legend_1':'noload mono regen','legend_2':'noload mono non-regen'
          }
fig = plt.figure(num='Pareto Curve: noload mono',figsize=(10.4, 8.8))
utils.plot_pareto_avg_curve (plot_dic,loadcond='noload',line=True)
plt.xlabel('Metabolic Cost Reduction (%)')
plt.ylabel('Exoskeleton Energy Consumption (W/kg)')
ax = plt.gca()
utils.no_top_right(ax)
plt.legend(loc='best',frameon=False)
fig.tight_layout()
fig.savefig('./Figures/Paretofront/Mean_Pareto/Pareto_Noload_Mono_GenVsNonGen.pdf',orientation='landscape',bbox_inches='tight')
plt.show()

# average pareto curve: noload bi non-regenerated vs regenerated

plot_dic = {'x1_data':mean_bi_noload_regen_paretofront[:,0],'x1err_data':std_bi_noload_regen_paretofront[:,0],
          'x2_data':mean_bi_noload_paretofront[:,0],'x2err_data':std_bi_noload_paretofront[:,0],
          'y1_data':mean_bi_noload_regen_paretofront[:,1],'y1err_data':std_bi_noload_regen_paretofront[:,1],
          'y2_data':mean_bi_noload_paretofront[:,1],'y2err_data':std_bi_noload_paretofront[:,1],
          'color_1':mycolors['crimson red'],'color_2':mycolors['olympic blue'],
          'legend_1':'noload bi regen','legend_2':'noload bi non-regen'
          }
fig = plt.figure(num='Pareto Curve: noload bi',figsize=(10.4, 8.8))
utils.plot_pareto_avg_curve (plot_dic,loadcond='noload',line=True)
plt.xlabel('Metabolic Cost Reduction (%)')
plt.ylabel('Exoskeleton Energy Consumption (W/kg)')
ax = plt.gca()
utils.no_top_right(ax)
plt.legend(loc='best',frameon=False)
fig.tight_layout()
fig.savefig('./Figures/Paretofront/Mean_Pareto/Pareto_Noload_Bi_GenVsNonGen.pdf',orientation='landscape',bbox_inches='tight')
plt.show()
'''
