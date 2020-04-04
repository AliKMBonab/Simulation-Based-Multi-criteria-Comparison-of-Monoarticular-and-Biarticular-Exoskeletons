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
# pareto exo energy dataset
directory = 'D:/Ali.K.M.Bonab/Walking_Mass_Inertia_Effect/Data/Data/Pareto/*_energy.csv'
files = enumerate(glob.iglob(directory), 1)
assisted_energy_dataset = {pathlib.PurePath(f[1]).stem: np.loadtxt(f[1], delimiter=',') for f in files}
# unassist energy dataset
directory = 'D:/Ali.K.M.Bonab/Walking_Mass_Inertia_Effect/Data/Data/Unassist/*_energy.csv'
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

# metabolics cost reduction percents
bi_loaded_metabolics_energy = np.reshape(assisted_energy_dataset['biarticular_pareto_load_metabolics_energy'],(25,21),order='F')
bi_noload_metabolics_energy = np.reshape(assisted_energy_dataset['biarticular_pareto_noload_metabolics_energy'],(25,21),order='F')
mono_loaded_metabolics_energy = np.reshape(assisted_energy_dataset['monoarticular_pareto_load_metabolics_energy'],(25,21),order='F')
mono_noload_metabolics_energy = np.reshape(assisted_energy_dataset['monoarticular_pareto_noload_metabolics_energy'],(25,21),order='F')
unassist_noload_metabolics_energy = unassisted_energy_dataset['noload_metabolics_energy']
unassist_loaded_metabolics_energy = unassisted_energy_dataset['loaded_metabolics_energy']

# actuators energy
bi_loaded_hip_energy= np.reshape(assisted_energy_dataset['biarticular_pareto_load_hipactuator_energy'],(25,21),order='F')
bi_loaded_knee_energy= np.reshape(assisted_energy_dataset['biarticular_pareto_load_kneeactuator_energy'],(25,21),order='F')
bi_loaded_energy = bi_loaded_hip_energy + bi_loaded_knee_energy
bi_noload_hip_energy= np.reshape(assisted_energy_dataset['biarticular_pareto_noload_hipactuator_energy'],(25,21),order='F')
bi_noload_knee_energy= np.reshape(assisted_energy_dataset['biarticular_pareto_noload_kneeactuator_energy'],(25,21),order='F')
bi_noload_energy = bi_noload_hip_energy + bi_noload_knee_energy
mono_loaded_hip_energy= np.reshape(assisted_energy_dataset['monoarticular_pareto_load_hipactuator_energy'],(25,21),order='F')
mono_loaded_knee_energy= np.reshape(assisted_energy_dataset['monoarticular_pareto_load_kneeactuator_energy'],(25,21),order='F')
mono_loaded_energy = mono_loaded_hip_energy + mono_loaded_knee_energy
mono_noload_hip_energy= np.reshape(assisted_energy_dataset['monoarticular_pareto_noload_hipactuator_energy'],(25,21),order='F')
mono_noload_knee_energy= np.reshape(assisted_energy_dataset['monoarticular_pareto_noload_kneeactuator_energy'],(25,21),order='F')
mono_noload_energy = mono_noload_hip_energy + mono_noload_knee_energy

#####################################################################################
# Paretofront
bi_loaded_metabolics_percent_Paretofront,\
bi_loaded_energy_Paretofront = utils.paretofront_subjects(bi_loaded_metabolics_energy,bi_loaded_energy,unassist_loaded_metabolics_energy)
bi_noload_metabolics_percent_Paretofront,\
bi_noload_energy_Paretofront = utils.paretofront_subjects(bi_noload_metabolics_energy,bi_noload_energy,unassist_noload_metabolics_energy)
mono_loaded_metabolics_percent_Paretofront,\
mono_loaded_energy_Paretofront = utils.paretofront_subjects(mono_loaded_metabolics_energy,mono_loaded_energy,unassist_loaded_metabolics_energy)
mono_noload_metabolics_percent_Paretofront,\
mono_noload_energy_Paretofront = utils.paretofront_subjects(mono_noload_metabolics_energy,mono_noload_energy,unassist_noload_metabolics_energy)

#####################################################################################
# plots
# subjects pareto curve: loaded mono vs biarticular
fig = plt.figure(num='Pareto Curve: loaded mono vs bi',figsize=(12.4, 18.8))
plot_dic = {'x1_data':bi_loaded_metabolics_percent_Paretofront,'x2_data':mono_loaded_metabolics_percent_Paretofront,
          'y1_data':bi_loaded_energy_Paretofront,'y2_data':mono_loaded_energy_Paretofront,
          'color_1':mycolors['crimson red'],'color_2':mycolors['dark purple'],
          'ylabel':'Energy Consumption (W/kg)','xlabel':'Metabolic Reduction (%)'
          }
utils.plot_pareto_curve_subjects (nrows=7,ncols=3,nplot=21,plot_dic=plot_dic,loadcond='loaded')
fig.tight_layout()
fig.savefig('./Figures/Paretofront/Subjects_Pareto/Pareto_Load_Subjects_BiVsMono.pdf',orientation='landscape',bbox_inches='tight')
plt.show()

# subjects pareto curve: noload mono vs biarticular

plot_dic = {'x1_data':bi_noload_metabolics_percent_Paretofront,'x2_data':mono_noload_metabolics_percent_Paretofront,
          'y1_data':bi_noload_energy_Paretofront,'y2_data':mono_noload_energy_Paretofront,
          'color_1':mycolors['crimson red'],'color_2':mycolors['dark purple'],
          'ylabel':'Energy Consumption (W/kg)','xlabel':'Metabolic Reduction (%)'
          }
fig = plt.figure(num='Pareto Curve: noload mono vs bi',figsize=(12.4, 18.8))
utils.plot_pareto_curve_subjects (nrows=7,ncols=3,nplot=21,plot_dic=plot_dic,loadcond='noload')
fig.tight_layout()
fig.savefig('./Figures/Paretofront/Subjects_Pareto/Pareto_Noload_Subjects_BiVsMono.pdf',orientation='landscape',bbox_inches='tight')
plt.show()
