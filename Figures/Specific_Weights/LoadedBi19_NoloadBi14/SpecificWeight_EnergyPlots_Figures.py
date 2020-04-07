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
# assisted subjects energy dataset
directory = 'D:/Ali.K.M.Bonab/Walking_Mass_Inertia_Effect/Data/Data/Specific_Weights/*_energy.csv'
files = enumerate(glob.iglob(directory), 1)
assisted_energy_dataset = {pathlib.PurePath(f[1]).stem: np.loadtxt(f[1], delimiter=',') for f in files}
# unassisted subjects energy dataset
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
# Energy
unassist_loadedvsnoload_metabolics = utils.reduction_calc(unassisted_energy_dataset['loaded_metabolics_energy'],unassisted_energy_dataset['noload_metabolics_energy'])
loaded_bi_metabolics = utils.reduction_calc(unassisted_energy_dataset['loaded_metabolics_energy'],assisted_energy_dataset['biarticular_hip40knee40_load_metabolics_energy'])
noload_bi_metabolics = utils.reduction_calc(unassisted_energy_dataset['noload_metabolics_energy'],assisted_energy_dataset['biarticular_hip50knee40_noload_metabolics_energy'])
bi_loadedvsnoload_metabolics = noload_bi_metabolics - loaded_bi_metabolics
#  Mean/std energy percent
mean_unassist_loadedvsnoload_metabolics, std_unassist_loadedvsnoload_metabolics = utils.mean_std_over_subjects(unassist_loadedvsnoload_metabolics,ax=0)
mean_loaded_bi_metabolics, std_loaded_bi_metabolics = utils.mean_std_over_subjects(loaded_bi_metabolics,ax=0)
mean_noload_bi_metabolics, std_noload_bi_metabolics = utils.mean_std_over_subjects(noload_bi_metabolics,ax=0)
mean_bi_loadedvsnoload_metabolics, std_bi_loadedvsnoload_metabolics = utils.mean_std_over_subjects(bi_loadedvsnoload_metabolics,ax=0)
# Mean/std energy
mean_noload_unassist_energy,std_noload_unassist_energy = utils.mean_std_over_subjects(unassisted_energy_dataset['noload_metabolics_energy'],ax=0)
mean_loaded_unassist_energy,std_loaded_unassist_energy = utils.mean_std_over_subjects(unassisted_energy_dataset['loaded_metabolics_energy'],ax=0)
mean_noload_bi_energy,std_noload_bi_energy = utils.mean_std_over_subjects(assisted_energy_dataset['biarticular_hip50knee40_noload_metabolics_energy'],ax=0)
mean_loaded_bi_energy,std_loaded_bi_energy = utils.mean_std_over_subjects(assisted_energy_dataset['biarticular_hip40knee40_load_metabolics_energy'],ax=0)

#####################################################################################
# Plots
# Biarticular
names = ['unassist, noload','unassist, loaded','bi, noload','bi, loaded']
data = [mean_noload_unassist_energy,mean_loaded_unassist_energy,mean_noload_bi_energy,mean_loaded_bi_energy]
err = [std_noload_unassist_energy,std_loaded_unassist_energy,std_noload_bi_energy,std_loaded_bi_energy]
fig, ax = plt.subplots(figsize=(6.4, 4.8))
barplot = ax.bar(names, data, yerr=err, align='center',color=mycolors['manatee grey'],width=0.35, ecolor='black', capsize=10)
barplot[0].set_color(mycolors['dark spring green'])
barplot[2].set_color(mycolors['magenta pink'])
barplot[3].set_color(mycolors['crimson red'])
ax.set_ylabel('Metabolic Rate (W/Kg)')
ax.set_xticks(names)
ax.set_xticklabels(names)
utils.no_top_right(ax)
plt.show()
fig.savefig('./Figures/Specific_Weights/LoadedBi19_NoloadBi14/Metabolics_Energy_BarPlot.pdf',orientation='landscape',bbox_inches='tight')

#******************************************************************
# Metabolic Bar Plot
# Biarticular Load Vs Noload
names = ['bi, loaded','bi, noload']
data = [mean_loaded_bi_metabolics,mean_noload_bi_metabolics]
err = [std_loaded_bi_metabolics,std_noload_bi_metabolics]
fig, ax = plt.subplots(figsize=(6.4, 4.8))
barplot = ax.bar(names, data, yerr=err, align='center',color=mycolors['manatee grey'],width=0.35, ecolor='black', capsize=10)
barplot[0].set_color(mycolors['crimson red'])
barplot[1].set_color(mycolors['magenta pink'])
ax.set_ylabel('Metabolic Change (%)')
ax.set_xticks(names)
ax.set_xticklabels(names)
utils.no_top_right(ax)
plt.show()
fig.savefig('./Figures/Specific_Weights/LoadedBi19_NoloadBi14/Metabolics_Reduction_BarPlot.pdf',orientation='landscape',bbox_inches='tight')

#******************************************************************
# Metabolic Box Plot
# Biarticular
names = ['unassist, loaded','bi, loaded','unassist, noload','bi, noload']
x = np.arange(1,len(names)+1,1)
data = [unassisted_energy_dataset['loaded_metabolics_energy'],assisted_energy_dataset['biarticular_hip40knee40_load_metabolics_energy'][~np.isnan(assisted_energy_dataset['biarticular_hip40knee40_load_metabolics_energy'])],\
        unassisted_energy_dataset['noload_metabolics_energy'],assisted_energy_dataset['biarticular_hip50knee40_noload_metabolics_energy']]
fig, ax = plt.subplots(figsize=(6.4, 4.8))
bp = ax.boxplot(data, patch_artist=True)
utils.beautiful_boxplot(bp)
ax.set_ylabel('Metabolic Rate (W/Kg)')
ax.set_xticks(x)
ax.set_xticklabels(names)
utils.no_top_right(ax)
plt.show()
fig.savefig('./Figures/Specific_Weights/LoadedBi19_NoloadBi14/Metabolics_Energy_BoxPlot.pdf',orientation='landscape',bbox_inches='tight')

#******************************************************************
# Actuators Energy Box Plot

# Biarticular Loaded Vs Noload
names = ['bi hip, loaded','bi knee, loaded','bi hip, noload','bi knee, noload',]
x = np.arange(1,len(names)+1,1)
data = [assisted_energy_dataset['biarticular_hip40knee40_load_hipactuator_energy'][~np.isnan(assisted_energy_dataset['biarticular_hip40knee40_load_hipactuator_energy'])],assisted_energy_dataset['biarticular_hip40knee40_load_kneeactuator_energy'][~np.isnan(assisted_energy_dataset['biarticular_hip40knee40_load_kneeactuator_energy'])],\
        assisted_energy_dataset['biarticular_hip50knee40_noload_hipactuator_energy'],assisted_energy_dataset['biarticular_hip50knee40_noload_kneeactuator_energy']]
fig, ax = plt.subplots(figsize=(6.4, 4.8))
bp = ax.boxplot(data, patch_artist=True)
utils.beautiful_boxplot(bp)
ax.set_ylabel('Actuator Energy Rate (W/Kg)')
ax.set_xticks(x)
ax.set_xticklabels(names)
utils.no_top_right(ax)
plt.show()
fig.savefig('./Figures/Specific_Weights/LoadedBi19_NoloadBi14/Actuator_Energy_BoxPlot.pdf',orientation='landscape',bbox_inches='tight')
