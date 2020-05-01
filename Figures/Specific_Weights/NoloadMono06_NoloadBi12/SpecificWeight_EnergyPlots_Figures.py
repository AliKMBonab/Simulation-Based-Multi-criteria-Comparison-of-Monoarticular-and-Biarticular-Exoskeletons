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
gl_noload = {'noload_subject{}_trial{}'.format(i,j): utils.construct_gl_mass_side(subjectno=i,trialno=j,loadcond='noload') for i in subjects for j in trials_num}
#####################################################################################
# Processing Data
# toe-off
noload_mean_toe_off,_,noload_mean_toe_off,_ = utils.toe_off_avg_std(gl_noload,gl_noload)
# Energy
noload_bi_metabolics = utils.reduction_calc(unassisted_energy_dataset['noload_metabolics_energy'],assisted_energy_dataset['biarticular_hip50knee60_noload_metabolics_energy'])
noload_mono_metabolics = utils.reduction_calc(unassisted_energy_dataset['noload_metabolics_energy'],assisted_energy_dataset['monoarticular_hip60knee70_noload_metabolics_energy'])
#  Mean/std energy percent
mean_noload_bi_metabolics, std_noload_bi_metabolics = utils.mean_std_over_subjects(noload_bi_metabolics,ax=0)
mean_noload_mono_metabolics, std_noload_mono_metabolics = utils.mean_std_over_subjects(noload_mono_metabolics,ax=0)
# Mean/std energy
mean_noload_unassist_energy,std_noload_unassist_energy = utils.mean_std_over_subjects(unassisted_energy_dataset['noload_metabolics_energy'],ax=0)
mean_noload_bi_energy,std_noload_bi_energy = utils.mean_std_over_subjects(assisted_energy_dataset['biarticular_hip50knee60_noload_metabolics_energy'],ax=0)
mean_noload_mono_energy,std_noload_mono_energy = utils.mean_std_over_subjects(assisted_energy_dataset['monoarticular_hip60knee70_noload_metabolics_energy'],ax=0)

#####################################################################################
# Plots
# Biarticular VS Monoarticular
names = ['unassist, noload','mono, noload','bi, noload']
data = [mean_noload_unassist_energy,mean_noload_mono_energy,mean_noload_bi_energy]
err = [std_noload_unassist_energy,std_noload_mono_energy,std_noload_bi_energy]
fig, ax = plt.subplots(figsize=(6.4, 4.8))
barplot = ax.bar(names, data, yerr=err, align='center',color=mycolors['manatee grey'],width=0.45, ecolor='black', capsize=10)
barplot[1].set_color(mycolors['olympic blue'])
barplot[2].set_color(mycolors['crimson red'])
ax.set_ylabel('Metabolic Rate (W/Kg)')
ax.set_xticks(names)
ax.set_xticklabels(names)
utils.no_top_right(ax)
fig.savefig('./Figures/Specific_Weights/NoloadMono06_NoloadBi12/Metabolic_Rate_BarPlot.pdf',orientation='landscape',bbox_inches='tight')

#******************************************************************
# Metabolic Bar Plot
# noload Mono Vs Bi
names = ['bi, noload','mono, noload']
data = [mean_noload_bi_metabolics,mean_noload_mono_metabolics]
err = [std_noload_bi_metabolics,std_noload_mono_metabolics]
fig, ax = plt.subplots(figsize=(6.4, 4.8))
barplot = ax.bar(names, data, yerr=err, align='center',color=mycolors['manatee grey'],width=0.45, ecolor='black', capsize=10)
barplot[1].set_color(mycolors['dark purple'])
barplot[0].set_color(mycolors['crimson red'])
ax.set_ylabel('Metabolic Change (%)')
ax.set_xticks(names)
ax.set_xticklabels(names)
utils.no_top_right(ax)
fig.tight_layout()
plt.show()
fig.savefig('./Figures/Specific_Weights/NoloadMono06_NoloadBi12/Metabolic_Percent_ParPlot.pdf',orientation='landscape',bbox_inches='tight')


#******************************************************************
# Metabolic Box Plot
# Biarticular

names = ['unassist, noload','mono, noload','bi, noload']
x = np.arange(1,len(names)+1,1)
data = [utils.mean_over_trials(unassisted_energy_dataset['noload_metabolics_energy']),\
        utils.mean_over_trials(assisted_energy_dataset['monoarticular_hip60knee70_noload_metabolics_energy']),\
        utils.mean_over_trials(assisted_energy_dataset['biarticular_hip50knee60_noload_metabolics_energy'])]
fig, ax = plt.subplots(figsize=(6.4, 4.8))
bp = ax.boxplot(data, patch_artist=True)
utils.beautiful_boxplot(bp)
ax.set_ylabel('Metabolic Rate (W/Kg)')
ax.set_xticks(x)
ax.set_xticklabels(names)
utils.no_top_right(ax)
fig.tight_layout()
plt.show()
fig.savefig('./Figures/Specific_Weights/NoloadMono06_NoloadBi12/Metabolic_Rate_BoxPlot.pdf',orientation='landscape',bbox_inches='tight')


#******************************************************************
# Actuators Energy Box Plot
# Biarticular Vs Monoarticular noload
names = ['bi hip, noload','bi knee, noload','mono hip, noload','mono knee, noload',]
x = np.arange(1,len(names)+1,1)
data = [utils.mean_over_trials(assisted_energy_dataset['biarticular_hip50knee60_noload_hipactuator_energy']),\
        utils.mean_over_trials(assisted_energy_dataset['biarticular_hip50knee60_noload_kneeactuator_energy']),\
        utils.mean_over_trials(assisted_energy_dataset['monoarticular_hip60knee70_noload_hipactuator_energy']),\
        utils.mean_over_trials(assisted_energy_dataset['monoarticular_hip60knee70_noload_kneeactuator_energy'])]
fig, ax = plt.subplots(figsize=(6.4, 4.8))
bp = ax.boxplot(data, patch_artist=True)
utils.beautiful_boxplot(bp)
ax.set_ylabel('Actuator Energy Rate (W/Kg)')
ax.set_xticks(x)
ax.set_xticklabels(names)
utils.no_top_right(ax)
fig.tight_layout()
plt.show()
fig.savefig('./Figures/Specific_Weights/NoloadMono06_NoloadBi12/Actuator_Energy_BoxPlot.pdf',orientation='landscape',bbox_inches='tight')

#******************************************************************
# Paper Figure
# Metabolic Box Plot
# Biarticular

names = ['unassist, noload','mono, noload','bi, noload']
x = np.arange(1,len(names)+1,1)
data = [utils.mean_over_trials(unassisted_energy_dataset['noload_metabolics_energy']),\
        utils.mean_over_trials(assisted_energy_dataset['monoarticular_hip60knee70_noload_metabolics_energy']),\
        utils.mean_over_trials(assisted_energy_dataset['biarticular_hip50knee60_noload_metabolics_energy'])]
fig= plt.figure(figsize=(9.6, 4.8))
plt.subplot(1,2,1)
bp = plt.boxplot(data, patch_artist=True)
ax = plt.gca()
utils.beautiful_boxplot(bp)
ax.set_ylabel('Metabolic Rate (W/Kg)')
ax.set_xticks(x)
ax.set_ylim((4,8))
plt.tick_params(axis='both',direction='in')
ax.set_xticklabels(names)
utils.no_top_right(ax)


#******************************************************************
# Actuators Energy Box Plot
# Biarticular Vs Monoarticular noload
names = ['bi hip,\nnoload','bi knee,\nnoload','mono hip,\nnoload','mono knee,\nnoload',]
x = np.arange(1,len(names)+1,1)
data = [utils.mean_over_trials(assisted_energy_dataset['biarticular_hip50knee60_noload_hipactuator_energy']),\
        utils.mean_over_trials(assisted_energy_dataset['biarticular_hip50knee60_noload_kneeactuator_energy']),\
        utils.mean_over_trials(assisted_energy_dataset['monoarticular_hip60knee70_noload_hipactuator_energy']),\
        utils.mean_over_trials(assisted_energy_dataset['monoarticular_hip60knee70_noload_kneeactuator_energy'])]
plt.subplot(1,2,2)
bp = plt.boxplot(data, patch_artist=True)
ax = plt.gca()
utils.beautiful_boxplot(bp)
ax.set_ylabel('Actuators Absolute\nPower (W/Kg)')
ax.set_xticks(x)
ax.set_ylim((0.6,2))
plt.tick_params(axis='both',direction='in')
ax.set_xticklabels(names)
utils.no_top_right(ax)
fig.tight_layout(h_pad=-1, w_pad=-1.5)
fig.subplots_adjust(top=0.98, bottom=0.075, left=0.100, right=0.975,hspace=0.35,wspace=0.15)
fig.savefig('./Figures/Specific_Weights/NoloadMono06_NoloadBi12/PaperFigure_BoxPlot.pdf',orientation='landscape',bbox_inches='tight')
plt.show()
