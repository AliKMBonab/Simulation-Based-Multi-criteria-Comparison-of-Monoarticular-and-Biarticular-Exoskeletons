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
directory = 'D:/Ali.K.M.Bonab/Walking_Mass_Inertia_Effect/Data/Data/Ideal/*_energy.csv'
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
loaded_bi_metabolics = utils.reduction_calc(unassisted_energy_dataset['loaded_metabolics_energy'],assisted_energy_dataset['biarticular_ideal_loaded_metabolics_energy'])
noload_bi_metabolics = utils.reduction_calc(unassisted_energy_dataset['noload_metabolics_energy'],assisted_energy_dataset['biarticular_ideal_noload_metabolics_energy'])
loaded_mono_metabolics = utils.reduction_calc(unassisted_energy_dataset['loaded_metabolics_energy'],assisted_energy_dataset['monoarticular_ideal_loaded_metabolics_energy'])
noload_mono_metabolics = utils.reduction_calc(unassisted_energy_dataset['noload_metabolics_energy'],assisted_energy_dataset['monoarticular_ideal_noload_metabolics_energy'])
bi_loadedvsnoload_metabolics = noload_bi_metabolics - loaded_bi_metabolics
mono_loadedvsnoload_metabolics = noload_mono_metabolics - loaded_mono_metabolics
#  Mean/std metabolic percent
mean_unassist_loadedvsnoload_metabolics, std_unassist_loadedvsnoload_metabolics = utils.mean_std_over_subjects(unassist_loadedvsnoload_metabolics,ax=0)
mean_loaded_bi_metabolics, std_loaded_bi_metabolics = utils.mean_std_over_subjects(loaded_bi_metabolics,ax=0)
mean_noload_bi_metabolics, std_noload_bi_metabolics = utils.mean_std_over_subjects(noload_bi_metabolics,ax=0)
mean_loaded_mono_metabolics, std_loaded_mono_metabolics = utils.mean_std_over_subjects(loaded_mono_metabolics,ax=0)
mean_noload_mono_metabolics, std_noload_mono_metabolics = utils.mean_std_over_subjects(noload_mono_metabolics,ax=0)
mean_bi_loadedvsnoload_metabolics, std_bi_loadedvsnoload_metabolics = utils.mean_std_over_subjects(bi_loadedvsnoload_metabolics,ax=0)
mean_mono_loadedvsnoload_metabolics, std_mono_loadedvsnoload_metabolics = utils.mean_std_over_subjects(mono_loadedvsnoload_metabolics,ax=0)
# Mean/std metabolic energy
mean_noload_unassist_energy,std_noload_unassist_energy = utils.mean_std_over_subjects(unassisted_energy_dataset['noload_metabolics_energy'],ax=0)
mean_loaded_unassist_energy,std_loaded_unassist_energy = utils.mean_std_over_subjects(unassisted_energy_dataset['loaded_metabolics_energy'],ax=0)
mean_noload_bi_energy,std_noload_bi_energy = utils.mean_std_over_subjects(assisted_energy_dataset['biarticular_ideal_noload_metabolics_energy'],ax=0)
mean_loaded_bi_energy,std_loaded_bi_energy = utils.mean_std_over_subjects(assisted_energy_dataset['biarticular_ideal_loaded_metabolics_energy'],ax=0)
mean_noload_mono_energy,std_noload_mono_energy = utils.mean_std_over_subjects(assisted_energy_dataset['monoarticular_ideal_noload_metabolics_energy'],ax=0)
mean_loaded_mono_energy,std_loaded_mono_energy = utils.mean_std_over_subjects(assisted_energy_dataset['monoarticular_ideal_loaded_metabolics_energy'],ax=0)
# Mean/std actuators energy
# noload
mean_noload_bi_hip_energy,std_noload_bi_hip_energy = utils.mean_std_over_subjects(assisted_energy_dataset['biarticular_ideal_noload_hipactuator_energy'],ax=0)
mean_noload_bi_knee_energy,std_noload_bi_knee_energy = utils.mean_std_over_subjects(assisted_energy_dataset['biarticular_ideal_noload_kneeactuator_energy'],ax=0)
mean_noload_mono_hip_energy,std_noload_mono_hip_energy = utils.mean_std_over_subjects(assisted_energy_dataset['monoarticular_ideal_noload_hipactuator_energy'],ax=0)
mean_noload_mono_knee_energy,std_noload_mono_knee_energy = utils.mean_std_over_subjects(assisted_energy_dataset['monoarticular_ideal_noload_kneeactuator_energy'],ax=0)
# loaded
mean_loaded_bi_hip_energy,std_loaded_bi_hip_energy = utils.mean_std_over_subjects(assisted_energy_dataset['biarticular_ideal_loaded_hipactuator_energy'],ax=0)
mean_loaded_bi_knee_energy,std_loaded_bi_knee_energy = utils.mean_std_over_subjects(assisted_energy_dataset['biarticular_ideal_loaded_kneeactuator_energy'],ax=0)
mean_loaded_mono_hip_energy,std_loaded_mono_hip_energy = utils.mean_std_over_subjects(assisted_energy_dataset['monoarticular_ideal_loaded_hipactuator_energy'],ax=0)
mean_loaded_mono_knee_energy,std_loaded_mono_knee_energy = utils.mean_std_over_subjects(assisted_energy_dataset['monoarticular_ideal_loaded_kneeactuator_energy'],ax=0)

#####################################################################################
cellText = np.array([[mean_noload_bi_hip_energy,mean_noload_bi_knee_energy, mean_noload_bi_energy,mean_noload_bi_metabolics,\
                     std_noload_bi_hip_energy,std_noload_bi_knee_energy, std_noload_bi_energy,std_noload_bi_metabolics],\
                    [mean_noload_mono_hip_energy,mean_noload_mono_knee_energy, mean_noload_mono_energy,mean_noload_mono_metabolics,\
                     std_noload_mono_hip_energy,std_noload_mono_knee_energy, std_noload_mono_energy,std_noload_mono_metabolics],\
                    [mean_loaded_bi_hip_energy,mean_loaded_bi_knee_energy, mean_loaded_bi_energy,mean_loaded_bi_metabolics,\
                     std_loaded_bi_hip_energy,std_loaded_bi_knee_energy, std_loaded_bi_energy,std_loaded_bi_metabolics],\
                    [mean_loaded_mono_hip_energy,mean_loaded_mono_knee_energy, mean_loaded_mono_energy,mean_loaded_mono_metabolics,\
                     std_loaded_mono_hip_energy,std_loaded_mono_knee_energy, std_loaded_mono_energy,std_loaded_mono_metabolics]])
columns = ['biarticular, noload','monoarticular, noload','biarticular, loaded','monoarticular, loaded']
rows = ['mean hip actuator energy (J/kg)','mean knee actuator energy (J/kg)','mean metabolic rate (J/kg)','mean metabolic reduction (%)',\
           'std hip actuator energy (J/kg)','std knee actuator energy (J/kg)','std metabolic rate (J/kg)','std metabolic reduction (%)']
fig, ax = plt.subplots(figsize=(12,6))
table = ax.table(cellText=np.transpose(cellText.round(3)),rowLabels=rows,colLabels=columns,loc='center')
table.scale(1,2)
table.set_fontsize(15)
ax.axis('off')
fig.savefig('./Figures/Ideal/Energy_Table.pdf',orientation='landscape',bbox_inches='tight')
plt.show()
#####################################################################################
# Plots
# Biarticular
names = ['unassist, noload','unassist, loaded','bi, noload','bi, loaded']
data = [mean_noload_unassist_energy,mean_loaded_unassist_energy,mean_noload_bi_energy,mean_loaded_bi_energy]
err = [std_noload_unassist_energy,std_loaded_unassist_energy,std_noload_bi_energy,std_loaded_bi_energy]
fig, ax = plt.subplots(nrows=1,ncols=2,figsize=(12.4, 5.4))
barplot = ax[0].bar(names, data, yerr=err, align='center',color=mycolors['manatee grey'],width=0.45, ecolor='black', capsize=10)
barplot[0].set_color(mycolors['dark spring green'])
barplot[2].set_color(mycolors['magenta pink'])
barplot[3].set_color(mycolors['crimson red'])
ax[0].set_ylabel('Metabolic Rate (W/Kg)')
ax[0].set_xticks(names)
ax[0].set_xticklabels(names)
utils.no_top_right(ax[0])
# Monoarticular
names = ['unassist, noload','unassist, loaded','mono, noload','mono, loaded']
data = [mean_noload_unassist_energy,mean_loaded_unassist_energy,mean_noload_mono_energy,mean_loaded_mono_energy]
err = [std_noload_unassist_energy,std_loaded_unassist_energy,std_noload_mono_energy,std_loaded_mono_energy]
barplot = ax[1].bar(names, data, yerr=err, align='center',color=mycolors['manatee grey'],width=0.45, ecolor='black', capsize=10)
barplot[0].set_color(mycolors['dark spring green'])
barplot[2].set_color(mycolors['olympic blue'])
barplot[3].set_color(mycolors['dark purple'])
ax[1].set_ylabel('Metabolic Rate (W/Kg)')
ax[1].set_xticks(names)
ax[1].set_xticklabels(names)
utils.no_top_right(ax[1])
plt.show()
fig.tight_layout()
fig.savefig('./Figures/Ideal/Metabolic_Rate_BarPlot.pdf',orientation='landscape',bbox_inches='tight')

#******************************************************************
# Metabolic Bar Plot
# Loaded Mono Vs Bi
names = ['bi, loaded','mono, loaded']
data = [mean_loaded_bi_metabolics,mean_loaded_mono_metabolics]
err = [std_loaded_bi_metabolics,std_loaded_mono_metabolics]
fig, ax = plt.subplots(nrows=2,ncols=2,figsize=(12.4, 9.8))
barplot = ax[0,0].bar(names, data, yerr=err, align='center',color=mycolors['manatee grey'],width=0.45, ecolor='black', capsize=10)
barplot[1].set_color(mycolors['dark purple'])
barplot[0].set_color(mycolors['crimson red'])
ax[0,0].set_ylabel('Metabolic Change (%)')
ax[0,0].set_xticks(names)
ax[0,0].set_xticklabels(names)
utils.no_top_right(ax[0,0])
# Noload Mono vs Bi
names = ['bi, noload','mono, noload']
data = [mean_noload_bi_metabolics,mean_noload_mono_metabolics]
err = [std_noload_bi_metabolics,std_noload_mono_metabolics]
barplot = ax[0,1].bar(names, data, yerr=err, align='center',color=mycolors['manatee grey'],width=0.45, ecolor='black', capsize=10)
barplot[1].set_color(mycolors['olympic blue'])
barplot[0].set_color(mycolors['magenta pink'])
ax[0,1].set_ylabel('Metabolic Change (%)')
ax[0,1].set_xticks(names)
ax[0,1].set_xticklabels(names)
utils.no_top_right(ax[0,1])
# Biarticular Load Vs Noload
names = ['bi, loaded','bi, noload','loaded vs noload']
data = [mean_loaded_bi_metabolics,mean_noload_bi_metabolics,mean_bi_loadedvsnoload_metabolics]
err = [std_loaded_bi_metabolics,std_noload_bi_metabolics,std_bi_loadedvsnoload_metabolics]
barplot = ax[1,0].bar(names, data, yerr=err, align='center',color=mycolors['manatee grey'],width=0.45, ecolor='black', capsize=10)
barplot[0].set_color(mycolors['crimson red'])
barplot[1].set_color(mycolors['magenta pink'])
ax[1,0].set_ylabel('Metabolic Change (%)')
ax[1,0].set_xticks(names)
ax[1,0].set_xticklabels(names)
utils.no_top_right(ax[1,0])
# Monoarticular Load Vs Noload
names = ['mono, loaded','mono, noload','loaded vs noload']
data = [mean_loaded_mono_metabolics,mean_noload_mono_metabolics,mean_mono_loadedvsnoload_metabolics]
err = [std_loaded_mono_metabolics,std_noload_mono_metabolics,std_mono_loadedvsnoload_metabolics]
barplot = ax[1,1].bar(names, data, yerr=err, align='center',color=mycolors['manatee grey'],width=0.45, ecolor='black', capsize=10)
barplot[0].set_color(mycolors['dark purple'])
barplot[1].set_color(mycolors['olympic blue'])
ax[1,1].set_ylabel('Metabolic Change (%)')
ax[1,1].set_xticks(names)
ax[1,1].set_xticklabels(names)
utils.no_top_right(ax[1,1])
fig.tight_layout()
plt.show()
fig.savefig('./Figures/Ideal/Metabolic_Percent_ParPlot.pdf',orientation='landscape',bbox_inches='tight')


#******************************************************************
# Metabolic Box Plot
# Biarticular

names = ['unassist, loaded','bi, loaded','unassist, noload','bi, noload']
x = np.arange(1,len(names)+1,1)
data = [utils.mean_over_trials(unassisted_energy_dataset['loaded_metabolics_energy'],ax=0),utils.mean_over_trials(unassisted_energy_dataset['noload_metabolics_energy'],ax=0),\
        utils.mean_over_trials(assisted_energy_dataset['biarticular_ideal_loaded_metabolics_energy'],ax=0),utils.mean_over_trials(assisted_energy_dataset['biarticular_ideal_noload_metabolics_energy'],ax=0)]
fig, ax = plt.subplots(nrows=2,ncols=2,figsize=(13.4, 10.8))
bp = ax[0,0].boxplot(data, patch_artist=True)
utils.beautiful_boxplot(bp)
ax[0,0].set_ylabel('Metabolic Rate (W/Kg)')
ax[0,0].set_xticks(x)
ax[0,0].set_xticklabels(names)
ax[0,0].set_title('biarticular')
utils.no_top_right(ax[0,0])

# Monoarticular
names = ['unassist, loaded','mono, loaded','unassist, noload','mono, noload']
x = np.arange(1,len(names)+1,1)
data = [utils.mean_over_trials(unassisted_energy_dataset['loaded_metabolics_energy'],ax=0),utils.mean_over_trials(unassisted_energy_dataset['noload_metabolics_energy'],ax=0),\
        utils.mean_over_trials(assisted_energy_dataset['monoarticular_ideal_loaded_metabolics_energy'],ax=0),utils.mean_over_trials(assisted_energy_dataset['monoarticular_ideal_noload_metabolics_energy'],ax=0)]
bp = ax[0,1].boxplot(data, patch_artist=True)
utils.beautiful_boxplot(bp)
ax[0,1].set_ylabel('Metabolic Rate (W/Kg)')
ax[0,1].set_xticks(x)
ax[0,1].set_xticklabels(names)
ax[0,1].set_title('monoarticular')
utils.no_top_right(ax[0,1])

# Biarticular Vs Monoarticular Loaded
names = ['unassist, loaded','bi, loaded','mono, loaded']
x = np.arange(1,len(names)+1,1)
data = [utils.mean_over_trials(unassisted_energy_dataset['loaded_metabolics_energy'],ax=0),\
        utils.mean_over_trials(assisted_energy_dataset['biarticular_ideal_loaded_metabolics_energy'],ax=0),\
        utils.mean_over_trials(assisted_energy_dataset['monoarticular_ideal_loaded_metabolics_energy'],ax=0)]
bp = ax[1,0].boxplot(data, patch_artist=True)
utils.beautiful_boxplot(bp)
ax[1,0].set_ylabel('Metabolic Rate (W/Kg)')
ax[1,0].set_xticks(x)
ax[1,0].set_xticklabels(names)
ax[1,0].set_title('biarticular vs monoarticular, loaded')
utils.no_top_right(ax[1,0])
# Biarticular Vs Monoarticular Noloaded
names = ['unassist, noload','bi, noload','mono, noload']
x = np.arange(1,len(names)+1,1)
data = [utils.mean_over_trials(unassisted_energy_dataset['noload_metabolics_energy'],ax=0),\
        utils.mean_over_trials(assisted_energy_dataset['biarticular_ideal_noload_metabolics_energy'],ax=0),\
        utils.mean_over_trials(assisted_energy_dataset['monoarticular_ideal_noload_metabolics_energy'],ax=0)]
bp = ax[1,1].boxplot(data, patch_artist=True)
utils.beautiful_boxplot(bp)
ax[1,1].set_ylabel('Metabolic Rate (W/Kg)')
ax[1,1].set_xticks(x)
ax[1,1].set_xticklabels(names)
ax[1,1].set_title('biarticular vs monoarticular, noload')
utils.no_top_right(ax[1,1])

fig.tight_layout()
plt.show()
fig.savefig('./Figures/Ideal/Metabolic_Rate_BoxPlot.pdf',orientation='landscape',bbox_inches='tight')


#******************************************************************
# Actuators Energy Box Plot
# Biarticular Vs Monoarticular Loaded
names = ['bi hip, loaded','bi knee, loaded','mono hip, loaded','mono knee, loaded',]
x = np.arange(1,len(names)+1,1)
data = [utils.mean_over_trials(assisted_energy_dataset['biarticular_ideal_loaded_hipactuator_energy'],ax=0),utils.mean_over_trials(assisted_energy_dataset['biarticular_ideal_loaded_kneeactuator_energy'],ax=0),\
        utils.mean_over_trials(assisted_energy_dataset['monoarticular_ideal_loaded_hipactuator_energy'],ax=0),utils.mean_over_trials(assisted_energy_dataset['monoarticular_ideal_loaded_kneeactuator_energy'],ax=0)]
fig, ax = plt.subplots(nrows=2,ncols=2,figsize=(13.4, 10.8))
bp = ax[0,0].boxplot(data, patch_artist=True)
utils.beautiful_boxplot(bp)
ax[0,0].set_ylabel('Actuator Energy Rate (W/Kg)')
ax[0,0].set_xticks(x)
ax[0,0].set_xticklabels(names)
ax[0,0].set_title('biarticular vs monoarticular, loaded')
utils.no_top_right(ax[0,0])

# Biarticular Vs Monoarticular Noloaded
names = ['bi hip, noload','bi knee, noload','mono hip, noload','mono knee, noload',]
x = np.arange(1,len(names)+1,1)
data = [utils.mean_over_trials(assisted_energy_dataset['biarticular_ideal_noload_hipactuator_energy'],ax=0),utils.mean_over_trials(assisted_energy_dataset['biarticular_ideal_noload_kneeactuator_energy'],ax=0),\
        utils.mean_over_trials(assisted_energy_dataset['monoarticular_ideal_noload_hipactuator_energy'],ax=0),utils.mean_over_trials(assisted_energy_dataset['monoarticular_ideal_noload_kneeactuator_energy'],ax=0)]
bp = ax[0,1].boxplot(data, patch_artist=True)
utils.beautiful_boxplot(bp)
ax[0,1].set_ylabel('Actuator Energy Rate (W/Kg)')
ax[0,1].set_xticks(x)
ax[0,1].set_xticklabels(names)
ax[0,1].set_title('biarticular vs monoarticular, noload')
utils.no_top_right(ax[0,1])

# Biarticular Loaded Vs Noload
names = ['bi hip, loaded','bi knee, loaded','bi hip, noload','bi knee, noload',]
x = np.arange(1,len(names)+1,1)
data = [utils.mean_over_trials(assisted_energy_dataset['biarticular_ideal_loaded_hipactuator_energy'],ax=0),utils.mean_over_trials(assisted_energy_dataset['biarticular_ideal_loaded_kneeactuator_energy'],ax=0),\
        utils.mean_over_trials(assisted_energy_dataset['biarticular_ideal_noload_hipactuator_energy'],ax=0),utils.mean_over_trials(assisted_energy_dataset['biarticular_ideal_noload_kneeactuator_energy'],ax=0)]
bp = ax[1,0].boxplot(data, patch_artist=True)
utils.beautiful_boxplot(bp)
ax[1,0].set_ylabel('Actuator Energy Rate (W/Kg)')
ax[1,0].set_xticks(x)
ax[1,0].set_xticklabels(names)
ax[1,0].set_title('loaded vs noload, biarticular')
utils.no_top_right(ax[1,0])

# Monoarticular Loaded Vs Noload
names = ['mono hip, loaded','mono knee, loaded','mono hip, noload','mono knee, noload',]
x = np.arange(1,len(names)+1,1)
data = [utils.mean_over_trials(assisted_energy_dataset['monoarticular_ideal_loaded_hipactuator_energy'],ax=0),utils.mean_over_trials(assisted_energy_dataset['monoarticular_ideal_loaded_kneeactuator_energy'],ax=0),\
        utils.mean_over_trials(assisted_energy_dataset['monoarticular_ideal_noload_hipactuator_energy'],ax=0),utils.mean_over_trials(assisted_energy_dataset['monoarticular_ideal_noload_kneeactuator_energy'],ax=0)]
bp = ax[1,1].boxplot(data, patch_artist=True)
utils.beautiful_boxplot(bp)
ax[1,1].set_ylabel('Actuator Energy Rate (W/Kg)')
ax[1,1].set_xticks(x)
ax[1,1].set_xticklabels(names)
ax[1,1].set_title('loaded vs noload, monoarticular')
utils.no_top_right(ax[1,1])
fig.tight_layout()
plt.show()
fig.savefig('./Figures/Ideal/Actuator_Energy_BoxPlot.pdf',orientation='landscape',bbox_inches='tight')


########################################################################################
# Paper figure
plt.rcParams.update({'font.size': 12})
names = ['unassist, loaded','bi, loaded','unassist, noload','bi, noload']
x = np.arange(1,len(names)+1,1)
data = [utils.mean_over_trials(unassisted_energy_dataset['loaded_metabolics_energy'],ax=0),utils.mean_over_trials(unassisted_energy_dataset['noload_metabolics_energy'],ax=0),\
        utils.mean_over_trials(assisted_energy_dataset['biarticular_ideal_loaded_metabolics_energy'],ax=0),utils.mean_over_trials(assisted_energy_dataset['biarticular_ideal_noload_metabolics_energy'],ax=0)]
fig, ax = plt.subplots(nrows=2,ncols=2,figsize=(12.4, 10.8))
bp = ax[0,0].boxplot(data, patch_artist=True)
utils.beautiful_boxplot(bp)
ax[0,0].set_ylabel('Metabolic Rate (W/Kg)')
ax[0,0].set_xticks(x)
ax[0,0].set_xticklabels(names)
ax[0,0].set_title('biarticular, metabolic rate')
utils.no_top_right(ax[0,0])

# Monoarticular metabolics
names = ['unassist, loaded','mono, loaded','unassist, noload','mono, noload']
x = np.arange(1,len(names)+1,1)
data = [utils.mean_over_trials(unassisted_energy_dataset['loaded_metabolics_energy'],ax=0),utils.mean_over_trials(unassisted_energy_dataset['noload_metabolics_energy'],ax=0),\
        utils.mean_over_trials(assisted_energy_dataset['monoarticular_ideal_loaded_metabolics_energy'],ax=0),utils.mean_over_trials(assisted_energy_dataset['monoarticular_ideal_noload_metabolics_energy'],ax=0)]
bp = ax[0,1].boxplot(data, patch_artist=True)
utils.beautiful_boxplot(bp)
ax[0,1].set_ylabel('Metabolic Rate (W/Kg)')
ax[0,1].set_xticks(x)
ax[0,1].set_xticklabels(names)
ax[0,1].set_title('monoarticular, metabolic rate')
utils.no_top_right(ax[0,1])

# Biarticular Loaded Vs Noload
names = ['bi hip, loaded','bi knee, loaded','bi hip, noload','bi knee, noload',]
x = np.arange(1,len(names)+1,1)
data = [utils.mean_over_trials(assisted_energy_dataset['biarticular_ideal_loaded_hipactuator_energy'],ax=0),utils.mean_over_trials(assisted_energy_dataset['biarticular_ideal_loaded_kneeactuator_energy'],ax=0),\
        utils.mean_over_trials(assisted_energy_dataset['biarticular_ideal_noload_hipactuator_energy'],ax=0),utils.mean_over_trials(assisted_energy_dataset['biarticular_ideal_noload_kneeactuator_energy'],ax=0)]
bp = ax[1,0].boxplot(data, patch_artist=True)
utils.beautiful_boxplot(bp)
ax[1,0].set_ylabel('Actuator Energy (J/Kg)')
ax[1,0].set_xticks(x)
ax[1,0].set_xticklabels(names)
ax[1,0].set_title('biarticular, actutors energy')
utils.no_top_right(ax[1,0])

# Monoarticular Loaded Vs Noload
names = ['mono hip, loaded','mono knee, loaded','mono hip, noload','mono knee, noload',]
x = np.arange(1,len(names)+1,1)
data = [utils.mean_over_trials(assisted_energy_dataset['monoarticular_ideal_loaded_hipactuator_energy'],ax=0),utils.mean_over_trials(assisted_energy_dataset['monoarticular_ideal_loaded_kneeactuator_energy'],ax=0),\
        utils.mean_over_trials(assisted_energy_dataset['monoarticular_ideal_noload_hipactuator_energy'],ax=0),utils.mean_over_trials(assisted_energy_dataset['monoarticular_ideal_noload_kneeactuator_energy'],ax=0)]
bp = ax[1,1].boxplot(data, patch_artist=True)
utils.beautiful_boxplot(bp)
ax[1,1].set_ylabel('Actuator Energy (J/Kg)')
ax[1,1].set_xticks(x)
ax[1,1].set_xticklabels(names)
ax[1,1].set_title('monoarticular, actuators energy')
utils.no_top_right(ax[1,1])
fig.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
plt.show()
fig.savefig('./Figures/Ideal/Paper_Figure_Energy_BoxPlot.pdf',orientation='landscape',bbox_inches='tight')
