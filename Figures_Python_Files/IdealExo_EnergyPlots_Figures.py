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
rra_dataset = utils.csv2numpy('./Data/RRA/rra_final_data.csv') 
unassist_dataset = utils.csv2numpy('./Data/Unassist/unassist_final_data.csv') 
# assisted subjects energy dataset
directory = './Data/Ideal/*_energy.csv'
files = enumerate(glob.iglob(directory), 1)
assisted_energy_dataset = {pathlib.PurePath(f[1]).stem: np.loadtxt(f[1], delimiter=',') for f in files}
# unassisted subjects energy dataset
directory = './Data/Unassist/*_energy.csv'
files = enumerate(glob.iglob(directory), 1)
unassisted_energy_dataset = {pathlib.PurePath(f[1]).stem: np.loadtxt(f[1], delimiter=',') for f in files}
# unassist muscles metabolic rate
directory = './Data/Unassist/*_metabolic_rate.csv'
files = enumerate(glob.iglob(directory), 1)
unassist_musclesmetabolicrate_dataset = {pathlib.PurePath(f[1]).stem: np.loadtxt(f[1], delimiter=',') for f in files}
# assisted muscles metabolic rate
directory = './Data/Ideal/*_metabolic_rate.csv'
files = enumerate(glob.iglob(directory), 1)
assisted_musclesmetabolicrate_dataset = {pathlib.PurePath(f[1]).stem: np.loadtxt(f[1], delimiter=',') for f in files}

# gls
gl_noload = {'noload_subject{}_trial{}'.format(i,j): utils.construct_gl_mass_side(subjectno=i,trialno=j,loadcond='noload') for i in subjects for j in trials_num}
gl_loaded = {'loaded_subject{}_trial{}'.format(i,j): utils.construct_gl_mass_side(subjectno=i,trialno=j,loadcond='loaded') for i in subjects for j in trials_num}
#####################################################################################
# Processing Data
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
# total energy
mean_noload_mono_energy,std_noload_mono_energy = utils.mean_std_over_subjects(assisted_energy_dataset['monoarticular_ideal_noload_hipactuator_energy']+assisted_energy_dataset['monoarticular_ideal_noload_kneeactuator_energy'],ax=0)
mean_noload_bi_energy,std_noload_bi_energy = utils.mean_std_over_subjects(assisted_energy_dataset['biarticular_ideal_noload_hipactuator_energy']+assisted_energy_dataset['biarticular_ideal_noload_kneeactuator_energy'],ax=0)
# loaded
mean_loaded_bi_hip_energy,std_loaded_bi_hip_energy = utils.mean_std_over_subjects(assisted_energy_dataset['biarticular_ideal_loaded_hipactuator_energy'],ax=0)
mean_loaded_bi_knee_energy,std_loaded_bi_knee_energy = utils.mean_std_over_subjects(assisted_energy_dataset['biarticular_ideal_loaded_kneeactuator_energy'],ax=0)
mean_loaded_mono_hip_energy,std_loaded_mono_hip_energy = utils.mean_std_over_subjects(assisted_energy_dataset['monoarticular_ideal_loaded_hipactuator_energy'],ax=0)
mean_loaded_mono_knee_energy,std_loaded_mono_knee_energy = utils.mean_std_over_subjects(assisted_energy_dataset['monoarticular_ideal_loaded_kneeactuator_energy'],ax=0)
# total energy
mean_loaded_bi_energy,std_loaded_bi_energy = utils.mean_std_over_subjects(assisted_energy_dataset['biarticular_ideal_loaded_hipactuator_energy']+assisted_energy_dataset['biarticular_ideal_loaded_kneeactuator_energy'],ax=0)
mean_loaded_mono_energy,std_loaded_mono_energy = utils.mean_std_over_subjects(assisted_energy_dataset['monoarticular_ideal_loaded_hipactuator_energy']+assisted_energy_dataset['monoarticular_ideal_loaded_kneeactuator_energy'],ax=0)
#####################################################################################
# Plots
<<<<<<< HEAD
<<<<<<< HEAD
=======
=======
>>>>>>> paper
# muscles metabolic rate figure
# noload mono vs bi
fig = plt.figure(num='Muscles Metabolic Rate',figsize=(20, 20))
utils.muscles_whisker_bar_plot(unassist_musclesmetabolicrate_dataset['noload_muscles_metabolic_rate'],assisted_musclesmetabolicrate_dataset['monoarticular_ideal_noload_muscles_metabolic_rate'],
                               assisted_musclesmetabolicrate_dataset['biarticular_ideal_noload_muscles_metabolic_rate'],xticklabel=['unassist','mono','bi'] )
plt.legend(loc='best',frameon=False)
plt.show()
fig.tight_layout()
fig.tight_layout(h_pad=-1, w_pad=-1.5)
fig.subplots_adjust(top=0.98, bottom=0.075, left=0.100, right=0.975,hspace=0.45,wspace=0.40)
fig.savefig('./Figures/Ideal/MusclesMetabolicRate_Noload_MonoVsBi.pdf',orientation='landscape',bbox_inches='tight')

# loaded mono vs bi
fig = plt.figure(num='Muscles Metabolic Rate',figsize=(20, 20))
utils.muscles_whisker_bar_plot(unassist_musclesmetabolicrate_dataset['loaded_muscles_metabolic_rate'],assisted_musclesmetabolicrate_dataset['monoarticular_ideal_loaded_muscles_metabolic_rate'],
                               assisted_musclesmetabolicrate_dataset['biarticular_ideal_loaded_muscles_metabolic_rate'],xticklabel=['unassist','mono','bi'])
plt.legend(loc='best',frameon=False)
plt.show()
fig.tight_layout()
fig.tight_layout(h_pad=-1, w_pad=-1.5)
fig.subplots_adjust(top=0.98, bottom=0.075, left=0.100, right=0.975,hspace=0.45,wspace=0.40)
fig.savefig('./Figures/Ideal/MusclesMetabolicRate_Loaded_MonoVsBi.pdf',orientation='landscape',bbox_inches='tight')

# bi loaded vs noload
fig = plt.figure(num='Muscles Metabolic Rate',figsize=(20, 20))
utils.muscles_whisker_bar_plot(unassist_musclesmetabolicrate_dataset['loaded_muscles_metabolic_rate'],unassist_musclesmetabolicrate_dataset['noload_muscles_metabolic_rate'],
                                assisted_musclesmetabolicrate_dataset['biarticular_ideal_loaded_muscles_metabolic_rate'],
                                assisted_musclesmetabolicrate_dataset['biarticular_ideal_noload_muscles_metabolic_rate'],xticklabel=['unassist,\nloaded','unassist,\nnoload','bi,loaded','bi,noload'])
plt.legend(loc='best',frameon=False)
plt.show()
fig.tight_layout()
fig.tight_layout(h_pad=-1, w_pad=-1.5)
fig.subplots_adjust(top=0.98, bottom=0.075, left=0.100, right=0.975,hspace=0.45,wspace=0.40)
fig.savefig('./Figures/Ideal/MusclesMetabolicRate_Bi_LoadedVsNoload.pdf',orientation='landscape',bbox_inches='tight')

# mono loaded vs noload
fig = plt.figure(num='Muscles Metabolic Rate',figsize=(20, 20))
utils.muscles_whisker_bar_plot(unassist_musclesmetabolicrate_dataset['loaded_muscles_metabolic_rate'],unassist_musclesmetabolicrate_dataset['noload_muscles_metabolic_rate'],
                                assisted_musclesmetabolicrate_dataset['monoarticular_ideal_loaded_muscles_metabolic_rate'],
                                assisted_musclesmetabolicrate_dataset['monoarticular_ideal_noload_muscles_metabolic_rate'],xticklabel=['unassist,\nloaded','unassist,\nnoload','mono,loaded','mono,noload'])
plt.legend(loc='best',frameon=False)
plt.show()
fig.tight_layout()
fig.tight_layout(h_pad=-1, w_pad=-1.5)
fig.subplots_adjust(top=0.98, bottom=0.075, left=0.100, right=0.975,hspace=0.45,wspace=0.40)
fig.savefig('./Figures/Ideal/MusclesMetabolicRate_Mono_LoadedVsNoload.pdf',orientation='landscape',bbox_inches='tight')

<<<<<<< HEAD
>>>>>>> development
=======
>>>>>>> paper
#####################################################################################
# writing data to csv for adding the ideal exoskeletons on the paretofronts
headers = ['mean_noload_bi_hip_energy','std_noload_bi_hip_energy','mean_noload_bi_knee_energy','std_noload_bi_knee_energy',\
           'mean_noload_mono_hip_energy','std_noload_mono_hip_energy','mean_noload_mono_knee_energy','std_noload_mono_knee_energy',\
           'mean_loaded_bi_hip_energy','std_loaded_bi_hip_energy','mean_loaded_bi_knee_energy','std_loaded_bi_knee_energy',\
           'mean_loaded_mono_hip_energy','std_loaded_mono_hip_energy','mean_loaded_mono_knee_energy','std_loaded_mono_knee_energy',\
           'mean_noload_mono_energy','std_noload_mono_energy','mean_noload_bi_energy','std_noload_bi_energy',\
           'mean_loaded_mono_energy','std_loaded_mono_energy','mean_loaded_bi_energy','std_loaded_bi_energy',\
           'mean_noload_bi_metabolics', 'std_noload_bi_metabolics','mean_noload_mono_metabolics', 'std_noload_mono_metabolics',\
           'mean_loaded_bi_metabolics', 'std_loaded_bi_metabolics','mean_loaded_mono_metabolics', 'std_loaded_mono_metabolics']
dataset = np.c_[mean_noload_bi_hip_energy,std_noload_bi_hip_energy,mean_noload_bi_knee_energy,std_noload_bi_knee_energy,\
           mean_noload_mono_hip_energy,std_noload_mono_hip_energy,mean_noload_mono_knee_energy,std_noload_mono_knee_energy,\
           mean_loaded_bi_hip_energy,std_loaded_bi_hip_energy,mean_loaded_bi_knee_energy,std_loaded_bi_knee_energy,\
           mean_loaded_mono_hip_energy,std_loaded_mono_hip_energy,mean_loaded_mono_knee_energy,std_loaded_mono_knee_energy,\
           mean_noload_mono_energy,std_noload_mono_energy,mean_noload_bi_energy,std_noload_bi_energy,\
           mean_loaded_mono_energy,std_loaded_mono_energy,mean_loaded_bi_energy,std_loaded_bi_energy,\
           mean_noload_bi_metabolics, std_noload_bi_metabolics,mean_noload_mono_metabolics, std_noload_mono_metabolics,\
           mean_loaded_bi_metabolics, std_loaded_bi_metabolics,mean_loaded_mono_metabolics, std_loaded_mono_metabolics]
with open(r'.\Data\Ideal\ideal_exos_dataset.csv', 'wb') as f:
  f.write(bytes(utils.listToString(headers)+'\n','UTF-8'))
  np.savetxt(f, dataset, fmt='%s', delimiter=",")
# writing data to csv for statistical analyses
# general columns 
subjects = np.array(['subject05','subject07','subject09','subject10','subject11','subject12','subject14'])
loaded_unassist_col = np.repeat(np.array('loaded unassist'),7)
loaded_biarticular_col = np.repeat(np.array('loaded biarticular'),7)
loaded_monoarticular_col = np.repeat(np.array('loaded monoarticular'),7)
noload_unassist_col = np.repeat(np.array('noload unassist'),7)
noload_biarticular_col = np.repeat(np.array('noload biarticular'),7)
noload_monoarticular_col = np.repeat(np.array('noload monoarticular'),7)

# establishing dataset for metabolic rate
headers = ['subjects','assistance','metabolic rate 01','metabolic rate 02','metabolic rate 03']
subject_col = np.tile(subjects,6)
assistance_col = np.concatenate((loaded_unassist_col,loaded_monoarticular_col,loaded_biarticular_col,
                                 noload_unassist_col,noload_monoarticular_col,noload_biarticular_col),axis=0)
metabolic_rate_data = np.concatenate((np.reshape(unassisted_energy_dataset['loaded_metabolics_energy'],(7,3)),\
                                      np.reshape(assisted_energy_dataset['monoarticular_ideal_loaded_metabolics_energy'],(7,3)),\
                                      np.reshape(assisted_energy_dataset['biarticular_ideal_loaded_metabolics_energy'],(7,3)),\
                                      np.reshape(unassisted_energy_dataset['noload_metabolics_energy'],(7,3)),\
                                      np.reshape(assisted_energy_dataset['monoarticular_ideal_noload_metabolics_energy'],(7,3)),\
                                      np.reshape(assisted_energy_dataset['biarticular_ideal_noload_metabolics_energy'],(7,3))),axis=0)
final_dataset = np.column_stack([assistance_col,metabolic_rate_data])
final_dataset = np.column_stack([subject_col,final_dataset])
with open(r'.\Statistics\Ideal\MetabolicRate_Dataset.csv', 'wb') as f:
  f.write(bytes(utils.listToString(headers)+'\n','UTF-8'))
  np.savetxt(f, final_dataset, fmt='%s', delimiter=",")

#****************************************************************
#establishing dataset for assistive actuators average total power 
headers = ['subjects','assistive actuator','avg total power 01','avg total power 02','avg total power 03']
loaded_biarticular_hip_col = np.repeat(np.array('loaded biarticular hip actuator'),7)
loaded_monoarticular_hip_col = np.repeat(np.array('loaded monoarticular hip actuator'),7)
loaded_biarticular_knee_col = np.repeat(np.array('loaded biarticular knee actuator'),7)
loaded_monoarticular_knee_col = np.repeat(np.array('loaded monoarticular knee actuator'),7)
noload_biarticular_hip_col = np.repeat(np.array('noload biarticular hip actuator'),7)
noload_monoarticular_hip_col = np.repeat(np.array('noload monoarticular hip actuator'),7)
noload_biarticular_knee_col = np.repeat(np.array('noload biarticular knee actuator'),7)
noload_monoarticular_knee_col = np.repeat(np.array('noload monoarticular knee actuator'),7)
subject_col = np.tile(subjects,8)
assistive_actuators_col = np.concatenate((loaded_biarticular_hip_col,loaded_biarticular_knee_col,loaded_monoarticular_hip_col,loaded_monoarticular_knee_col,
                                          noload_biarticular_hip_col,noload_biarticular_knee_col,noload_monoarticular_hip_col,noload_monoarticular_knee_col),axis=0)
assistive_actuators_avg_totalpower_data = np.concatenate((np.reshape(assisted_energy_dataset['biarticular_ideal_loaded_hipactuator_energy'],(7,3)),\
                                                        np.reshape(assisted_energy_dataset['biarticular_ideal_loaded_kneeactuator_energy'],(7,3)),\
                                                        np.reshape(assisted_energy_dataset['monoarticular_ideal_loaded_hipactuator_energy'],(7,3)),\
                                                        np.reshape(assisted_energy_dataset['monoarticular_ideal_loaded_kneeactuator_energy'],(7,3)),\
                                                        np.reshape(assisted_energy_dataset['biarticular_ideal_noload_hipactuator_energy'],(7,3)),\
                                                        np.reshape(assisted_energy_dataset['biarticular_ideal_noload_kneeactuator_energy'],(7,3)),\
                                                        np.reshape(assisted_energy_dataset['monoarticular_ideal_noload_hipactuator_energy'],(7,3)),\
                                                        np.reshape(assisted_energy_dataset['monoarticular_ideal_noload_kneeactuator_energy'],(7,3))),axis=0)
final_dataset = np.column_stack([assistive_actuators_col,assistive_actuators_avg_totalpower_data])
final_dataset = np.column_stack([subject_col,final_dataset])
with open(r'.\Statistics\Ideal\ActuatorsAvgPower_Dataset.csv', 'wb') as f:
  f.write(bytes(utils.listToString(headers)+'\n','UTF-8'))
  np.savetxt(f, final_dataset, fmt='%s', delimiter=",")
#####################################################################################
# The contribution of the actuators
#####################################################################################
# monoarticular versus biarticular
# bi noload actuators contributions
mean_bi_noload_hip_contribution, std_bi_noload_hip_contribution,\
mean_bi_noload_knee_contribution, std_bi_noload_knee_contribution,_,_\
 = utils.actuators_energy_contribution(assisted_energy_dataset['biarticular_ideal_noload_hipactuator_energy'],assisted_energy_dataset['biarticular_ideal_noload_kneeactuator_energy'])
# mono noload actuators contributions
mean_mono_noload_hip_contribution, std_mono_noload_hip_contribution,\
mean_mono_noload_knee_contribution, std_mono_noload_knee_contribution,_,_\
 = utils.actuators_energy_contribution(assisted_energy_dataset['monoarticular_ideal_noload_hipactuator_energy'],assisted_energy_dataset['monoarticular_ideal_noload_kneeactuator_energy'],\
                                        assisted_energy_dataset['biarticular_ideal_noload_hipactuator_energy']+assisted_energy_dataset['biarticular_ideal_noload_kneeactuator_energy'])
# bi loaded actuators contributions
mean_bi_loaded_hip_contribution, std_bi_loaded_hip_contribution,\
mean_bi_loaded_knee_contribution, std_bi_loaded_knee_contribution,_,_\
 = utils.actuators_energy_contribution(assisted_energy_dataset['biarticular_ideal_loaded_hipactuator_energy'],assisted_energy_dataset['biarticular_ideal_loaded_kneeactuator_energy'])
# mono vs bi loaded actuators contributions
mean_mono_loaded_hip_contribution, std_mono_loaded_hip_contribution,\
mean_mono_loaded_knee_contribution, std_mono_loaded_knee_contribution,_,_\
 = utils.actuators_energy_contribution(assisted_energy_dataset['monoarticular_ideal_loaded_hipactuator_energy'],assisted_energy_dataset['monoarticular_ideal_loaded_kneeactuator_energy'],\
                                        assisted_energy_dataset['biarticular_ideal_loaded_hipactuator_energy']+assisted_energy_dataset['biarticular_ideal_loaded_kneeactuator_energy'])
#*****************************
# loaded versus noload
# bi noload actuators contributions
mean_bi_2_noload_hip_contribution, std_bi_2_noload_hip_contribution,\
mean_bi_2_noload_knee_contribution, std_bi_2_noload_knee_contribution,_,_\
 = utils.actuators_energy_contribution(assisted_energy_dataset['biarticular_ideal_noload_hipactuator_energy'],assisted_energy_dataset['biarticular_ideal_noload_kneeactuator_energy'])
# mono noload actuators contributions
mean_mono_2_noload_hip_contribution, std_mono_2_noload_hip_contribution,\
mean_mono_2_noload_knee_contribution, std_mono_2_noload_knee_contribution,_,_\
 = utils.actuators_energy_contribution(assisted_energy_dataset['monoarticular_ideal_noload_hipactuator_energy'],assisted_energy_dataset['monoarticular_ideal_noload_kneeactuator_energy'])
# bi loaded actuators contributions
mean_bi_2_loaded_hip_contribution, std_bi_2_loaded_hip_contribution,\
mean_bi_2_loaded_knee_contribution, std_bi_2_loaded_knee_contribution,_,_\
 = utils.actuators_energy_contribution(assisted_energy_dataset['biarticular_ideal_loaded_hipactuator_energy'],assisted_energy_dataset['biarticular_ideal_loaded_kneeactuator_energy'],\
                                       assisted_energy_dataset['biarticular_ideal_noload_hipactuator_energy']+assisted_energy_dataset['biarticular_ideal_noload_kneeactuator_energy'])
# mono vs bi loaded actuators contributions
mean_mono_2_loaded_hip_contribution, std_mono_2_loaded_hip_contribution,\
mean_mono_2_loaded_knee_contribution, std_mono_2_loaded_knee_contribution,_,_\
 = utils.actuators_energy_contribution(assisted_energy_dataset['monoarticular_ideal_loaded_hipactuator_energy'],assisted_energy_dataset['monoarticular_ideal_loaded_kneeactuator_energy'],\
                                       assisted_energy_dataset['monoarticular_ideal_noload_hipactuator_energy']+assisted_energy_dataset['monoarticular_ideal_noload_kneeactuator_energy'])

# figure
width = 0.35
labels= ['hip','knee']
x = np.arange(len(labels))
# first bar plot setting
means_1 = [mean_bi_noload_hip_contribution,mean_bi_noload_knee_contribution]
means_2 = [mean_mono_noload_hip_contribution,mean_mono_noload_knee_contribution]
std_1 = [std_bi_noload_hip_contribution,std_bi_noload_knee_contribution]
std_2 = [std_mono_noload_hip_contribution,std_mono_noload_knee_contribution]
# second bar plot setting
means_3 = [mean_bi_loaded_hip_contribution,mean_bi_loaded_knee_contribution]
means_4 = [mean_mono_loaded_hip_contribution,mean_mono_loaded_knee_contribution]
std_3 = [std_bi_loaded_hip_contribution,std_bi_loaded_knee_contribution]
std_4 = [std_mono_loaded_hip_contribution,std_mono_loaded_knee_contribution]
# third bar plot setting
means_5 = [mean_bi_2_loaded_hip_contribution,mean_bi_2_loaded_knee_contribution]
means_6 = [mean_bi_2_noload_hip_contribution,mean_bi_2_noload_knee_contribution]
std_5 = [std_bi_2_loaded_hip_contribution,std_bi_2_loaded_knee_contribution]
std_6 = [std_bi_2_noload_hip_contribution,std_bi_2_noload_knee_contribution]
# fourth bar plot setting
means_7 = [mean_mono_2_loaded_hip_contribution,mean_mono_2_loaded_knee_contribution]
means_8 = [mean_mono_2_noload_hip_contribution,mean_mono_2_noload_knee_contribution]
std_7 = [std_mono_2_loaded_hip_contribution,std_mono_2_loaded_knee_contribution]
std_8 = [std_mono_2_noload_hip_contribution,std_mono_2_noload_knee_contribution]
# plot
fig,ax = plt.subplots(num='Actuators Contribution',nrows=2,ncols=2,figsize=(12.8, 9.6))
# noload bi vs mono
ax[0,0].bar(x - width/2, means_1, width, label='noload, biarticular', yerr = std_1, color=mycolors['french rose'])
ax[0,0].bar(x + width/2, means_2, width, label='noload, monoarticular', yerr = std_2, color=mycolors['olympic blue'])
# Add some text for labels, title and custom x-axis tick labels, etc.
ax[0,0].set_ylabel('contribution (%)')
ax[0,0].set_title('actuators contribution,\nnoload bi vs mono')
ax[0,0].set_xticks(x)
ax[0,0].set_yticks([0,20,40,60,80])
ax[0,0].set_xticklabels(labels)
ax[0,0].legend(['biarticular','monoarticular'],loc='best',frameon=False)
utils.no_top_right(ax[0,0])
# loaded bi vs mono
ax[0,1].bar(x - width/2, means_3, width, label='loaded, biarticular', yerr = std_1, color=mycolors['crimson red'])
ax[0,1].bar(x + width/2, means_4, width, label='loaded, monoarticular', yerr = std_2, color=mycolors['royal blue'])
# Add some text for labels, title and custom x-axis tick labels, etc.
ax[0,1].set_title('actuators contribution,\nloaded bi vs mono')
ax[0,1].set_xticks(x)
ax[0,1].set_yticks([0,20,40,60,80])
ax[0,1].set_xticklabels(labels)
ax[0,1].legend(['biarticular','monoarticular'],loc='best',frameon=False)
utils.no_top_right(ax[0,1])
# bi loaded vs noload
ax[1,0].bar(x - width/2, means_3, width, label='loaded, biarticular', yerr = std_1, color=mycolors['crimson red'])
ax[1,0].bar(x + width/2, means_4, width, label='noload, biarticular', yerr = std_2, color=mycolors['french rose'])
# Add some text for labels, title and custom x-axis tick labels, etc.
ax[1,0].set_title('actuators contribution,\nbi loaded vs noload')
ax[1,0].set_xticks(x)
ax[1,0].set_yticks([0,20,40,60,80])
ax[1,0].set_xticklabels(labels)
ax[1,0].legend(['loaded','noload'],loc='best',frameon=False)
utils.no_top_right(ax[1,0])
# mono loaded vs noload
ax[1,1].bar(x - width/2, means_3, width, label='loaded, monoarticular', yerr = std_1, color=mycolors['royal blue'])
ax[1,1].bar(x + width/2, means_4, width, label='noload, monoarticular', yerr = std_2, color=mycolors['olympic blue'])
# Add some text for labels, title and custom x-axis tick labels, etc.
ax[1,1].set_title('actuators contribution,\nmono loaded vs noload')
ax[1,1].set_xticks(x)
ax[1,1].set_yticks([0,20,40,60,80])
ax[1,1].set_xticklabels(labels)
ax[1,1].legend(['loaded','noload'],loc='best',frameon=False)
utils.no_top_right(ax[1,1])
fig.tight_layout(h_pad=-1, w_pad=-1.5)
fig.subplots_adjust(top=0.98, bottom=0.075, left=0.100, right=0.975,hspace=0.25,wspace=0.15)
fig.savefig('./Figures/Ideal/Actuators_Contributions.pdf',orientation='landscape',bbox_inches='tight')
plt.show()
#####################################################################################
# gait phases
# Processing Data
# toe-off
mean_noload_toe_off,_,mean_loaded_toe_off,_,noload_toe_off,loaded_toe_off = utils.toe_off_avg_std(gl_noload,gl_loaded,subjects=True)
mean_loaded_gaitcycle, std_loaded_gaitcycle = utils.mean_std_gaitcycle_phases(loaded_toe_off)
mean_noload_gaitcycle, std_noload_gaitcycle = utils.mean_std_gaitcycle_phases(noload_toe_off)

fig = plt.figure(num='gait cycles',figsize=(12.8, 9.6))
plt.subplot(211)
utils.plot_gait_cycle_phase(mean_loaded_gaitcycle, std_loaded_gaitcycle,mean_loaded_toe_off,loadcond='loaded')
plt.subplot(212)
utils.plot_gait_cycle_phase(mean_noload_gaitcycle, std_noload_gaitcycle,mean_noload_toe_off,loadcond='noload')
plt.xlabel('gait cycle (%)')
fig.tight_layout(h_pad=-1, w_pad=-1.5)
fig.subplots_adjust(top=0.98, bottom=0.075, left=0.100, right=0.975,hspace=0.25,wspace=0.15)
fig.savefig('./Figures/Ideal/Gait_Cycle_Phases.pdf',orientation='landscape',bbox_inches='tight')
plt.show()

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
plt.rcParams.update({'font.size': 14})
fig = plt.figure(constrained_layout=True,figsize=(6.4*2, 4.8*6))
gs = fig.add_gridspec(3, 2)
# Biarticular
names = ['unassist,\n noload','mono,\n noload','bi, noload','unassist,\n loaded','mono,\n loaded','bi, loaded']
x = np.arange(1,len(names)+1,1)
data = [utils.mean_over_trials(unassisted_energy_dataset['noload_metabolics_energy'],ax=0),\
        utils.mean_over_trials(assisted_energy_dataset['monoarticular_ideal_noload_metabolics_energy'],ax=0),\
        utils.mean_over_trials(assisted_energy_dataset['biarticular_ideal_noload_metabolics_energy'],ax=0),\
        utils.mean_over_trials(unassisted_energy_dataset['loaded_metabolics_energy'],ax=0),\
        utils.mean_over_trials(assisted_energy_dataset['monoarticular_ideal_loaded_metabolics_energy'],ax=0),\
        utils.mean_over_trials(assisted_energy_dataset['biarticular_ideal_loaded_metabolics_energy'],ax=0)]
ax = fig.add_subplot(gs[0, :])
bp = ax.boxplot(data, patch_artist=True)
utils.beautiful_boxplot(bp)
ax.tick_params(axis='both',direction='in')
ax.set_ylabel('metabolic rate (W/Kg)')
ax.set_xticks(x)
ax.set_yticks([4,6,8,10])
ax.set_xticklabels(names)
ax.set_title('metabolic rate\n')
utils.no_top_right(ax)

# Biarticular Loaded Vs Noload
names = ['bi hip,\n loaded','bi knee,\n loaded','bi hip,\n noload','bi knee,\n noload',]
x = np.arange(1,len(names)+1,1)
data = [utils.mean_over_trials(assisted_energy_dataset['biarticular_ideal_loaded_hipactuator_energy'],ax=0),utils.mean_over_trials(assisted_energy_dataset['biarticular_ideal_loaded_kneeactuator_energy'],ax=0),\
        utils.mean_over_trials(assisted_energy_dataset['biarticular_ideal_noload_hipactuator_energy'],ax=0),utils.mean_over_trials(assisted_energy_dataset['biarticular_ideal_noload_kneeactuator_energy'],ax=0)]
ax = fig.add_subplot(gs[1, 0])
bp = ax.boxplot(data, patch_artist=True)
utils.beautiful_boxplot(bp)
ax.tick_params(axis='both',direction='in')
ax.set_ylabel('actuator power (W/Kg)')
ax.set_xticks(x)
ax.set_yticks([0,1,2,3])
ax.set_xticklabels(names)
ax.set_title('biarticular, actutor power\n')
utils.no_top_right(ax)

# Monoarticular Loaded Vs Noload
names = ['mono hip,\n loaded','mono knee,\n loaded','mono hip,\n noload','mono knee,\n noload',]
x = np.arange(1,len(names)+1,1)
data = [utils.mean_over_trials(assisted_energy_dataset['monoarticular_ideal_loaded_hipactuator_energy'],ax=0),utils.mean_over_trials(assisted_energy_dataset['monoarticular_ideal_loaded_kneeactuator_energy'],ax=0),\
        utils.mean_over_trials(assisted_energy_dataset['monoarticular_ideal_noload_hipactuator_energy'],ax=0),utils.mean_over_trials(assisted_energy_dataset['monoarticular_ideal_noload_kneeactuator_energy'],ax=0)]
ax = fig.add_subplot(gs[1, 1])
bp = ax.boxplot(data, patch_artist=True)
utils.beautiful_boxplot(bp)
ax.tick_params(axis='both',direction='in')
ax.set_ylabel('actuator power (W/Kg)')
ax.set_xticks(x)
ax.set_yticks([0,1,2,3])
ax.set_xticklabels(names)
ax.set_title('monoarticular, actuator power\n')
utils.no_top_right(ax)
# Loaded Biarticular Vs Monoarticular
names = ['bi hip,\n loaded','bi knee,\n loaded','mono hip,\n loaded','mono knee,\n loaded']
x = np.arange(1,len(names)+1,1)
data = [utils.mean_over_trials(assisted_energy_dataset['biarticular_ideal_loaded_hipactuator_energy'],ax=0),utils.mean_over_trials(assisted_energy_dataset['biarticular_ideal_loaded_kneeactuator_energy'],ax=0),\
        utils.mean_over_trials(assisted_energy_dataset['monoarticular_ideal_loaded_hipactuator_energy'],ax=0),utils.mean_over_trials(assisted_energy_dataset['monoarticular_ideal_loaded_kneeactuator_energy'],ax=0)]
ax = fig.add_subplot(gs[2, 0])
bp = ax.boxplot(data, patch_artist=True)
utils.beautiful_boxplot(bp)
ax.tick_params(axis='both',direction='in')
ax.set_ylabel('actuator power (W/Kg)')
ax.set_xticks(x)
ax.set_yticks([0,1,2,3])
ax.set_xticklabels(names)
ax.set_title('actutor power, loaded condition\n')
utils.no_top_right(ax)

# Noload Biarticular Vs Monoarticular
names = ['bi hip,\n noload','bi knee,\n noload','mono hip,\n noload','mono knee,\n noload']
x = np.arange(1,len(names)+1,1)
data = [utils.mean_over_trials(assisted_energy_dataset['biarticular_ideal_noload_hipactuator_energy'],ax=0),utils.mean_over_trials(assisted_energy_dataset['biarticular_ideal_noload_kneeactuator_energy'],ax=0),\
        utils.mean_over_trials(assisted_energy_dataset['monoarticular_ideal_noload_hipactuator_energy'],ax=0),utils.mean_over_trials(assisted_energy_dataset['monoarticular_ideal_noload_kneeactuator_energy'],ax=0)]
ax = fig.add_subplot(gs[2, 1])
bp = ax.boxplot(data, patch_artist=True)
utils.beautiful_boxplot(bp)
ax.tick_params(axis='both',direction='in')
ax.set_ylabel('actuator power (W/Kg)')
ax.set_xticks(x)
ax.set_yticks([0,1,2,3])
ax.set_xticklabels(names)
ax.set_title('actuator power, noload conditon\n')
utils.no_top_right(ax)
fig.tight_layout(h_pad=-1, w_pad=-1.5)
fig.subplots_adjust(top=0.95, bottom=0.075, left=0.100, right=0.95,hspace=0.90,wspace=0.25)
plt.show()
fig.savefig('./Figures/Ideal/Paper_Figure_Energy_BoxPlot.pdf',orientation='landscape',bbox_inches='tight')
