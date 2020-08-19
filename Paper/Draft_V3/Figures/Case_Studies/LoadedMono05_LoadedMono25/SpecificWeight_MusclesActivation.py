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
# assisted subjects muscles activation dataset
directory = 'D:/Ali.K.M.Bonab/Walking_Mass_Inertia_Effect/Data/Data/Specific_Weights/*_activation.csv'
files = enumerate(glob.iglob(directory), 1)
musclesactivation_dataset = {pathlib.PurePath(f[1]).stem: np.loadtxt(f[1], delimiter=',') for f in files}
# gls
gl_noload = {'noload_subject{}_trial{}'.format(i,j): utils.construct_gl_mass_side(subjectno=i,trialno=j,loadcond='noload') for i in subjects for j in trials_num}
gl_loaded = {'loaded_subject{}_trial{}'.format(i,j): utils.construct_gl_mass_side(subjectno=i,trialno=j,loadcond='loaded') for i in subjects for j in trials_num}
#####################################################################################
# Processing Data
# toe-off
noload_mean_toe_off,_,loaded_mean_toe_off,_ = utils.toe_off_avg_std(gl_noload,gl_loaded)
# muscles activation
# monoarticular_ee
mono_Ee_loaded_muscles_activation = utils.normalize_direction_data(musclesactivation_dataset['monoarticular_hip30knee30_load_ninemuscles_activation'],gl_loaded,direction=False,normalize=False)
mean_mono_Ee_loaded_muscles_activation,std_mono_Ee_loaded_muscles_activation = utils.mean_std_muscles_subjects(mono_Ee_loaded_muscles_activation)
# monoarticular
mono_Ae_loaded_muscles_activation = utils.normalize_direction_data(musclesactivation_dataset['monoarticular_hip70knee30_load_ninemuscles_activation'],gl_noload,direction=False,normalize=False)
mean_mono_Ae_loaded_muscles_activation,std_mono_Ae_loaded_muscles_activation = utils.mean_std_muscles_subjects(mono_Ae_loaded_muscles_activation)
#unassist
mean_unassist_loaded_muscles_activation = utils.recover_muscledata(unassist_dataset,'mean_norm_loaded_muscles_activation')
std_unassist_loaded_muscles_activation  = utils.recover_muscledata(unassist_dataset,'std_norm_loaded_muscles_activation')

#####################################################################################
# Plots
# muscles activation plot dictionaries
mono_Ee_loaded_muscles_activation_dic = {'pgc':gait_cycle,'avg':mean_mono_Ee_loaded_muscles_activation,'label':'Loaded "Ee"',
                        'std':std_mono_Ee_loaded_muscles_activation,'avg_toeoff':loaded_mean_toe_off}
mono_Ae_loaded_muscles_activation_dic = {'pgc':gait_cycle,'avg':mean_mono_Ae_loaded_muscles_activation,'label':'Loaded "Ae"',
                        'std':std_mono_Ae_loaded_muscles_activation,'avg_toeoff':loaded_mean_toe_off}
unassist_loaded_muscles_activation_dic = {'pgc':gait_cycle,'avg':mean_unassist_loaded_muscles_activation,'label':'Loaded joint',
                        'std':std_unassist_loaded_muscles_activation,'avg_toeoff':loaded_mean_toe_off}

#******************************************************************************************************************************
#******************************************************************************************************************************
muscles_activation_loaded_plot_dic = {'pgc':gait_cycle,'avg':mean_unassist_loaded_muscles_activation,'muscle_group': 'nine_muscles',
                                      'label':'loaded, unassist','std':std_unassist_loaded_muscles_activation,'avg_toeoff':loaded_mean_toe_off}

muscles_activation_monoarticular_ee_loaded_plot_dic = {'pgc':gait_cycle,'avg':mean_mono_Ee_loaded_muscles_activation,'muscle_group': 'nine_muscles',
                                                'label':'monoarticular "Ee"','std':std_mono_Ee_loaded_muscles_activation,'avg_toeoff':loaded_mean_toe_off}
muscles_activation_monoarticular_loaded_plot_dic = {'pgc':gait_cycle,'avg':mean_mono_Ae_loaded_muscles_activation,'muscle_group': 'nine_muscles',
                                                'label':'monoarticular "Ae"','std':std_mono_Ae_loaded_muscles_activation,'avg_toeoff':loaded_mean_toe_off}

# muscles activation figure
# monoarticular_ee
fig, ax = plt.subplots(num='Monoarticular "Ee" Loaded Muscles Activation',figsize=(8.4, 6.8))
utils.plot_muscles_avg(plot_dic=muscles_activation_loaded_plot_dic,toeoff_color='grey',color='k',is_std=True)
utils.plot_muscles_avg(plot_dic=muscles_activation_monoarticular_ee_loaded_plot_dic,toeoff_color='grey',color=mycolors['french rose'],is_std=True)
plt.legend(loc='best',frameon=False)
plt.show()
fig.tight_layout()
fig.savefig('./Figures/Specific_Weights/LoadedMono05_LoadedMono25/Monoarticular_Ee_Loaded_MusclesActivation.pdf',orientation='landscape',bbox_inches='tight')

# monoarticular
fig, ax = plt.subplots(num='Monoarticular Loaded Muscles Activation',figsize=(8.4, 6.8))
utils.plot_muscles_avg(plot_dic=muscles_activation_loaded_plot_dic,toeoff_color='grey',color='k',is_std=True)
utils.plot_muscles_avg(plot_dic=muscles_activation_monoarticular_loaded_plot_dic,toeoff_color='grey',color=mycolors['lavender purple'],is_std=True)
plt.legend(loc='best',frameon=False)
plt.show()
fig.tight_layout()
fig.savefig('./Figures/Specific_Weights/LoadedMono05_LoadedMono25/Monoarticular_Ae_Loaded_MusclesActivation.pdf',orientation='landscape',bbox_inches='tight')

# monoarticular versus monoarticular_ee
fig, ax = plt.subplots(num='Loaded Muscles Activation',figsize=(8.4, 6.8))
utils.plot_muscles_avg(plot_dic=muscles_activation_loaded_plot_dic,toeoff_color='grey',color='k',is_std=False)
utils.plot_muscles_avg(plot_dic=muscles_activation_monoarticular_ee_loaded_plot_dic,toeoff_color='grey',color='lightsteelblue',is_std=False)
utils.plot_muscles_avg(plot_dic=muscles_activation_monoarticular_loaded_plot_dic,toeoff_color='grey',color='blue',is_std=False)
plt.legend(loc='best',frameon=False)
plt.show()
fig.tight_layout()
fig.savefig('./Figures/Specific_Weights/LoadedMono05_LoadedMono25/Monoarticular_Ae_VS_Monoarticular_Ee_Loaded_MusclesActivation.pdf',orientation='landscape',bbox_inches='tight')

