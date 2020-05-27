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
# noload monoarticular
mono_noload_muscles_activation = utils.normalize_direction_data(musclesactivation_dataset['monoarticular_hip30knee30_noload_ninemuscles_activation'],gl_noload,direction=False,normalize=False)
mean_mono_noload_muscles_activation,std_mono_noload_muscles_activation = utils.mean_std_muscles_subjects(mono_noload_muscles_activation)
# loaded monoarticular
mono_loaded_muscles_activation = utils.normalize_direction_data(musclesactivation_dataset['monoarticular_hip70knee30_load_ninemuscles_activation'],gl_noload,direction=False,normalize=False)
mean_mono_loaded_muscles_activation,std_mono_loaded_muscles_activation = utils.mean_std_muscles_subjects(mono_loaded_muscles_activation)
#unassist
mean_unassist_noload_muscles_activation = utils.recover_muscledata(unassist_dataset,'mean_norm_noload_muscles_activation')
std_unassist_noload_muscles_activation  = utils.recover_muscledata(unassist_dataset,'std_norm_noload_muscles_activation')
mean_unassist_loaded_muscles_activation = utils.recover_muscledata(unassist_dataset,'mean_norm_loaded_muscles_activation')
std_unassist_loaded_muscles_activation  = utils.recover_muscledata(unassist_dataset,'std_norm_loaded_muscles_activation')

#####################################################################################
# Plots
# muscles activation plot dictionaries
mono_noload_muscles_activation_dic = {'pgc':gait_cycle,'avg':mean_mono_noload_muscles_activation,'label':'Noload',
                        'std':std_mono_noload_muscles_activation,'avg_toeoff':noload_mean_toe_off}
mono_loaded_muscles_activation_dic = {'pgc':gait_cycle,'avg':mean_mono_loaded_muscles_activation,'label':'Loaded',
                        'std':std_mono_loaded_muscles_activation,'avg_toeoff':loaded_mean_toe_off}
unassist_noload_muscles_activation_dic = {'pgc':gait_cycle,'avg':mean_unassist_noload_muscles_activation,'label':'Noload',
                        'std':std_unassist_noload_muscles_activation,'avg_toeoff':noload_mean_toe_off}

#******************************************************************************************************************************
#******************************************************************************************************************************
muscles_activation_noload_plot_dic = {'pgc':gait_cycle,'avg':mean_unassist_noload_muscles_activation,'muscle_group': 'nine_muscles',
                                      'label':'noload, unassist','std':std_unassist_noload_muscles_activation,'avg_toeoff':noload_mean_toe_off}
muscles_activation_loaded_plot_dic = {'pgc':gait_cycle,'avg':mean_unassist_loaded_muscles_activation,'muscle_group': 'nine_muscles',
                                      'label':'loaded, unassist','std':std_unassist_loaded_muscles_activation,'avg_toeoff':loaded_mean_toe_off}
muscles_activation_monoarticular_noload_plot_dic = {'pgc':gait_cycle,'avg':mean_mono_noload_muscles_activation,'muscle_group': 'nine_muscles',
                                                'label':'noload, monoarticular','std':std_mono_noload_muscles_activation,'avg_toeoff':noload_mean_toe_off}
muscles_activation_monoarticular_loaded_plot_dic = {'pgc':gait_cycle,'avg':mean_mono_loaded_muscles_activation,'muscle_group': 'nine_muscles',
                                                'label':'loaded, monoarticular','std':std_mono_loaded_muscles_activation,'avg_toeoff':loaded_mean_toe_off}

# muscles activation figure
# monoarticular
fig, ax = plt.subplots(num='Monoarticular Noload Muscles Activation',figsize=(8.4, 6.8))
utils.plot_muscles_avg(plot_dic=muscles_activation_noload_plot_dic,toeoff_color='xkcd:shamrock green',color='xkcd:irish green',is_std=True)
utils.plot_muscles_avg(plot_dic=muscles_activation_monoarticular_noload_plot_dic,toeoff_color='xkcd:shamrock green',color=mycolors['lavender purple'],is_std=True)
plt.legend(loc='best',frameon=False)
plt.show()
fig.tight_layout()
fig.savefig('./Figures/Specific_Weights/NoloadMono25_LoadedMono05/Monoarticular_Noload_MusclesActivation.pdf',orientation='landscape',bbox_inches='tight')

# monoarticular
fig, ax = plt.subplots(num='Monoarticular Load Muscles Activation',figsize=(8.4, 6.8))
utils.plot_muscles_avg(plot_dic=muscles_activation_loaded_plot_dic,toeoff_color='grey',color='k',is_std=True)
utils.plot_muscles_avg(plot_dic=muscles_activation_monoarticular_loaded_plot_dic,toeoff_color='grey',color=mycolors['dark purple'],is_std=True)
plt.legend(loc='best',frameon=False)
plt.show()
fig.tight_layout()
fig.savefig('./Figures/Specific_Weights/NoloadMono25_LoadedMono05/Monoarticular_Loaded_MusclesActivation.pdf',orientation='landscape',bbox_inches='tight')

# loaded versus noload
fig, ax = plt.subplots(num='Muscles Activation',figsize=(8.4, 6.8))
utils.plot_muscles_avg(plot_dic=muscles_activation_noload_plot_dic,toeoff_color='xkcd:shamrock green',color='xkcd:irish green',is_std=False)
utils.plot_muscles_avg(plot_dic=muscles_activation_loaded_plot_dic,toeoff_color='grey',color='k',is_std=False)
utils.plot_muscles_avg(plot_dic=muscles_activation_monoarticular_noload_plot_dic,toeoff_color='xkcd:shamrock green',color=mycolors['lavender purple'],is_std=False)
utils.plot_muscles_avg(plot_dic=muscles_activation_monoarticular_loaded_plot_dic,toeoff_color='grey',color=mycolors['dark purple'],is_std=False)
plt.legend(loc='best',frameon=False)
plt.show()
fig.tight_layout()
fig.savefig('./Figures/Specific_Weights/NoloadMono25_LoadedMono05/Monoarticular_LoadedVsNoload_MusclesActivation.pdf',orientation='landscape',bbox_inches='tight')

