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
#####################################################################################
subjects = ['05','07','09','10','11','12','14']
trials_num = ['01','02','03']
gait_cycle = np.linspace(0,100,1000)
#####################################################################################
# Reading CSV files into a dictionary and constructing gls
# muscles moment dataset
directory = 'D:/Ali.K.M.Bonab/Walking_Mass_Inertia_Effect/Data/Data/Unassist/*_musclesmoment.csv'
files = enumerate(glob.iglob(directory), 1)
musclesmoment_dataset = {pathlib.PurePath(f[1]).stem: np.loadtxt(f[1], delimiter=',') for f in files}
# muscles activation dataset
directory = 'D:/Ali.K.M.Bonab/Walking_Mass_Inertia_Effect/Data/Data/Unassist/*_activation.csv'
files = enumerate(glob.iglob(directory), 1)
musclesactivation_dataset = {pathlib.PurePath(f[1]).stem: np.loadtxt(f[1], delimiter=',') for f in files}

# gls
gl_noload = {'noload_subject{}_trial{}'.format(i,j): utils.construct_gl_mass_side(subjectno=i,trialno=j,loadcond='noload') for i in subjects for j in trials_num}
gl_loaded = {'loaded_subject{}_trial{}'.format(i,j): utils.construct_gl_mass_side(subjectno=i,trialno=j,loadcond='loaded') for i in subjects for j in trials_num}
#####################################################################################
# Processing Data
# toe-off
noload_mean_toe_off,_,loaded_mean_toe_off,_ = utils.toe_off_avg_std(gl_noload,gl_loaded)
# hip muscles moment
normal_loaded_hipmuscles_moment = utils.normalize_direction_data(musclesmoment_dataset['loaded_hip_musclesmoment'],gl_noload,direction=True)
normal_noload_hipmuscles_moment = utils.normalize_direction_data(musclesmoment_dataset['noload_hip_musclesmoment'],gl_noload,direction=True)
mean_norm_loaded_hipmuscles_moment,std_norm_loaded_hipmuscles_moment = utils.mean_std_over_subjects(normal_loaded_hipmuscles_moment)
mean_norm_noload_hipmuscles_moment,std_norm_noload_hipmuscles_moment = utils.mean_std_over_subjects(normal_noload_hipmuscles_moment)
# knee muscles moment
normal_loaded_kneemuscles_moment = utils.normalize_direction_data(musclesmoment_dataset['loaded_knee_musclesmoment'],gl_noload,direction=True)
normal_noload_kneemuscles_moment = utils.normalize_direction_data(musclesmoment_dataset['noload_knee_musclesmoment'],gl_noload,direction=True)
mean_norm_loaded_kneemuscles_moment,std_norm_loaded_kneemuscles_moment = utils.mean_std_over_subjects(normal_loaded_kneemuscles_moment)
mean_norm_noload_kneemuscles_moment,std_norm_noload_kneemuscles_moment = utils.mean_std_over_subjects(normal_noload_kneemuscles_moment)
# muscles activation
normal_loaded_muscles_activation = utils.normalize_direction_data(musclesactivation_dataset['loaded_ninemuscles_activation'],gl_noload,direction=False,normalize=False)
normal_noload_muscles_activation = utils.normalize_direction_data(musclesactivation_dataset['noload_ninemuscles_activation'],gl_noload,direction=False,normalize=False)
mean_norm_loaded_muscles_activation,std_norm_loaded_muscles_activation = utils.mean_std_muscles_subjects(normal_loaded_muscles_activation)
mean_norm_noload_muscles_activation,std_norm_noload_muscles_activation = utils.mean_std_muscles_subjects(normal_noload_muscles_activation)

#####################################################################################
# Plots
# hip joint moment plot dictionaries
hip_moment_loaded_plot_dic = {'pgc':gait_cycle,'avg':utils.smooth(mean_norm_loaded_hipmuscles_moment,3),'label':'Loaded',
                        'std':utils.smooth(std_norm_loaded_hipmuscles_moment,3),'avg_toeoff':loaded_mean_toe_off}
hip_moment_noload_plot_dic = {'pgc':gait_cycle,'avg':utils.smooth(mean_norm_noload_hipmuscles_moment,3),'label':'Noload',
                        'std':utils.smooth(std_norm_noload_hipmuscles_moment,3),'avg_toeoff':noload_mean_toe_off}
# knee joint moment plot dictionaries
knee_moment_loaded_plot_dic = {'pgc':gait_cycle,'avg':utils.smooth(mean_norm_loaded_kneemuscles_moment,3),'label':'Loaded',
                        'std':utils.smooth(std_norm_loaded_kneemuscles_moment,3),'avg_toeoff':loaded_mean_toe_off}
knee_moment_noload_plot_dic = {'pgc':gait_cycle,'avg':utils.smooth(mean_norm_noload_kneemuscles_moment,3),'label':'Noload',
                        'std':utils.smooth(std_norm_noload_kneemuscles_moment,3),'avg_toeoff':noload_mean_toe_off}
# muscles moment plot dictionaries
muscles_activation_loaded_plot_dic = {'pgc':gait_cycle,'avg':mean_norm_loaded_muscles_activation,'muscle_group': 'nine_muscles',
                                      'std':std_norm_loaded_muscles_activation,'avg_toeoff':loaded_mean_toe_off}
muscles_activation_noload_plot_dic = {'pgc':gait_cycle,'avg':mean_norm_noload_muscles_activation,'muscle_group': 'nine_muscles',
                                      'std':std_norm_noload_muscles_activation,'avg_toeoff':noload_mean_toe_off}

#*****************************

# hip joint moment figure
fig, ax = plt.subplots(num='Hip Muscles Moment',figsize=(6.4, 4.8))
utils.plot_shaded_avg(plot_dic=hip_moment_loaded_plot_dic,color='k')
utils.plot_shaded_avg(plot_dic=hip_moment_noload_plot_dic,toeoff_color='xkcd:shamrock green',color='xkcd:irish green')
plt.legend(loc='best',frameon=False)
plt.ylim((-1.5,2))
plt.xlabel('gait cycle (%)')
plt.ylabel('hip muscles moment (N-m/kg)')
utils.no_top_right(ax)
plt.show()
fig.tight_layout(h_pad=-4.0, w_pad=-4.0)
fig.savefig('./Figures/Unassist/HipMusclesMoment.pdf',orientation='landscape',bbox_inches='tight')

# knee joint moment figure
fig, ax = plt.subplots(num='Knee Muscles Moment',figsize=(6.4, 4.8))
utils.plot_shaded_avg(plot_dic=knee_moment_loaded_plot_dic,color='k')
utils.plot_shaded_avg(plot_dic=knee_moment_noload_plot_dic,toeoff_color='xkcd:shamrock green',color='xkcd:irish green')
plt.legend(loc='best',frameon=False)
plt.ylim((-1.5,2))
plt.xlabel('gait cycle (%)')
plt.ylabel('knee muscles moment (N-m/kg)')
utils.no_top_right(ax)
plt.show()
fig.tight_layout(h_pad=-4.0, w_pad=-4.0)
fig.savefig('./Figures/Unassist/KneeMusclesMoment.pdf',orientation='landscape',bbox_inches='tight')

# muscles activation figure
fig, ax = plt.subplots(num='Muscles Activation',figsize=(7.4, 5.8))
utils.plot_muscles_avg(plot_dic=muscles_activation_loaded_plot_dic,color='k',is_std=True)
utils.plot_muscles_avg(plot_dic=muscles_activation_noload_plot_dic,toeoff_color='xkcd:shamrock green',color='xkcd:irish green',is_std=True)
plt.show()
fig.tight_layout(h_pad=-4.0, w_pad=-4.0)
fig.savefig('./Figures/Unassist/NineMusclesActivation.pdf',orientation='landscape',bbox_inches='tight')
