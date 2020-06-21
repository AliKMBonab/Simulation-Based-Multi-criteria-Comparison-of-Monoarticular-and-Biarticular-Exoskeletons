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
rra_dataset = utils.csv2numpy('./Data/RRA/rra_final_data.csv') 
# muscles moment dataset
directory = './Data/Unassist/*_musclesmoment.csv'
files = enumerate(glob.iglob(directory), 1)
musclesmoment_dataset = {pathlib.PurePath(f[1]).stem: np.loadtxt(f[1], delimiter=',') for f in files}
# joint kinematics dataset
directory = '.\Data\RRA\*_kinematics.csv'
files = enumerate(glob.iglob(directory), 1)
jointkinematics_dataset = {pathlib.PurePath(f[1]).stem: np.loadtxt(f[1], delimiter=',') for f in files}
# gls
gl_noload = {'noload_subject{}_trial{}'.format(i,j): utils.construct_gl_mass_side(subjectno=i,trialno=j,loadcond='noload') for i in subjects for j in trials_num}
gl_loaded = {'loaded_subject{}_trial{}'.format(i,j): utils.construct_gl_mass_side(subjectno=i,trialno=j,loadcond='loaded') for i in subjects for j in trials_num}
#####################################################################################
# Processing Data
# toe-off
noload_mean_toe_off,_,loaded_mean_toe_off,_,subjects_noload_toe_off,subjects_loaded_toe_off = utils.toe_off_avg_std(gl_noload,gl_loaded,subjects=True)
# hip muscles moment
normal_loaded_hipmuscles_moment = utils.normalize_direction_data(musclesmoment_dataset['loaded_hip_musclesmoment'],gl_noload,direction=True)
normal_noload_hipmuscles_moment = utils.normalize_direction_data(musclesmoment_dataset['noload_hip_musclesmoment'],gl_noload,direction=True)
mean_norm_loaded_hipmuscles_moment,std_norm_loaded_hipmuscles_moment = utils.mean_std_over_subjects(normal_loaded_hipmuscles_moment,avg_trials=False)
mean_norm_noload_hipmuscles_moment,std_norm_noload_hipmuscles_moment = utils.mean_std_over_subjects(normal_noload_hipmuscles_moment,avg_trials=False)
# not normalized
loaded_hipmuscles_moment = utils.normalize_direction_data(musclesmoment_dataset['loaded_hip_musclesmoment'],gl_noload,normalize=True,direction=True)
noload_hipmuscles_moment = utils.normalize_direction_data(musclesmoment_dataset['noload_hip_musclesmoment'],gl_noload,normalize=True,direction=True)
mean_loaded_hipmuscles_moment,std_loaded_hipmuscles_moment = utils.mean_std_over_subjects(loaded_hipmuscles_moment,avg_trials=False)
mean_noload_hipmuscles_moment,std_noload_hipmuscles_moment = utils.mean_std_over_subjects(noload_hipmuscles_moment,avg_trials=False)

# knee muscles moment
normal_loaded_kneemuscles_moment = utils.normalize_direction_data(musclesmoment_dataset['loaded_knee_musclesmoment'],gl_noload,direction=True)
normal_noload_kneemuscles_moment = utils.normalize_direction_data(musclesmoment_dataset['noload_knee_musclesmoment'],gl_noload,direction=True)
mean_norm_loaded_kneemuscles_moment,std_norm_loaded_kneemuscles_moment = utils.mean_std_over_subjects(normal_loaded_kneemuscles_moment,avg_trials=False)
mean_norm_noload_kneemuscles_moment,std_norm_noload_kneemuscles_moment = utils.mean_std_over_subjects(normal_noload_kneemuscles_moment,avg_trials=False)
# not normalized
loaded_kneemuscles_moment = utils.normalize_direction_data(musclesmoment_dataset['loaded_knee_musclesmoment'],gl_noload,normalize=True,direction=True)
noload_kneemuscles_moment = utils.normalize_direction_data(musclesmoment_dataset['noload_knee_musclesmoment'],gl_noload,normalize=True,direction=True)
mean_loaded_kneemuscles_moment,std_loaded_kneemuscles_moment = utils.mean_std_over_subjects(loaded_kneemuscles_moment,avg_trials=False)
mean_noload_kneemuscles_moment,std_noload_kneemuscles_moment = utils.mean_std_over_subjects(noload_kneemuscles_moment,avg_trials=False)

# hip joint kinematics
loaded_hipjoint_kinematics = utils.normalize_direction_data(jointkinematics_dataset['loaded_hipjoint_kinematics'],gl_noload,normalize=False,direction=False)
noload_hipjoint_kinematics = utils.normalize_direction_data(jointkinematics_dataset['noload_hipjoint_kinematics'],gl_noload,normalize=False,direction=False)
mean_loaded_hipjoint_kinematics,std_loaded_hipjoint_kinematics = utils.mean_std_over_subjects(loaded_hipjoint_kinematics)
mean_noload_hipjoint_kinematics,std_noload_hipjoint_kinematics = utils.mean_std_over_subjects(noload_hipjoint_kinematics)
# knee joint kinematics
loaded_kneejoint_kinematics = utils.normalize_direction_data(jointkinematics_dataset['loaded_kneejoint_kinematics'],gl_noload,normalize=False,direction=False)
noload_kneejoint_kinematics = utils.normalize_direction_data(jointkinematics_dataset['noload_kneejoint_kinematics'],gl_noload,normalize=False,direction=False)
mean_loaded_kneejoint_kinematics,std_loaded_kneejoint_kinematics = utils.mean_std_over_subjects(loaded_kneejoint_kinematics)
mean_noload_kneejoint_kinematics,std_noload_kneejoint_kinematics = utils.mean_std_over_subjects(noload_kneejoint_kinematics)
#############################################################################################
# Stiffness analysis
# hip joint stiffness
hip_loaded_stiffness_dict, hip_loaded_Rsquare_dict, hip_loaded_bias_dict = utils.calculate_quasi_stiffness(angle=loaded_hipjoint_kinematics,moment=loaded_hipmuscles_moment,toe_off=subjects_loaded_toe_off,joint='hip')
hip_noload_stiffness_dict, hip_noload_Rsquare_dict, hip_noload_bias_dict = utils.calculate_quasi_stiffness(angle=noload_hipjoint_kinematics,moment=noload_hipmuscles_moment,toe_off=subjects_noload_toe_off,joint='hip')
# knee joint stiffness
knee_loaded_stiffness_dict, knee_loaded_Rsquare_dict, knee_loaded_bias_dict = utils.calculate_quasi_stiffness(angle=loaded_kneejoint_kinematics,moment=loaded_kneemuscles_moment,toe_off=subjects_loaded_toe_off,joint='knee')
knee_noload_stiffness_dict, knee_noload_Rsquare_dict, knee_noload_bias_dict = utils.calculate_quasi_stiffness(angle=noload_kneejoint_kinematics,moment=noload_kneemuscles_moment,toe_off=subjects_noload_toe_off,joint='knee')
# dataset for plotting the 
# hip joint
hip_loaded_linear_angle_dict,hip_loaded_linear_moment_dict,hip_loaded_fitted_line = utils.mean_linear_phases(mean_loaded_hipjoint_kinematics,mean_loaded_hipmuscles_moment,\
                                                                                    loaded_mean_toe_off,'hip',hip_loaded_bias_dict,hip_loaded_stiffness_dict)
hip_noload_linear_angle_dict,hip_noload_linear_moment_dict,hip_noload_fitted_line = utils.mean_linear_phases(mean_noload_hipjoint_kinematics,mean_noload_hipmuscles_moment,\
                                                                                    noload_mean_toe_off,'hip',hip_noload_bias_dict,hip_noload_stiffness_dict)
# knee joint
knee_loaded_linear_angle_dict,knee_loaded_linear_moment_dict,knee_loaded_fitted_line = utils.mean_linear_phases(mean_loaded_kneejoint_kinematics,mean_loaded_kneemuscles_moment,\
                                                                                    loaded_mean_toe_off,'knee',knee_loaded_bias_dict,knee_loaded_stiffness_dict)
knee_noload_linear_angle_dict,knee_noload_linear_moment_dict,knee_noload_fitted_line = utils.mean_linear_phases(mean_noload_kneejoint_kinematics,mean_noload_kneemuscles_moment,\
                                                                                    noload_mean_toe_off,'knee',knee_noload_bias_dict,knee_noload_stiffness_dict)
# save dataset
#***********************************************************************************************
# stiffness
Dataset = np.c_[hip_loaded_stiffness_dict['hip_extension_stiffness'],hip_loaded_stiffness_dict['hip_flexion_stiffness'],hip_loaded_stiffness_dict['hip_total_stiffness'],\
           hip_noload_stiffness_dict['hip_extension_stiffness'],hip_noload_stiffness_dict['hip_flexion_stiffness'],hip_noload_stiffness_dict['hip_total_stiffness'],\
           knee_loaded_stiffness_dict['knee_extension_stiffness'],knee_loaded_stiffness_dict['knee_flexion_stiffness'],knee_loaded_stiffness_dict['knee_total_stiffness'],\
           knee_noload_stiffness_dict['knee_extension_stiffness'],knee_noload_stiffness_dict['knee_flexion_stiffness'],knee_noload_stiffness_dict['knee_total_stiffness'],\
           hip_loaded_stiffness_dict['hip_extension_stiffness_raw'],hip_loaded_stiffness_dict['hip_flexion_stiffness_raw'],hip_loaded_stiffness_dict['hip_total_stiffness_raw'],\
           hip_noload_stiffness_dict['hip_extension_stiffness_raw'],hip_noload_stiffness_dict['hip_flexion_stiffness_raw'],hip_noload_stiffness_dict['hip_total_stiffness_raw'],\
           knee_loaded_stiffness_dict['knee_extension_stiffness_raw'],knee_loaded_stiffness_dict['knee_flexion_stiffness_raw'],knee_loaded_stiffness_dict['knee_total_stiffness_raw'],\
           knee_noload_stiffness_dict['knee_extension_stiffness_raw'],knee_noload_stiffness_dict['knee_flexion_stiffness_raw'],knee_noload_stiffness_dict['knee_total_stiffness_raw']]
Headers = ['loaded_hip_extension_stiffness','loaded_hip_flexion_stiffness','loaded_hip_total_stiffness',\
           'noload_hip_extension_stiffness','noload_hip_flexion_stiffness','noload_hip_total_stiffness',\
           'loaded_knee_extension_stiffness','loaded_knee_flexion_stiffness','loaded_knee_total_stiffness',\
           'noload_knee_extension_stiffness','noload_knee_flexion_stiffness','noload_knee_total_stiffness',\
           'loaded_hip_extension_stiffness_raw','loaded_hip_flexion_stiffness_raw','loaded_hip_total_stiffness_raw',\
           'noload_hip_extension_stiffness_raw','noload_hip_flexion_stiffness_raw','noload_hip_total_stiffness_raw',\
           'loaded_knee_extension_stiffness_raw','loaded_knee_flexion_stiffness_raw','loaded_knee_total_stiffness_raw',\
           'noload_knee_extension_stiffness_raw','noload_knee_flexion_stiffness_raw','noload_knee_total_stiffness_raw']
# List of numpy vectors to a numpy ndarray and save to csv file
with open(r'.\Data\Unassist\unassist_stiffness_data.csv', 'wb') as f:
  f.write(bytes(utils.listToString(Headers)+'\n','UTF-8'))
  np.savetxt(f, Dataset, fmt='%s', delimiter=",")
#***********************************************************************************************
mean_cellText = np.c_[hip_loaded_stiffness_dict['mean_hip_extension_stiffness'],hip_loaded_stiffness_dict['mean_hip_flexion_stiffness'],hip_loaded_stiffness_dict['mean_hip_total_stiffness'],\
           hip_noload_stiffness_dict['mean_hip_extension_stiffness'],hip_noload_stiffness_dict['mean_hip_flexion_stiffness'],hip_noload_stiffness_dict['mean_hip_total_stiffness'],\
           knee_loaded_stiffness_dict['mean_knee_extension_stiffness'],knee_loaded_stiffness_dict['mean_knee_flexion_stiffness'],knee_loaded_stiffness_dict['mean_knee_total_stiffness'],\
           knee_noload_stiffness_dict['mean_knee_extension_stiffness'],knee_noload_stiffness_dict['mean_knee_flexion_stiffness'],knee_noload_stiffness_dict['mean_knee_total_stiffness']]
std_cellText = np.c_[hip_loaded_stiffness_dict['std_hip_extension_stiffness'],hip_loaded_stiffness_dict['std_hip_flexion_stiffness'],hip_loaded_stiffness_dict['std_hip_total_stiffness'],\
           hip_noload_stiffness_dict['std_hip_extension_stiffness'],hip_noload_stiffness_dict['std_hip_flexion_stiffness'],hip_noload_stiffness_dict['std_hip_total_stiffness'],\
           knee_loaded_stiffness_dict['std_knee_extension_stiffness'],knee_loaded_stiffness_dict['std_knee_flexion_stiffness'],knee_loaded_stiffness_dict['std_knee_total_stiffness'],\
           knee_noload_stiffness_dict['std_knee_extension_stiffness'],knee_noload_stiffness_dict['std_knee_flexion_stiffness'],knee_noload_stiffness_dict['std_knee_total_stiffness']]
Dataset = np.concatenate((mean_cellText,std_cellText),axis=1)
Headers = []
for data_type in ['mean','std']:
      for joint in ['hip','knee']:
            for load in ['loaded','noload']:
                  for phase in ['extension','flexion','total']:
                        Headers.append('{}_{}_{}_{}_stiffness'.format(data_type,load,joint,phase))
# List of numpy vectors to a numpy ndarray and save to csv file
with open(r'.\Data\Unassist\unassist_meanstd_stiffness_data.csv', 'wb') as f:
  f.write(bytes(utils.listToString(Headers)+'\n','UTF-8'))
  np.savetxt(f, Dataset, fmt='%s', delimiter=",")                       
#***********************************************************************************************
# bias
Dataset = np.c_[hip_loaded_bias_dict['hip_extension_bias'],hip_loaded_bias_dict['hip_flexion_bias'],hip_loaded_bias_dict['hip_total_bias'],\
           hip_noload_bias_dict['hip_extension_bias'],hip_noload_bias_dict['hip_flexion_bias'],hip_noload_bias_dict['hip_total_bias'],\
           knee_loaded_bias_dict['knee_extension_bias'],knee_loaded_bias_dict['knee_flexion_bias'],knee_loaded_bias_dict['knee_total_bias'],\
           knee_noload_bias_dict['knee_extension_bias'],knee_noload_bias_dict['knee_flexion_bias'],knee_noload_bias_dict['knee_total_bias']]
Headers = ['loaded_hip_extension_bias','loaded_hip_flexion_bias','loaded_hip_total_bias',\
           'noload_hip_extension_bias','noload_hip_flexion_bias','noload_hip_total_bias',\
           'loaded_knee_extension_bias','loaded_knee_flexion_bias','loaded_knee_total_bias',\
           'noload_knee_extension_bias','noload_knee_flexion_bias','noload_knee_total_bias']
# List of numpy vectors to a numpy ndarray and save to csv file
with open(r'.\Data\Unassist\unassist_stiffness_bias_data.csv', 'wb') as f:
  f.write(bytes(utils.listToString(Headers)+'\n','UTF-8'))
  np.savetxt(f, Dataset, fmt='%s', delimiter=",")
#####################################################################################
# moment dataset without normalizing for stiffness plots
Headers = ['mean_loaded_hipmuscles_moment','std_loaded_hipmuscles_moment','mean_noload_hipmuscles_moment','std_noload_hipmuscles_moment',\
       'mean_loaded_kneemuscles_moment','std_loaded_kneemuscles_moment','mean_noload_kneemuscles_moment','std_noload_kneemuscles_moment']
# Dataset
Data =[mean_loaded_hipmuscles_moment,std_loaded_hipmuscles_moment,mean_noload_hipmuscles_moment,std_noload_hipmuscles_moment,\
       mean_loaded_kneemuscles_moment,std_loaded_kneemuscles_moment,mean_noload_kneemuscles_moment,std_noload_kneemuscles_moment]
# List of numpy vectors to a numpy ndarray and save to csv file
Data = utils.vec2mat(Data,matrix_cols=9,num_matrix=4)
with open(r'.\Data\Unassist\unassist_unnormalized_moment_data.csv', 'wb') as f:
  f.write(bytes(utils.listToString(Headers)+'\n','UTF-8'))
  np.savetxt(f, Data, fmt='%s', delimiter=",")
#*****************************
# joints stiffness
fig, ax = plt.subplots(nrows=2,ncols=4,figsize=(13.4, 10.8))
# hip loaded boxplot
# box plot
names = ['extension,\n loaded','flexion,\n loaded','total,\n loaded']
x = np.arange(1,len(names)+1,1)
data = [hip_loaded_stiffness_dict['hip_extension_stiffness'],hip_loaded_stiffness_dict['hip_flexion_stiffness'],hip_loaded_stiffness_dict['hip_total_stiffness']]
bp = ax[0,0].boxplot(data, patch_artist=True)
utils.beautiful_boxplot(bp)
ax[0,0].set_ylabel('Joint Stiffness (N-m/kg-rad)')
ax[0,0].set_xticks(x)
ax[0,0].set_yticks([0,2,4,6,8,10])
ax[0,0].set_ylim([0,10])
ax[0,0].set_xticklabels(names)
ax[0,0].set_title('hip stiffness, loaded')
utils.no_top_right(ax[0,0])
# bar plot
data = [hip_loaded_stiffness_dict['mean_hip_extension_stiffness'],hip_loaded_stiffness_dict['mean_hip_flexion_stiffness'],hip_loaded_stiffness_dict['mean_hip_total_stiffness']]
err = [hip_loaded_stiffness_dict['std_hip_extension_stiffness'],hip_loaded_stiffness_dict['std_hip_flexion_stiffness'],hip_loaded_stiffness_dict['std_hip_total_stiffness']]
barplot = ax[0,2].bar(names, data, yerr=err, align='center',color='gray',width=0.45, ecolor='black', capsize=10)
barplot[0].set_color('darkgreen')
barplot[1].set_color('mediumseagreen')
barplot[2].set_color('lightgreen')
ax[0,2].set_ylabel('Joint Stiffness (N-m/kg-rad)')
ax[0,2].set_yticks([0,2,4,6,8,10])
ax[0,2].set_xticks(names)
ax[0,2].set_xticklabels(names)
ax[0,2].set_title('hip stiffness, loaded')
utils.no_top_right(ax[0,2])
# hip noload boxplot
# box plot
names = ['extension,\n noload','flexion,\n noload','total,\n noload']
x = np.arange(1,len(names)+1,1)
data = [hip_noload_stiffness_dict['hip_extension_stiffness'],hip_noload_stiffness_dict['hip_flexion_stiffness'],hip_noload_stiffness_dict['hip_total_stiffness']]
bp = ax[0,1].boxplot(data, patch_artist=True)
utils.beautiful_boxplot(bp)
ax[0,1].set_ylabel('Joint Stiffness (N-m/kg-rad)')
ax[0,1].set_xticks(x)
ax[0,1].set_yticks([0,2,4,6,8,10])
ax[0,1].set_xticklabels(names)
ax[0,1].set_title('hip stiffness, noload')
utils.no_top_right(ax[0,1])
# bar plot
data = [hip_noload_stiffness_dict['mean_hip_extension_stiffness'],hip_noload_stiffness_dict['mean_hip_flexion_stiffness'],hip_noload_stiffness_dict['mean_hip_total_stiffness']]
err = [hip_noload_stiffness_dict['std_hip_extension_stiffness'],hip_noload_stiffness_dict['std_hip_flexion_stiffness'],hip_noload_stiffness_dict['std_hip_total_stiffness']]
barplot = ax[0,3].bar(names, data, yerr=err, align='center',color='gray',width=0.45, ecolor='black', capsize=10)
barplot[0].set_color('darkgreen')
barplot[1].set_color('mediumseagreen')
barplot[2].set_color('lightgreen')
ax[0,3].set_ylabel('Joint Stiffness (N-m/kg-rad)')
ax[0,3].set_yticks([0,2,4,6,8,10])
ax[0,3].set_xticks(names)
ax[0,3].set_xticklabels(names)
ax[0,3].set_title('hip stiffness, noload')
utils.no_top_right(ax[0,3])
# knee loaded boxplot
# box plot
names = ['extension,\n loaded','flexion,\n loaded','total,\n loaded']
x = np.arange(1,len(names)+1,1)
data = [knee_loaded_stiffness_dict['knee_extension_stiffness'],knee_loaded_stiffness_dict['knee_flexion_stiffness'],knee_loaded_stiffness_dict['knee_total_stiffness']]
bp = ax[1,0].boxplot(data, patch_artist=True)
utils.beautiful_boxplot(bp)
ax[1,0].set_ylabel('Joint Stiffness (N-m/kg-rad)')
ax[1,0].set_xticks(x)
ax[1,0].set_yticks([0,2,4,6,8,10])
ax[1,0].set_ylim([0,10])
ax[1,0].set_xticklabels(names)
ax[1,0].set_title('knee stiffness, loaded')
utils.no_top_right(ax[1,0])
# bar plot
data = [knee_loaded_stiffness_dict['mean_knee_extension_stiffness'],knee_loaded_stiffness_dict['mean_knee_flexion_stiffness'],knee_loaded_stiffness_dict['mean_knee_total_stiffness']]
err = [knee_loaded_stiffness_dict['std_knee_extension_stiffness'],knee_loaded_stiffness_dict['std_knee_flexion_stiffness'],knee_loaded_stiffness_dict['std_knee_total_stiffness']]
barplot = ax[1,2].bar(names, data, yerr=err, align='center',color='gray',width=0.45, ecolor='black', capsize=10)
barplot[0].set_color('darkgreen')
barplot[1].set_color('mediumseagreen')
barplot[2].set_color('lightgreen')
ax[1,2].set_ylabel('Joint Stiffness (N-m/kg-rad)')
ax[1,2].set_yticks([0,2,4,6,8,10])
ax[1,2].set_xticks(names)
ax[1,2].set_xticklabels(names)
ax[1,2].set_title('knee stiffness, loaded')
utils.no_top_right(ax[1,2])

# knee noload boxplot
# box plot
names = ['extension,\n noload','flexion,\n noload','total,\n noload']
x = np.arange(1,len(names)+1,1)
data = [knee_noload_stiffness_dict['knee_extension_stiffness'],knee_noload_stiffness_dict['knee_flexion_stiffness'],knee_noload_stiffness_dict['knee_total_stiffness']]
bp = ax[1,1].boxplot(data, patch_artist=True)
utils.beautiful_boxplot(bp)
ax[1,1].set_ylabel('Joint Stiffness (N-m/kg-rad)')
ax[1,1].set_xticks(x)
ax[1,1].set_yticks([0,2,4,6,8,10])
ax[1,1].set_ylim([0,10])
ax[1,1].set_xticklabels(names)
ax[1,1].set_title('knee stiffness, noload')
utils.no_top_right(ax[1,1])
# bar plot
data = [knee_noload_stiffness_dict['mean_knee_extension_stiffness'],knee_noload_stiffness_dict['mean_knee_flexion_stiffness'],knee_noload_stiffness_dict['mean_knee_total_stiffness']]
err = [knee_noload_stiffness_dict['std_knee_extension_stiffness'],knee_noload_stiffness_dict['std_knee_flexion_stiffness'],knee_noload_stiffness_dict['std_knee_total_stiffness']]
barplot = ax[1,3].bar(names, data, yerr=err, align='center',color='gray',width=0.45, ecolor='black', capsize=10)
barplot[0].set_color('darkgreen')
barplot[1].set_color('mediumseagreen')
barplot[2].set_color('lightgreen')
ax[1,3].set_ylabel('Joint Stiffness (N-m/kg-rad)')
ax[1,3].set_yticks([0,2,4,6,8,10])
ax[1,3].set_xticks(names)
ax[1,3].set_xticklabels(names)
ax[1,3].set_title('knee stiffness, noload')
utils.no_top_right(ax[1,3])
fig.tight_layout(h_pad=-1, w_pad=-1.5)
fig.subplots_adjust(top=0.95, bottom=0.075, left=0.100, right=0.975,hspace=0.40,wspace=0.25)
plt.show()
fig.savefig('./Figures/Unassist/Paper_Figure_Stiffness_Box_Bar_Plot.pdf',orientation='landscape',bbox_inches='tight')


#*****************************
fig = plt.figure(num='Hip Joint Stiffness',figsize=(12, 10))
gridsize = (3, 2)
ax1 = plt.subplot2grid(gridsize, (0, 0), colspan=2, rowspan=2) # stiffness plot
ax2 = plt.subplot2grid(gridsize, (2, 0)) # kinematics plot
ax3 = plt.subplot2grid(gridsize, (2, 1)) # moment plot
# hip loaded case
hip_stiffness_plot_dic = {'loaded_toe_off':loaded_mean_toe_off,'noload_toe_off':noload_mean_toe_off,'kinematics':utils.smooth(np.deg2rad(rra_dataset['mean_loaded_hipjoint_kinematics']),5),
                          'kinematics_std':utils.smooth(np.deg2rad(rra_dataset['std_loaded_hipjoint_kinematics']),5),'moment':utils.smooth(mean_loaded_hipmuscles_moment,5),
                          'moment_std':utils.smooth(std_loaded_hipmuscles_moment,5),'color':'k','toe_off_color':'grey','label':'hip joint',
                          'phases_kinematics':hip_loaded_linear_angle_dict,'phases_moment':hip_loaded_linear_moment_dict,'phases_fitted_line':hip_loaded_fitted_line}
utils.plot_stiffness(plot_dic = hip_stiffness_plot_dic, load_condition='loaded',\
                     kinematics_ticks=[-0.5,0,0.5,1],moment_ticks=[-1,-0.5,0,0.5,1,1.5],
                     ax1=ax1,ax2=ax2,ax3=ax3,joint='hip',plot_phases=True,plot_fitted_line=True)
# hip noload case
hip_stiffness_plot_dic = {'loaded_toe_off':loaded_mean_toe_off,'noload_toe_off':noload_mean_toe_off,'kinematics':utils.smooth(np.deg2rad(rra_dataset['mean_noload_hipjoint_kinematics']),5),
                          'kinematics_std':utils.smooth(np.deg2rad(rra_dataset['std_noload_hipjoint_kinematics']),5),'moment':utils.smooth(mean_noload_hipmuscles_moment,5),
                          'moment_std':utils.smooth(std_noload_hipmuscles_moment,5),'color':'xkcd:irish green','toe_off_color':'xkcd:shamrock green','label':'hip joint',
                          'phases_kinematics':hip_noload_linear_angle_dict,'phases_moment':hip_noload_linear_moment_dict,'phases_fitted_line':hip_noload_fitted_line}
utils.plot_stiffness(plot_dic = hip_stiffness_plot_dic, load_condition='noload',
                     kinematics_ticks=[-0.5,0,0.5,1],moment_ticks=[-1,-0.5,0,0.5,1,1.5],
                     ax1=ax1,ax2=ax2,ax3=ax3,joint='hip',plot_phases=True,plot_fitted_line=True)
fig.tight_layout(h_pad=-1, w_pad=-1.5)
fig.subplots_adjust(top=0.98, bottom=0.075, left=0.100, right=0.975,hspace=0.45,wspace=0.15)
plt.show()
fig.savefig('./Figures/Unassist/HipJointStiffness.pdf',orientation='landscape',bbox_inches='tight')

#*****************************
# knee joint stiffness
fig = plt.figure(num='Knee Joint Stiffness',figsize=(12, 10))
gridsize = (3, 2)
ax1 = plt.subplot2grid(gridsize, (0, 0), colspan=2, rowspan=2) # stiffness plot
ax2 = plt.subplot2grid(gridsize, (2, 0)) # kinematics plot
ax3 = plt.subplot2grid(gridsize, (2, 1)) # moment plot
# knee loaded case
knee_stiffness_plot_dic = {'loaded_toe_off':loaded_mean_toe_off,'noload_toe_off':noload_mean_toe_off,'kinematics':utils.smooth(np.deg2rad(rra_dataset['mean_loaded_kneejoint_kinematics']),5),
                          'kinematics_std':utils.smooth(np.deg2rad(rra_dataset['std_loaded_kneejoint_kinematics']),5),'moment':utils.smooth(mean_loaded_kneemuscles_moment,5),
                          'moment_std':utils.smooth(std_loaded_kneemuscles_moment,5),'color':'k','toe_off_color':'grey','label':'hip joint',
                          'phases_kinematics':knee_loaded_linear_angle_dict,'phases_moment':knee_loaded_linear_moment_dict,'phases_fitted_line':knee_loaded_fitted_line}
utils.plot_stiffness(plot_dic = knee_stiffness_plot_dic, load_condition='loaded',\
                     kinematics_ticks=[0,10,20,30,40,50,60,70,80],moment_ticks=[-1,-0.5,0,0.5,1,1.5],
                     ax1=ax1,ax2=ax2,ax3=ax3,joint='knee',plot_phases=True,plot_fitted_line=True)
# knee noload case
knee_stiffness_plot_dic = {'loaded_toe_off':loaded_mean_toe_off,'noload_toe_off':noload_mean_toe_off,'kinematics':utils.smooth(np.deg2rad(rra_dataset['mean_noload_kneejoint_kinematics']),5),
                          'kinematics_std':utils.smooth(np.deg2rad(rra_dataset['std_noload_kneejoint_kinematics']),5),'moment':utils.smooth(mean_noload_kneemuscles_moment,5),
                          'moment_std':utils.smooth(std_noload_kneemuscles_moment,5),'color':'xkcd:irish green','toe_off_color':'xkcd:shamrock green','label':'hip joint',
                          'phases_kinematics':knee_noload_linear_angle_dict,'phases_moment':knee_noload_linear_moment_dict,'phases_fitted_line':knee_noload_fitted_line}
utils.plot_stiffness(plot_dic = knee_stiffness_plot_dic, load_condition='noload',
                     kinematics_ticks=[0,0.5,1,1.5],moment_ticks=[-1,-0.5,0,0.5,1,1.5],
                     ax1=ax1,ax2=ax2,ax3=ax3,joint='knee',plot_phases=True,plot_fitted_line=True)
fig.tight_layout(h_pad=-1, w_pad=-1.5)
fig.subplots_adjust(top=0.98, bottom=0.075, left=0.100, right=0.975,hspace=0.45,wspace=0.15)
plt.show()
fig.savefig('./Figures/Unassist/KneeJointStiffness.pdf',orientation='landscape',bbox_inches='tight')

#****************************************************************************************************
mean_cellText = np.array([[hip_loaded_stiffness_dict['mean_hip_extension_stiffness'],hip_loaded_stiffness_dict['mean_hip_flexion_stiffness'],hip_loaded_stiffness_dict['mean_hip_total_stiffness']],\
           [hip_noload_stiffness_dict['mean_hip_extension_stiffness'],hip_noload_stiffness_dict['mean_hip_flexion_stiffness'],hip_noload_stiffness_dict['mean_hip_total_stiffness']],\
           [knee_loaded_stiffness_dict['mean_knee_extension_stiffness'],knee_loaded_stiffness_dict['mean_knee_flexion_stiffness'],knee_loaded_stiffness_dict['mean_knee_total_stiffness']],\
           [knee_noload_stiffness_dict['mean_knee_extension_stiffness'],knee_noload_stiffness_dict['mean_knee_flexion_stiffness'],knee_noload_stiffness_dict['mean_knee_total_stiffness']]])
std_cellText = np.array([[hip_loaded_stiffness_dict['std_hip_extension_stiffness'],hip_loaded_stiffness_dict['std_hip_flexion_stiffness'],hip_loaded_stiffness_dict['std_hip_total_stiffness']],\
           [hip_noload_stiffness_dict['std_hip_extension_stiffness'],hip_noload_stiffness_dict['std_hip_flexion_stiffness'],hip_noload_stiffness_dict['std_hip_total_stiffness']],\
           [knee_loaded_stiffness_dict['std_knee_extension_stiffness'],knee_loaded_stiffness_dict['std_knee_flexion_stiffness'],knee_loaded_stiffness_dict['std_knee_total_stiffness']],\
           [knee_noload_stiffness_dict['std_knee_extension_stiffness'],knee_noload_stiffness_dict['std_knee_flexion_stiffness'],knee_noload_stiffness_dict['std_knee_total_stiffness']]])
Rsquare_cellText = np.array([[hip_loaded_Rsquare_dict['mean_hip_extension_R_square'],hip_loaded_Rsquare_dict['mean_hip_flexion_R_square'],hip_loaded_Rsquare_dict['mean_hip_total_R_square']],\
           [hip_noload_Rsquare_dict['mean_hip_extension_R_square'],hip_noload_Rsquare_dict['mean_hip_flexion_R_square'],hip_noload_Rsquare_dict['mean_hip_total_R_square']],\
           [knee_loaded_Rsquare_dict['mean_knee_extension_R_square'],knee_loaded_Rsquare_dict['mean_knee_flexion_R_square'],knee_loaded_Rsquare_dict['mean_knee_total_R_square']],\
           [knee_noload_Rsquare_dict['mean_knee_extension_R_square'],knee_noload_Rsquare_dict['mean_knee_flexion_R_square'],knee_noload_Rsquare_dict['mean_knee_total_R_square']]])
main_cellText = np.concatenate((mean_cellText,std_cellText,Rsquare_cellText),axis=1)
rows = ['mean extension stiffness','mean flexion stiffness','mean total stiffness',\
           'std extension stiffness','std flexion stiffness','std total stiffness',\
           'Rsquare extension stiffness','Rsquare flexion stiffness','Rsquare total stiffness']
columns = ['hip, loaded','hip, noload','knee, loaded','knee, noload']
fig, ax = plt.subplots(figsize=(12,6))
table = ax.table(cellText=np.transpose(main_cellText.round(3)),rowLabels=rows,colLabels=columns,loc='center')
table.scale(1,2)
table.set_fontsize(15)
ax.axis('off')
fig.subplots_adjust(top=0.98, bottom=0.075, left=0.25, right=0.975,hspace=0.45,wspace=0.15)
fig.savefig('./Figures/Unassist/Stiffness_Table.pdf',orientation='landscape',bbox_inches='tight')
plt.show()
