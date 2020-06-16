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
# muscles activation dataset
directory = './Data/Unassist/*_activation.csv'
files = enumerate(glob.iglob(directory), 1)
musclesactivation_dataset = {pathlib.PurePath(f[1]).stem: np.loadtxt(f[1], delimiter=',') for f in files}
# muscles metabolic rate
directory = './Data/Unassist/*_metabolic_rate.csv'
files = enumerate(glob.iglob(directory), 1)
musclesmetabolicrate_dataset = {pathlib.PurePath(f[1]).stem: np.loadtxt(f[1], delimiter=',') for f in files}
# total metabolic rate
directory = './Data/Unassist/*_energy.csv'
files = enumerate(glob.iglob(directory), 1)
total_metabolicrate_dataset = {pathlib.PurePath(f[1]).stem: np.loadtxt(f[1], delimiter=',') for f in files}
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
loaded_hipmuscles_moment = utils.normalize_direction_data(musclesmoment_dataset['loaded_hip_musclesmoment'],gl_noload,normalize=False,direction=True)
noload_hipmuscles_moment = utils.normalize_direction_data(musclesmoment_dataset['noload_hip_musclesmoment'],gl_noload,normalize=False,direction=True)
mean_loaded_hipmuscles_moment,std_loaded_hipmuscles_moment = utils.mean_std_over_subjects(loaded_hipmuscles_moment,avg_trials=False)
mean_noload_hipmuscles_moment,std_noload_hipmuscles_moment = utils.mean_std_over_subjects(noload_hipmuscles_moment,avg_trials=False)

# knee muscles moment
normal_loaded_kneemuscles_moment = utils.normalize_direction_data(musclesmoment_dataset['loaded_knee_musclesmoment'],gl_noload,direction=True)
normal_noload_kneemuscles_moment = utils.normalize_direction_data(musclesmoment_dataset['noload_knee_musclesmoment'],gl_noload,direction=True)
mean_norm_loaded_kneemuscles_moment,std_norm_loaded_kneemuscles_moment = utils.mean_std_over_subjects(normal_loaded_kneemuscles_moment,avg_trials=False)
mean_norm_noload_kneemuscles_moment,std_norm_noload_kneemuscles_moment = utils.mean_std_over_subjects(normal_noload_kneemuscles_moment,avg_trials=False)
# not normalized
loaded_kneemuscles_moment = utils.normalize_direction_data(musclesmoment_dataset['loaded_knee_musclesmoment'],gl_noload,normalize=False,direction=True)
noload_kneemuscles_moment = utils.normalize_direction_data(musclesmoment_dataset['noload_knee_musclesmoment'],gl_noload,normalize=False,direction=True)
mean_loaded_kneemuscles_moment,std_loaded_kneemuscles_moment = utils.mean_std_over_subjects(loaded_kneemuscles_moment,avg_trials=False)
mean_noload_kneemuscles_moment,std_noload_kneemuscles_moment = utils.mean_std_over_subjects(noload_kneemuscles_moment,avg_trials=False)

# muscles activation
normal_loaded_muscles_activation = utils.normalize_direction_data(musclesactivation_dataset['loaded_ninemuscles_activation'],gl_noload,direction=False,normalize=False)
normal_noload_muscles_activation = utils.normalize_direction_data(musclesactivation_dataset['noload_ninemuscles_activation'],gl_noload,direction=False,normalize=False)
mean_norm_loaded_muscles_activation,std_norm_loaded_muscles_activation = utils.mean_std_muscles_subjects(normal_loaded_muscles_activation)
mean_norm_noload_muscles_activation,std_norm_noload_muscles_activation = utils.mean_std_muscles_subjects(normal_noload_muscles_activation)

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

#####################################################################################
# Write final data to csv file.
# TODO: optimize data saving method.
# Headers
Headers = ['mean_norm_loaded_hipmuscles_moment','std_norm_loaded_hipmuscles_moment','mean_norm_noload_hipmuscles_moment','std_norm_noload_hipmuscles_moment',\
       'mean_norm_loaded_kneemuscles_moment','std_norm_loaded_kneemuscles_moment','mean_norm_noload_kneemuscles_moment','std_norm_noload_kneemuscles_moment']
mean_norm_loaded_muscles_activation_header = utils.muscles_header('mean_norm_loaded_muscles_activation')
mean_norm_noload_muscles_activation_header = utils.muscles_header('mean_norm_noload_muscles_activation')
std_norm_loaded_muscles_activation_header = utils.muscles_header('std_norm_loaded_muscles_activation')
std_norm_noload_muscles_activation_header = utils.muscles_header('std_norm_noload_muscles_activation')
Headers.extend(mean_norm_loaded_muscles_activation_header)
Headers.extend(std_norm_loaded_muscles_activation_header)
Headers.extend(mean_norm_noload_muscles_activation_header)
Headers.extend(std_norm_noload_muscles_activation_header)
# Dataset
Data =[mean_norm_loaded_hipmuscles_moment,std_norm_loaded_hipmuscles_moment,mean_norm_noload_hipmuscles_moment,std_norm_noload_hipmuscles_moment,\
       mean_norm_loaded_kneemuscles_moment,std_norm_loaded_kneemuscles_moment,mean_norm_noload_kneemuscles_moment,std_norm_noload_kneemuscles_moment,\
       mean_norm_loaded_muscles_activation,std_norm_loaded_muscles_activation,mean_norm_noload_muscles_activation,std_norm_noload_muscles_activation]
# List of numpy vectors to a numpy ndarray and save to csv file
Data = utils.vec2mat(Data,matrix_cols=9,num_matrix=4)
with open(r'.\Data\Unassist\unassist_final_data.csv', 'wb') as f:
  f.write(bytes(utils.listToString(Headers)+'\n','UTF-8'))
  np.savetxt(f, Data, fmt='%s', delimiter=",")
#####################################################################################
# moment dataset without normalizing for stiffness plots
Headers = ['mean_loaded_hipmuscles_moment','std_loaded_hipmuscles_moment','mean_noload_hipmuscles_moment','std_noload_hipmuscles_moment',\
       'mean_loaded_kneemuscles_moment','std_loaded_kneemuscles_moment','mean_noload_kneemuscles_moment','std_noload_kneemuscles_moment']
# Dataset
Data =[mean_loaded_hipmuscles_moment,std_loaded_hipmuscles_moment,mean_noload_hipmuscles_moment,std_noload_hipmuscles_moment,\
       mean_loaded_kneemuscles_moment,std_loaded_kneemuscles_moment,mean_noload_kneemuscles_moment,std_noload_kneemuscles_moment]
# List of numpy vectors to a numpy ndarray and save to csv file
Data = utils.vec2mat(Data,matrix_cols=9,num_matrix=4)
with open(r'.\Data\Unassist\unassist_stiffness_data.csv', 'wb') as f:
  f.write(bytes(utils.listToString(Headers)+'\n','UTF-8'))
  np.savetxt(f, Data, fmt='%s', delimiter=",")
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
                                      'label':'loaded','std':std_norm_loaded_muscles_activation,'avg_toeoff':loaded_mean_toe_off}
muscles_activation_noload_plot_dic = {'pgc':gait_cycle,'avg':mean_norm_noload_muscles_activation,'muscle_group': 'nine_muscles',
                                      'label':'noload','std':std_norm_noload_muscles_activation,'avg_toeoff':noload_mean_toe_off}

#*****************************
'''
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
fig, ax = plt.subplots(num='Muscles Activation',figsize=(8.4, 6.8))
utils.plot_muscles_avg(plot_dic=muscles_activation_loaded_plot_dic,color='k',is_std=True)
utils.plot_muscles_avg(plot_dic=muscles_activation_noload_plot_dic,toeoff_color='xkcd:shamrock green',color='xkcd:irish green',is_std=True)
plt.legend(loc='best',frameon=False)
plt.show()
fig.tight_layout()
fig.savefig('./Figures/Unassist/NineMusclesActivation.pdf',orientation='landscape',bbox_inches='tight')

#*****************************
# muscles metabolic rate figure
fig = plt.figure(num='Muscles Metabolic Rate',figsize=(16.8, 13.6))
utils.muscles_whisker_bar_plot(musclesmetabolicrate_dataset['noload_muscles_metabolic_rate'],musclesmetabolicrate_dataset['loaded_muscles_metabolic_rate'] )
plt.legend(loc='best',frameon=False)
fig.tight_layout(h_pad=-1, w_pad=-1.5)
fig.subplots_adjust(top=0.98, bottom=0.075, left=0.100, right=0.975,hspace=0.45,wspace=0.40)
fig.savefig('./Figures/Unassist/MusclesMetabolicRate.pdf',orientation='landscape',bbox_inches='tight')
plt.show()

# muscles metabolic rate figure
fig = plt.figure(num='Muscles Metabolic Rate',figsize=(25, 25))
utils.muscles_whisker_bar_plot(musclesmetabolicrate_dataset['noload_muscles_metabolic_rate'],musclesmetabolicrate_dataset['loaded_muscles_metabolic_rate'],which_plot='bar' )
plt.legend(loc='best',frameon=False)
fig.tight_layout(h_pad=-1, w_pad=-1.5)
fig.subplots_adjust(top=0.98, bottom=0.075, left=0.100, right=0.975,hspace=0.45,wspace=0.40)
fig.savefig('./Figures/Unassist/MusclesMetabolicRate_BarPlot.pdf',orientation='landscape',bbox_inches='tight')
plt.show()

# muscles contribution on metabolic rate wrt noload unassist metabolic rate
fig = plt.figure(num='Muscles Metabolic Rate',figsize=(9.6, 16.8))
utils.muscles_metabolics_contribution(musclesmetabolicrate_dataset['loaded_muscles_metabolic_rate'],total_metabolicrate_dataset['noload_metabolics_energy'],
                                      musclesmetabolicrate_dataset['noload_muscles_metabolic_rate'],total_metabolicrate_dataset['noload_metabolics_energy'],
                                      xticks=[0,2,4,6,8,10])
plt.legend(loc='best',frameon=False)
plt.title('Muscles contribution change by loading subjects')
fig.tight_layout(h_pad=-1, w_pad=-1.5)
fig.savefig('./Figures/Unassist/MusclesMetabolic_Contribution_WRT_noload.pdf',orientation='landscape',bbox_inches='tight')
plt.show()
# muscles contribution on metabolic rate
fig = plt.figure(num='Muscles Metabolic Rate',figsize=(9.6, 16.8))
utils.muscles_metabolics_contribution(musclesmetabolicrate_dataset['loaded_muscles_metabolic_rate'],total_metabolicrate_dataset['loaded_metabolics_energy'],
                                      musclesmetabolicrate_dataset['noload_muscles_metabolic_rate'],total_metabolicrate_dataset['noload_metabolics_energy'])
plt.legend(loc='best',frameon=False)
plt.title('Muscles contributions in different load condition')
fig.tight_layout(h_pad=-1, w_pad=-1.5)
fig.savefig('./Figures/Unassist/MusclesMetabolic_Contribution.pdf',orientation='landscape',bbox_inches='tight')
plt.show()
'''
#*****************************
# joints stiffness
ax,fig = plt.subplots(nrows=2,ncols=4)
ax[0,1].boxplot()
# hip joint stiffness
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
                     kinematics_ticks=[-0.5,0,0.5,1],moment_ticks=[-80,-40,0,40,80,120],
                     ax1=ax1,ax2=ax2,ax3=ax3,joint='hip',plot_phases=True,plot_fitted_line=True)
# hip noload case
hip_stiffness_plot_dic = {'loaded_toe_off':loaded_mean_toe_off,'noload_toe_off':noload_mean_toe_off,'kinematics':utils.smooth(np.deg2rad(rra_dataset['mean_noload_hipjoint_kinematics']),5),
                          'kinematics_std':utils.smooth(np.deg2rad(rra_dataset['std_noload_hipjoint_kinematics']),5),'moment':utils.smooth(mean_noload_hipmuscles_moment,5),
                          'moment_std':utils.smooth(std_noload_hipmuscles_moment,5),'color':'xkcd:irish green','toe_off_color':'xkcd:shamrock green','label':'hip joint',
                          'phases_kinematics':hip_noload_linear_angle_dict,'phases_moment':hip_noload_linear_moment_dict,'phases_fitted_line':hip_noload_fitted_line}
utils.plot_stiffness(plot_dic = hip_stiffness_plot_dic, load_condition='noload',
                     kinematics_ticks=[-0.5,0,0.5,1],moment_ticks=[-80,-40,0,40,80,120],
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
                     kinematics_ticks=[0,10,20,30,40,50,60,70,80],moment_ticks=[-1,-0.5,0,0.5,1,1.5,2],
                     ax1=ax1,ax2=ax2,ax3=ax3,joint='knee',plot_phases=True,plot_fitted_line=True)
# knee noload case
knee_stiffness_plot_dic = {'loaded_toe_off':loaded_mean_toe_off,'noload_toe_off':noload_mean_toe_off,'kinematics':utils.smooth(np.deg2rad(rra_dataset['mean_noload_kneejoint_kinematics']),5),
                          'kinematics_std':utils.smooth(np.deg2rad(rra_dataset['std_noload_kneejoint_kinematics']),5),'moment':utils.smooth(mean_noload_kneemuscles_moment,5),
                          'moment_std':utils.smooth(std_noload_kneemuscles_moment,5),'color':'xkcd:irish green','toe_off_color':'xkcd:shamrock green','label':'hip joint',
                          'phases_kinematics':knee_noload_linear_angle_dict,'phases_moment':knee_noload_linear_moment_dict,'phases_fitted_line':knee_noload_fitted_line}
utils.plot_stiffness(plot_dic = knee_stiffness_plot_dic, load_condition='noload',
                     kinematics_ticks=[0,0.5,1,1.5],moment_ticks=[-60,-30,0,30,60,90],
                     ax1=ax1,ax2=ax2,ax3=ax3,joint='knee',plot_phases=True,plot_fitted_line=True)
fig.tight_layout(h_pad=-1, w_pad=-1.5)
fig.subplots_adjust(top=0.98, bottom=0.075, left=0.100, right=0.975,hspace=0.45,wspace=0.15)
plt.show()
fig.savefig('./Figures/Unassist/KneeJointStiffness.pdf',orientation='landscape',bbox_inches='tight')

