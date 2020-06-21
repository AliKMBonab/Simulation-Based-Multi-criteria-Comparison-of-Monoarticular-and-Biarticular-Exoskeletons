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
unassist_stiffness_dataset = utils.csv2numpy('./Data/Unassist/unassist_meanstd_stiffness_data.csv') 
ideal_stiffness_dataset = utils.csv2numpy('./Data/Ideal/ideal_exos_stiffness_dataset.csv')
# pareto exo torque dataset
directory = './Data/Specific_Weights/*_torque.csv'
files = enumerate(glob.iglob(directory), 1)
exo_torque_dataset = {pathlib.PurePath(f[1]).stem: np.loadtxt(f[1], delimiter=',') for f in files}
# pareto joint kinematics dataset
directory = './Data/Specific_Weights/*_kinematics.csv'
files = enumerate(glob.iglob(directory), 1)
joint_kinematics_dataset = {pathlib.PurePath(f[1]).stem: np.loadtxt(f[1], delimiter=',') for f in files}
# pareto muscles moment dataset
directory = './Data/Specific_Weights/*_moment.csv'
files = enumerate(glob.iglob(directory), 1)
muscles_moment_dataset = {pathlib.PurePath(f[1]).stem: np.loadtxt(f[1], delimiter=',') for f in files}
# gls
gl_noload = {'noload_subject{}_trial{}'.format(i,j): utils.construct_gl_mass_side(subjectno=i,trialno=j,loadcond='noload') for i in subjects for j in trials_num}
gl_loaded = {'loaded_subject{}_trial{}'.format(i,j): utils.construct_gl_mass_side(subjectno=i,trialno=j,loadcond='loaded') for i in subjects for j in trials_num}
#####################################################################################
# Processing Data
# toe-off
noload_mean_toe_off,_,loaded_mean_toe_off,_,noload_mean_toe_off_subjects,loaded_mean_toe_off_subjects = utils.toe_off_avg_std(gl_noload,gl_loaded,subjects=True)
# muscles stiffness
# noload
bi_noload_muscles_paretofront_stiffness,bi_noload_muscles_paretofront_R_square = utils.paretofront_quasi_stiffness(joint_kinematics_dataset,muscles_moment_dataset,\
                                                                               noload_mean_toe_off_subjects,device='biarticular',\
                                                                               loadcondition='noload',gl_noload=gl_noload,muscles_actuator='muscles')
mono_noload_muscles_paretofront_stiffness,mono_noload_muscles_paretofront_R_square = utils.paretofront_quasi_stiffness(joint_kinematics_dataset,muscles_moment_dataset,\
                                                                               noload_mean_toe_off_subjects,device='monoarticular',\
                                                                               loadcondition='noload',gl_noload=gl_noload,muscles_actuator='muscles')
# loaded
bi_loaded_muscles_paretofront_stiffness,bi_loaded_muscles_paretofront_R_square = utils.paretofront_quasi_stiffness(joint_kinematics_dataset,muscles_moment_dataset,\
                                                                               loaded_mean_toe_off_subjects,device='biarticular',\
                                                                               loadcondition='loaded',gl_noload=gl_noload,muscles_actuator='muscles')
mono_loaded_muscles_paretofront_stiffness,mono_loaded_muscles_paretofront_R_square = utils.paretofront_quasi_stiffness(joint_kinematics_dataset,muscles_moment_dataset,\
                                                                               loaded_mean_toe_off_subjects,device='monoarticular',\
                                                                               loadcondition='loaded',gl_noload=gl_noload,muscles_actuator='muscles')
# actuators stiffness
# noload
bi_noload_actuators_paretofront_stiffness,bi_noload_actuators_paretofront_R_square = utils.paretofront_quasi_stiffness(joint_kinematics_dataset,exo_torque_dataset,\
                                                                               noload_mean_toe_off_subjects,device='biarticular',\
                                                                               loadcondition='noload',gl_noload=gl_noload,muscles_actuator='actuator')
mono_noload_actuators_paretofront_stiffness,mono_noload_actuators_paretofront_R_square = utils.paretofront_quasi_stiffness(joint_kinematics_dataset,exo_torque_dataset,\
                                                                               noload_mean_toe_off_subjects,device='monoarticular',\
                                                                               loadcondition='noload',gl_noload=gl_noload,muscles_actuator='actuator')
# loaded
bi_loaded_actuators_paretofront_stiffness,bi_loaded_actuators_paretofront_R_square = utils.paretofront_quasi_stiffness(joint_kinematics_dataset,exo_torque_dataset,\
                                                                               loaded_mean_toe_off_subjects,device='biarticular',\
                                                                               loadcondition='loaded',gl_noload=gl_noload,muscles_actuator='actuator')
mono_loaded_actuators_paretofront_stiffness,mono_loaded_actuators_paretofront_R_square = utils.paretofront_quasi_stiffness(joint_kinematics_dataset,exo_torque_dataset,\
                                                                               loaded_mean_toe_off_subjects,device='monoarticular',\
                                                                               loadcondition='loaded',gl_noload=gl_noload,muscles_actuator='actuator')
#####################################################################################
# Paretofront
# indices
bi_loaded_indices = np.array([25,24,23,22,21,17,16,13,12,11,6,1])
mono_loaded_indices = np.array([25,20,15,10,5,4,3,2,1])
bi_noload_indices = np.array([25,24,23,22,21,19,18,17,13,12,11,1])
mono_noload_indices = np.array([25,20,15,14,13,8,7,6,2,1])

# figure
fig = plt.figure(num='paretofront joint stiffness',figsize=[6.4*2, 4.8*2.5])
# noload biarticular hip
ax = plt.subplot(2,4,1)
plot_dict = {'unassist_data':unassist_stiffness_dataset,'ideal_data':ideal_stiffness_dataset,
             'paretofront_dict':bi_noload_muscles_paretofront_stiffness,'indices':bi_noload_indices}
utils.plot_paretofront_stiffness(plot_dict,device='biarticular',loadcondition='noload',joint='hip',muscles_actuator='muscles')
ax.set_title('biarticular hip joint\nstiffness, noload')
ax.set_xticks([1,2,3,4])
ax.set_xlim([0,4])
ax.set_ylabel('optimal devices')
# noload biarticular knee
ax = plt.subplot(2,4,2)
plot_dict = {'unassist_data':unassist_stiffness_dataset,'ideal_data':ideal_stiffness_dataset,
             'paretofront_dict':bi_noload_muscles_paretofront_stiffness,'indices':bi_noload_indices}
utils.plot_paretofront_stiffness(plot_dict,device='biarticular',loadcondition='noload',joint='knee',muscles_actuator='muscles')
ax.set_title('biarticular knee joint\nstiffness, noload')
ax.set_xticks([1,2,3,4,5])
ax.set_xlim([0,5])

# noload monoarticular hip
ax = plt.subplot(2,4,3)
plot_dict = {'unassist_data':unassist_stiffness_dataset,'ideal_data':ideal_stiffness_dataset,
             'paretofront_dict':mono_noload_muscles_paretofront_stiffness,'indices':mono_noload_indices}
utils.plot_paretofront_stiffness(plot_dict,device='monoarticular',loadcondition='noload',joint='hip',muscles_actuator='muscles')
ax.set_title('monoarticular hip joint\nstiffness, noload')
ax.set_xticks([1,2,3,4])
ax.set_xlim([0,4])

# noload monoarticular knee
ax = plt.subplot(2,4,4)
plot_dict = {'unassist_data':unassist_stiffness_dataset,'ideal_data':ideal_stiffness_dataset,
             'paretofront_dict':mono_noload_muscles_paretofront_stiffness,'indices':mono_noload_indices}
utils.plot_paretofront_stiffness(plot_dict,device='monoarticular',loadcondition='noload',joint='knee',muscles_actuator='muscles')
ax.set_title('monoarticular knee joint\nstiffness, noload')
ax.set_xticks([1,2,3,4,5])
ax.set_xlim([0,5])

# loaded biarticular hip
ax = plt.subplot(2,4,5)
plot_dict = {'unassist_data':unassist_stiffness_dataset,'ideal_data':ideal_stiffness_dataset,
             'paretofront_dict':bi_loaded_muscles_paretofront_stiffness,'indices':bi_loaded_indices}
utils.plot_paretofront_stiffness(plot_dict,device='biarticular',loadcondition='loaded',joint='hip',muscles_actuator='muscles')
ax.set_title('biarticular hip joint\nstiffness, loaded')
ax.set_xticks([1,2,3,4,5])
ax.set_xlim([0,5])
ax.set_xlabel('stiffness (N-m/rad-kg)')
ax.set_ylabel('optimal devices')

# loaded biarticular knee
ax = plt.subplot(2,4,6)
plot_dict = {'unassist_data':unassist_stiffness_dataset,'ideal_data':ideal_stiffness_dataset,
             'paretofront_dict':bi_loaded_muscles_paretofront_stiffness,'indices':bi_loaded_indices}
utils.plot_paretofront_stiffness(plot_dict,device='biarticular',loadcondition='loaded',joint='knee',muscles_actuator='muscles')
ax.set_title('biarticular knee joint\nstiffness, loaded')
ax.set_xticks([2,4,6,8,10])
ax.set_xlim([0,10])
ax.set_xlabel('stiffness (N-m/rad-kg)')

# loaded monoarticular hip
ax = plt.subplot(2,4,7)
plot_dict = {'unassist_data':unassist_stiffness_dataset,'ideal_data':ideal_stiffness_dataset,
             'paretofront_dict':mono_loaded_muscles_paretofront_stiffness,'indices':mono_loaded_indices}
utils.plot_paretofront_stiffness(plot_dict,device='monoarticular',loadcondition='loaded',joint='hip',muscles_actuator='muscles')
ax.set_title('monoarticular hip joint\nstiffness, loaded')
ax.set_xticks([1,2,3,4,5])
ax.set_xlim([0,5])
ax.set_xlabel('stiffness (N-m/rad-kg)')

# loaded monoarticular knee
ax = plt.subplot(2,4,8)
plot_dict = {'unassist_data':unassist_stiffness_dataset,'ideal_data':ideal_stiffness_dataset,
             'paretofront_dict':mono_loaded_muscles_paretofront_stiffness,'indices':mono_loaded_indices}
utils.plot_paretofront_stiffness(plot_dict,device='monoarticular',loadcondition='loaded',joint='knee',muscles_actuator='muscles',legends=True)
ax.set_title('monoarticular knee joint\nstiffness, loaded')
ax.set_xticks([2,4,6,8,10])
ax.set_xlim([0,10])
ax.set_xlabel('stiffness (N-m/rad-kg)')
# final setting
fig.tight_layout(h_pad=-1, w_pad=-1.5)
fig.subplots_adjust(top=0.95, bottom=0.075, left=0.100, right=0.975,hspace=0.40,wspace=0.25)
plt.show()
fig.savefig('./Figures/Paretofront/Analyses_Pareto/Paper_Figure_Muscles_Stiffness_Plot.pdf',orientation='landscape',bbox_inches='tight')
#********************************************************************************************
#********************************************************************************************
# figure
fig = plt.figure(num='paretofront joint stiffness',figsize=[6.4*2, 4.8*2.5])
# noload biarticular hip
ax = plt.subplot(2,4,1)
plot_dict = {'unassist_data':unassist_stiffness_dataset,'ideal_data':ideal_stiffness_dataset,
             'paretofront_dict':bi_noload_actuators_paretofront_stiffness,'indices':bi_noload_indices}
utils.plot_paretofront_stiffness(plot_dict,device='biarticular',loadcondition='noload',joint='hip',muscles_actuator='actuator')
ax.set_title('biarticular hip joint\nstiffness, noload')
ax.set_xticks([1,2,3,4])
ax.set_xlim([0,4])
ax.set_ylabel('optimal devices')

# noload biarticular knee
ax = plt.subplot(2,4,2)
plot_dict = {'unassist_data':unassist_stiffness_dataset,'ideal_data':ideal_stiffness_dataset,
             'paretofront_dict':bi_noload_actuators_paretofront_stiffness,'indices':bi_noload_indices}
utils.plot_paretofront_stiffness(plot_dict,device='biarticular',loadcondition='noload',joint='knee',muscles_actuator='actuator')
ax.set_title('biarticular knee joint\nstiffness, noload')
ax.set_xticks([1,2,3,4,5])
ax.set_xlim([0,5])
# noload monoarticular hip
ax = plt.subplot(2,4,3)
plot_dict = {'unassist_data':unassist_stiffness_dataset,'ideal_data':ideal_stiffness_dataset,
             'paretofront_dict':mono_noload_actuators_paretofront_stiffness,'indices':mono_noload_indices}
utils.plot_paretofront_stiffness(plot_dict,device='monoarticular',loadcondition='noload',joint='hip',muscles_actuator='actuator')
ax.set_title('monoarticular hip joint\nstiffness, noload')
ax.set_xticks([1,2,3,4])
ax.set_xlim([0,4])

# noload monoarticular knee
ax = plt.subplot(2,4,4)
plot_dict = {'unassist_data':unassist_stiffness_dataset,'ideal_data':ideal_stiffness_dataset,
             'paretofront_dict':mono_noload_actuators_paretofront_stiffness,'indices':mono_noload_indices}
utils.plot_paretofront_stiffness(plot_dict,device='monoarticular',loadcondition='noload',joint='knee',muscles_actuator='actuator')
ax.set_title('monoarticular knee joint\nstiffness, noload')
ax.set_xticks([1,2,3,4,5])
ax.set_xlim([0,5])

# loaded biarticular hip
ax = plt.subplot(2,4,5)
plot_dict = {'unassist_data':unassist_stiffness_dataset,'ideal_data':ideal_stiffness_dataset,
             'paretofront_dict':bi_loaded_actuators_paretofront_stiffness,'indices':bi_loaded_indices}
utils.plot_paretofront_stiffness(plot_dict,device='biarticular',loadcondition='loaded',joint='hip',muscles_actuator='actuator')
ax.set_title('biarticular hip joint\nstiffness, loaded')
ax.set_xticks([1,2,3,4,5])
ax.set_xlim([0,5])
ax.set_xlabel('stiffness (N-m/rad-kg)')
ax.set_ylabel('optimal devices')

# loaded biarticular knee
ax = plt.subplot(2,4,6)
plot_dict = {'unassist_data':unassist_stiffness_dataset,'ideal_data':ideal_stiffness_dataset,
             'paretofront_dict':bi_loaded_actuators_paretofront_stiffness,'indices':bi_loaded_indices}
utils.plot_paretofront_stiffness(plot_dict,device='biarticular',loadcondition='loaded',joint='knee',muscles_actuator='actuator')
ax.set_title('biarticular knee joint\nstiffness, loaded')
ax.set_xticks([2,4,6,8,10])
ax.set_xlim([0,10])
ax.set_xlabel('stiffness (N-m/rad-kg)')

# loaded monoarticular hip
ax = plt.subplot(2,4,7)
plot_dict = {'unassist_data':unassist_stiffness_dataset,'ideal_data':ideal_stiffness_dataset,
             'paretofront_dict':mono_loaded_actuators_paretofront_stiffness,'indices':mono_loaded_indices}
utils.plot_paretofront_stiffness(plot_dict,device='monoarticular',loadcondition='loaded',joint='hip',muscles_actuator='actuator')
ax.set_title('monoarticular hip joint\nstiffness, loaded')
ax.set_xticks([1,2,3,4,5])
ax.set_xlim([0,5])
ax.set_xlabel('stiffness (N-m/rad-kg)')

# loaded monoarticular knee
ax = plt.subplot(2,4,8)
plot_dict = {'unassist_data':unassist_stiffness_dataset,'ideal_data':ideal_stiffness_dataset,
             'paretofront_dict':mono_loaded_actuators_paretofront_stiffness,'indices':mono_loaded_indices}
utils.plot_paretofront_stiffness(plot_dict,device='monoarticular',loadcondition='loaded',joint='knee',muscles_actuator='actuator',legends=True)
ax.set_title('monoarticular knee joint\nstiffness, loaded')
ax.set_xticks([2,4,6,8,10])
ax.set_xlim([0,10])
ax.set_xlabel('stiffness (N-m/rad-kg)')
# final setting
fig.tight_layout(h_pad=-1, w_pad=-1.5)
fig.subplots_adjust(top=0.95, bottom=0.075, left=0.100, right=0.975,hspace=0.40,wspace=0.25)
plt.show()
fig.savefig('./Figures/Paretofront/Analyses_Pareto/Paper_Figure_Actuators_Stiffness_Plot.pdf',orientation='landscape',bbox_inches='tight')

#*************************************************************************************************************
                                    ################## R square ##################
#*************************************************************************************************************
# figure
fig = plt.figure(num='paretofront joint R square',figsize=[6.4*2, 4.8*2.5])
# noload biarticular hip
ax = plt.subplot(2,4,1)
plot_dict = {'paretofront_dict':bi_noload_muscles_paretofront_R_square,'indices':bi_noload_indices}
utils.plot_paretofront_Rsquare(plot_dict,device='biarticular',loadcondition='noload',joint='hip',muscles_actuator='muscles')
ax.set_title('biarticular hip joint\nR square, noload')
ax.set_xticks([0,0.2,0.4,0.6,0.8,1])
ax.set_xlim([0,1])
ax.set_ylabel('optimal devices')
# noload biarticular knee
ax = plt.subplot(2,4,2)
plot_dict = {'paretofront_dict':bi_noload_muscles_paretofront_R_square,'indices':bi_noload_indices}
utils.plot_paretofront_Rsquare(plot_dict,device='biarticular',loadcondition='noload',joint='knee',muscles_actuator='muscles')
ax.set_title('biarticular knee joint\nR square, noload')
ax.set_xticks([0,0.2,0.4,0.6,0.8,1])
ax.set_xlim([0,1])

# noload monoarticular hip
ax = plt.subplot(2,4,3)
plot_dict = {'paretofront_dict':mono_noload_muscles_paretofront_R_square,'indices':mono_noload_indices}
utils.plot_paretofront_Rsquare(plot_dict,device='monoarticular',loadcondition='noload',joint='hip',muscles_actuator='muscles')
ax.set_title('monoarticular hip joint\nR square, noload')
ax.set_xticks([0,0.2,0.4,0.6,0.8,1])
ax.set_xlim([0,1])

# noload monoarticular knee
ax = plt.subplot(2,4,4)
plot_dict = {'paretofront_dict':mono_noload_muscles_paretofront_R_square,'indices':mono_noload_indices}
utils.plot_paretofront_Rsquare(plot_dict,device='monoarticular',loadcondition='noload',joint='knee',muscles_actuator='muscles')
ax.set_title('monoarticular knee joint\nR square, noload')
ax.set_xticks([0,0.2,0.4,0.6,0.8,1])
ax.set_xlim([0,1])

# loaded biarticular hip
ax = plt.subplot(2,4,5)
plot_dict = {'paretofront_dict':bi_loaded_muscles_paretofront_R_square,'indices':bi_loaded_indices}
utils.plot_paretofront_Rsquare(plot_dict,device='biarticular',loadcondition='loaded',joint='hip',muscles_actuator='muscles')
ax.set_title('biarticular hip joint\nR square, loaded')
ax.set_xticks([0,0.2,0.4,0.6,0.8,1])
ax.set_xlim([0,1])
ax.set_xlabel('R square')
ax.set_ylabel('optimal devices')

# loaded biarticular knee
ax = plt.subplot(2,4,6)
plot_dict = {'paretofront_dict':bi_loaded_muscles_paretofront_R_square,'indices':bi_loaded_indices}
utils.plot_paretofront_Rsquare(plot_dict,device='biarticular',loadcondition='loaded',joint='knee',muscles_actuator='muscles')
ax.set_title('biarticular knee joint\nR square, loaded')
ax.set_xticks([0,0.2,0.4,0.6,0.8,1])
ax.set_xlim([0,1])
ax.set_xlabel('R square')

# loaded monoarticular hip
ax = plt.subplot(2,4,7)
plot_dict = {'paretofront_dict':mono_loaded_muscles_paretofront_R_square,'indices':mono_loaded_indices}
utils.plot_paretofront_Rsquare(plot_dict,device='monoarticular',loadcondition='loaded',joint='hip',muscles_actuator='muscles')
ax.set_title('monoarticular hip joint\nR square, loaded')
ax.set_xticks([0,0.2,0.4,0.6,0.8,1])
ax.set_xlim([0,1])
ax.set_xlabel('R square')

# loaded monoarticular knee
ax = plt.subplot(2,4,8)
plot_dict = {'paretofront_dict':mono_loaded_muscles_paretofront_R_square,'indices':mono_loaded_indices}
utils.plot_paretofront_Rsquare(plot_dict,device='monoarticular',loadcondition='loaded',joint='knee',muscles_actuator='muscles',legends=True)
ax.set_title('monoarticular knee joint\nR square, loaded')
ax.set_xticks([0,0.2,0.4,0.6,0.8,1])
ax.set_xlim([0,1])
ax.set_xlabel('R square')
# final setting
fig.tight_layout(h_pad=-1, w_pad=-1.5)
fig.subplots_adjust(top=0.95, bottom=0.075, left=0.100, right=0.975,hspace=0.40,wspace=0.25)
plt.show()
fig.savefig('./Figures/Paretofront/Analyses_Pareto/Paper_Figure_Muscles_Rsquare_Plot.pdf',orientation='landscape',bbox_inches='tight')
#********************************************************************************************
#********************************************************************************************
# figure
fig = plt.figure(num='paretofront joint stiffness',figsize=[6.4*2, 4.8*2.5])
# noload biarticular hip
ax = plt.subplot(2,4,1)
plot_dict = {'paretofront_dict':bi_noload_actuators_paretofront_R_square,'indices':bi_noload_indices}
utils.plot_paretofront_Rsquare(plot_dict,device='biarticular',loadcondition='noload',joint='hip',muscles_actuator='actuator')
ax.set_title('biarticular hip joint\nR square, noload')
ax.set_xticks([0,0.2,0.4,0.6,0.8,1])
ax.set_xlim([0,1])
ax.set_ylabel('optimal devices')

# noload biarticular knee
ax = plt.subplot(2,4,2)
plot_dict = {'paretofront_dict':bi_noload_actuators_paretofront_R_square,'indices':bi_noload_indices}
utils.plot_paretofront_Rsquare(plot_dict,device='biarticular',loadcondition='noload',joint='knee',muscles_actuator='actuator')
ax.set_title('biarticular knee joint\nR square, noload')
ax.set_xticks([0,0.2,0.4,0.6,0.8,1])
ax.set_xlim([0,1])
# noload monoarticular hip
ax = plt.subplot(2,4,3)
plot_dict = {'paretofront_dict':mono_noload_actuators_paretofront_R_square,'indices':mono_noload_indices}
utils.plot_paretofront_Rsquare(plot_dict,device='monoarticular',loadcondition='noload',joint='hip',muscles_actuator='actuator')
ax.set_title('monoarticular hip joint\nR square, noload')
ax.set_xticks([0,0.2,0.4,0.6,0.8,1])
ax.set_xlim([0,1])

# noload monoarticular knee
ax = plt.subplot(2,4,4)
plot_dict = {'paretofront_dict':mono_noload_actuators_paretofront_R_square,'indices':mono_noload_indices}
utils.plot_paretofront_Rsquare(plot_dict,device='monoarticular',loadcondition='noload',joint='knee',muscles_actuator='actuator')
ax.set_title('monoarticular knee joint\nR square, noload')
ax.set_xticks([0,0.2,0.4,0.6,0.8,1])
ax.set_xlim([0,1])

# loaded biarticular hip
ax = plt.subplot(2,4,5)
plot_dict = {'paretofront_dict':bi_loaded_actuators_paretofront_R_square,'indices':bi_loaded_indices}
utils.plot_paretofront_Rsquare(plot_dict,device='biarticular',loadcondition='loaded',joint='hip',muscles_actuator='actuator')
ax.set_title('biarticular hip joint\nR square, loaded')
ax.set_xticks([0,0.2,0.4,0.6,0.8,1])
ax.set_xlim([0,1])
ax.set_xlabel('R square')
ax.set_ylabel('optimal devices')

# loaded biarticular knee
ax = plt.subplot(2,4,6)
plot_dict = {'paretofront_dict':bi_loaded_actuators_paretofront_R_square,'indices':bi_loaded_indices}
utils.plot_paretofront_Rsquare(plot_dict,device='biarticular',loadcondition='loaded',joint='knee',muscles_actuator='actuator')
ax.set_title('biarticular knee joint\nR square, loaded')
ax.set_xticks([0,0.2,0.4,0.6,0.8,1])
ax.set_xlim([0,1])
ax.set_xlabel('R square')

# loaded monoarticular hip
ax = plt.subplot(2,4,7)
plot_dict = {'paretofront_dict':mono_loaded_actuators_paretofront_R_square,'indices':mono_loaded_indices}
utils.plot_paretofront_Rsquare(plot_dict,device='monoarticular',loadcondition='loaded',joint='hip',muscles_actuator='actuator')
ax.set_title('monoarticular hip joint\nR square, loaded')
ax.set_xticks([0,0.2,0.4,0.6,0.8,1])
ax.set_xlim([0,1])
ax.set_xlabel('R square')

# loaded monoarticular knee
ax = plt.subplot(2,4,8)
plot_dict = {'paretofront_dict':mono_loaded_actuators_paretofront_R_square,'indices':mono_loaded_indices}
utils.plot_paretofront_Rsquare(plot_dict,device='monoarticular',loadcondition='loaded',joint='knee',muscles_actuator='actuator',legends=True)
ax.set_title('monoarticular knee joint\nR square, loaded')
ax.set_xticks([0,0.2,0.4,0.6,0.8,1])
ax.set_xlim([0,1])
ax.set_xlabel('R square')
# final setting
fig.tight_layout(h_pad=-1, w_pad=-1.5)
fig.subplots_adjust(top=0.95, bottom=0.075, left=0.100, right=0.975,hspace=0.40,wspace=0.25)
plt.show()
fig.savefig('./Figures/Paretofront/Analyses_Pareto/Paper_Figure_Actuators_Rsquare_Plot.pdf',orientation='landscape',bbox_inches='tight')
