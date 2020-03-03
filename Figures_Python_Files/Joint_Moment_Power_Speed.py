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
# joint moment dataset
directory = 'D:\Ali.K.M.Bonab\Walking_Mass_Inertia_Effect\Data\Data\RRA\*_torque.csv'
files = enumerate(glob.iglob(directory), 1)
jointmoment_dataset = {pathlib.PurePath(f[1]).stem: np.loadtxt(f[1], delimiter=',') for f in files}
# joint power dataset
directory = 'D:\Ali.K.M.Bonab\Walking_Mass_Inertia_Effect\Data\Data\RRA\*_power.csv'
files = enumerate(glob.iglob(directory), 1)
jointpower_dataset = {pathlib.PurePath(f[1]).stem: np.loadtxt(f[1], delimiter=',') for f in files}
# joint speed dataset
directory = 'D:\Ali.K.M.Bonab\Walking_Mass_Inertia_Effect\Data\Data\RRA\*_speed.csv'
files = enumerate(glob.iglob(directory), 1)
jointspeed_dataset = {pathlib.PurePath(f[1]).stem: np.loadtxt(f[1], delimiter=',') for f in files}
# gls
gl_noload = {'noload_subject{}_trial{}'.format(i,j): utils.construct_gl_mass_side(subjectno=i,trialno=j,loadcond='noload') for i in subjects for j in trials_num}
gl_loaded = {'loaded_subject{}_trial{}'.format(i,j): utils.construct_gl_mass_side(subjectno=i,trialno=j,loadcond='loaded') for i in subjects for j in trials_num}
#####################################################################################
# Processing Data
# toe-off
noload_mean_toe_off,_,loaded_mean_toe_off,_ = utils.toe_off_avg_std(gl_noload,gl_loaded)
# hip joint moment
normal_loaded_hipjoint_moment = utils.normalize_direction_data(jointmoment_dataset['loaded_hipjoint_torque'],gl_noload,direction=True)
normal_noload_hipjoint_moment = utils.normalize_direction_data(jointmoment_dataset['noload_hipjoint_torque'],gl_noload,direction=True)
mean_norm_loaded_hipjoint_moment,std_norm_loaded_hipjoint_moment = utils.mean_std_over_subjects(normal_loaded_hipjoint_moment)
mean_norm_noload_hipjoint_moment,std_norm_noload_hipjoint_moment = utils.mean_std_over_subjects(normal_noload_hipjoint_moment)
# knee joint moment
normal_loaded_kneejoint_moment = utils.normalize_direction_data(jointmoment_dataset['loaded_kneejoint_torque'],gl_noload,direction=True)
normal_noload_kneejoint_moment = utils.normalize_direction_data(jointmoment_dataset['noload_kneejoint_torque'],gl_noload,direction=True)
mean_norm_loaded_kneejoint_moment,std_norm_loaded_kneejoint_moment = utils.mean_std_over_subjects(normal_loaded_kneejoint_moment)
mean_norm_noload_kneejoint_moment,std_norm_noload_kneejoint_moment = utils.mean_std_over_subjects(normal_noload_kneejoint_moment)
# hip joint power
normal_loaded_hipjoint_power = utils.normalize_direction_data(jointpower_dataset['loaded_hipjoint_power'],gl_noload,direction=False)
normal_noload_hipjoint_power = utils.normalize_direction_data(jointpower_dataset['noload_hipjoint_power'],gl_noload,direction=False)
mean_norm_loaded_hipjoint_power,std_norm_loaded_hipjoint_power = utils.mean_std_over_subjects(normal_loaded_hipjoint_power)
mean_norm_noload_hipjoint_power,std_norm_noload_hipjoint_power = utils.mean_std_over_subjects(normal_noload_hipjoint_power)
# knee joint power
normal_loaded_kneejoint_power = utils.normalize_direction_data(jointpower_dataset['loaded_kneejoint_power'],gl_noload,direction=False)
normal_noload_kneejoint_power = utils.normalize_direction_data(jointpower_dataset['noload_kneejoint_power'],gl_noload,direction=False)
mean_norm_loaded_kneejoint_power,std_norm_loaded_kneejoint_power = utils.mean_std_over_subjects(normal_loaded_kneejoint_power)
mean_norm_noload_kneejoint_power,std_norm_noload_kneejoint_power = utils.mean_std_over_subjects(normal_noload_kneejoint_power)
# hip joint speed
loaded_hipjoint_speed = utils.normalize_direction_data(jointspeed_dataset['loaded_hipjoint_speed'],gl_noload,normalize=False,direction=True)
noload_hipjoint_speed = utils.normalize_direction_data(jointspeed_dataset['noload_hipjoint_speed'],gl_noload,normalize=False,direction=True)
mean_loaded_hipjoint_speed,std_loaded_hipjoint_speed = utils.mean_std_over_subjects(loaded_hipjoint_speed)
mean_noload_hipjoint_speed,std_noload_hipjoint_speed = utils.mean_std_over_subjects(noload_hipjoint_speed)
# knee joint speed
loaded_kneejoint_speed = utils.normalize_direction_data(jointspeed_dataset['loaded_kneejoint_speed'],gl_noload,normalize=False,direction=True)
noload_kneejoint_speed = utils.normalize_direction_data(jointspeed_dataset['noload_kneejoint_speed'],gl_noload,normalize=False,direction=True)
mean_loaded_kneejoint_speed,std_loaded_kneejoint_speed = utils.mean_std_over_subjects(loaded_kneejoint_speed)
mean_noload_kneejoint_speed,std_noload_kneejoint_speed = utils.mean_std_over_subjects(noload_kneejoint_speed)
    
#####################################################################################
# Plots
# hip joint moment plot dictionaries
hip_moment_loaded_plot_dic = {'pgc':gait_cycle,'avg':utils.smooth(mean_norm_loaded_hipjoint_moment,3),'label':'Loaded',
                        'std':utils.smooth(std_norm_loaded_hipjoint_moment,3),'avg_toeoff':loaded_mean_toe_off}
hip_moment_noload_plot_dic = {'pgc':gait_cycle,'avg':utils.smooth(mean_norm_noload_hipjoint_moment,3),'label':'Noload',
                        'std':utils.smooth(std_norm_noload_hipjoint_moment,3),'avg_toeoff':noload_mean_toe_off}
# knee joint moment plot dictionaries
knee_moment_loaded_plot_dic = {'pgc':gait_cycle,'avg':utils.smooth(mean_norm_loaded_kneejoint_moment,3),'label':'Loaded',
                        'std':utils.smooth(std_norm_loaded_kneejoint_moment,3),'avg_toeoff':loaded_mean_toe_off}
knee_moment_noload_plot_dic = {'pgc':gait_cycle,'avg':utils.smooth(mean_norm_noload_kneejoint_moment,3),'label':'Noload',
                        'std':utils.smooth(std_norm_noload_kneejoint_moment,3),'avg_toeoff':noload_mean_toe_off}
# hip joint power plot dictionaries
hip_power_loaded_plot_dic = {'pgc':gait_cycle,'avg':utils.smooth(mean_norm_loaded_hipjoint_power,3),'label':'Loaded',
                        'std':utils.smooth(std_norm_loaded_hipjoint_power,3),'avg_toeoff':loaded_mean_toe_off}
hip_power_noload_plot_dic = {'pgc':gait_cycle,'avg':utils.smooth(mean_norm_noload_hipjoint_power,3),'label':'Noload',
                        'std':utils.smooth(std_norm_noload_hipjoint_power,3),'avg_toeoff':noload_mean_toe_off}
# knee joint power plot dictionaries
knee_power_loaded_plot_dic = {'pgc':gait_cycle,'avg':utils.smooth(mean_norm_loaded_kneejoint_power,3),'label':'Loaded',
                        'std':utils.smooth(std_norm_loaded_kneejoint_power,3),'avg_toeoff':loaded_mean_toe_off}
knee_power_noload_plot_dic = {'pgc':gait_cycle,'avg':utils.smooth(mean_norm_noload_kneejoint_power,3),'label':'Noload',
                        'std':utils.smooth(std_norm_noload_kneejoint_power,3),'avg_toeoff':noload_mean_toe_off}
# hip joint speed plot dictionaries
hip_speed_loaded_plot_dic = {'pgc':gait_cycle,'avg':utils.smooth(mean_loaded_hipjoint_speed,3),'label':'Loaded',
                        'std':utils.smooth(std_loaded_hipjoint_speed,3),'avg_toeoff':loaded_mean_toe_off}
hip_speed_noload_plot_dic = {'pgc':gait_cycle,'avg':utils.smooth(mean_noload_hipjoint_speed,3),'label':'Noload',
                        'std':utils.smooth(std_noload_hipjoint_speed,3),'avg_toeoff':noload_mean_toe_off}
# knee joint speed plot dictionaries
knee_speed_loaded_plot_dic = {'pgc':gait_cycle,'avg':utils.smooth(mean_loaded_kneejoint_speed,3),'label':'Loaded',
                        'std':utils.smooth(std_loaded_kneejoint_speed,3),'avg_toeoff':loaded_mean_toe_off}
knee_speed_noload_plot_dic = {'pgc':gait_cycle,'avg':utils.smooth(mean_noload_kneejoint_speed,3),'label':'Noload',
                        'std':utils.smooth(std_noload_kneejoint_speed,3),'avg_toeoff':noload_mean_toe_off}

#*****************************

# hip joint moment figure
fig, ax = plt.subplots(num='Hip Joint Moment',figsize=(6.4, 4.8))
utils.plot_shaded_avg(plot_dic=hip_moment_loaded_plot_dic,color='k')
utils.plot_shaded_avg(plot_dic=hip_moment_noload_plot_dic,toeoff_color='xkcd:shamrock green',color='xkcd:irish green')
plt.legend(loc='best',frameon=False)
plt.ylim((-1,2))
plt.xlabel('gait cycle (%)')
plt.ylabel('hip joint moment (N-m/kg)')
utils.no_top_right(ax)
plt.show()
fig.tight_layout(h_pad=-4.0, w_pad=-4.0)
fig.savefig('./Figures/RRA_HipJointMoment.pdf',orientation='landscape',bbox_inches='tight')

# knee joint moment figure
fig, ax = plt.subplots(num='Knee Joint Moment',figsize=(6.4, 4.8))
utils.plot_shaded_avg(plot_dic=knee_moment_loaded_plot_dic,color='k')
utils.plot_shaded_avg(plot_dic=knee_moment_noload_plot_dic,toeoff_color='xkcd:shamrock green',color='xkcd:irish green')
plt.legend(loc='best',frameon=False)
plt.ylim((-1,2))
plt.xlabel('gait cycle (%)')
plt.ylabel('knee joint moment (N-m/kg)')
utils.no_top_right(ax)
plt.show()
fig.tight_layout(h_pad=-4.0, w_pad=-4.0)
fig.savefig('./Figures/RRA_KneeJointMoment.pdf',orientation='landscape',bbox_inches='tight')

#*****************************

# hip joint power figure
fig, ax = plt.subplots(num='Hip Joint Power',figsize=(6.4, 4.8))
utils.plot_shaded_avg(plot_dic=hip_power_loaded_plot_dic,color='k')
utils.plot_shaded_avg(plot_dic=hip_power_noload_plot_dic,toeoff_color='xkcd:shamrock green',color='xkcd:irish green')
plt.legend(loc='best',frameon=False)
plt.ylim((-3,3))
plt.xlabel('gait cycle (%)')
plt.ylabel('hip joint power (W/kg)')
utils.no_top_right(ax)
plt.show()
fig.tight_layout(h_pad=-4.0, w_pad=-4.0)
fig.savefig('./Figures/RRA_HipJointPower.pdf',orientation='landscape',bbox_inches='tight')

# knee joint power figure
fig, ax = plt.subplots(num='Knee Joint Power',figsize=(6.4, 4.8))
utils.plot_shaded_avg(plot_dic=knee_power_loaded_plot_dic,color='k')
utils.plot_shaded_avg(plot_dic=knee_power_noload_plot_dic,toeoff_color='xkcd:shamrock green',color='xkcd:irish green')
plt.legend(loc='best',frameon=False)
plt.ylim((-3,3))
plt.xlabel('gait cycle (%)')
plt.ylabel('knee joint power (W/kg)')
utils.no_top_right(ax)
plt.show()
fig.tight_layout(h_pad=-4.0, w_pad=-4.0)
fig.savefig('./Figures/RRA_KneeJointPower.pdf',orientation='landscape',bbox_inches='tight')

#*****************************

# hip joint power figure
fig, ax = plt.subplots(num='Hip Joint Speed',figsize=(6.4, 4.8))
utils.plot_shaded_avg(plot_dic=hip_speed_loaded_plot_dic,color='k')
utils.plot_shaded_avg(plot_dic=hip_speed_noload_plot_dic,toeoff_color='xkcd:shamrock green',color='xkcd:irish green')
plt.legend(loc='best',frameon=False)
plt.yticks([-6,-4,-2,0,2,4,6])
plt.xlabel('gait cycle (%)')
plt.ylabel('hip joint speed (rad/s)')
utils.no_top_right(ax)
plt.show()
fig.tight_layout(h_pad=-4.0, w_pad=-4.0)
fig.savefig('./Figures/RRA_HipJointSpeed.pdf',orientation='landscape',bbox_inches='tight')

# knee joint moment figure
fig, ax = plt.subplots(num='Knee Joint Speed',figsize=(6.4, 4.8))
utils.plot_shaded_avg(plot_dic=knee_speed_loaded_plot_dic,color='k')
utils.plot_shaded_avg(plot_dic=knee_speed_noload_plot_dic,toeoff_color='xkcd:shamrock green',color='xkcd:irish green')
plt.legend(loc='best',frameon=False)
plt.yticks([-6,-4,-2,0,2,4,6])
plt.xlabel('gait cycle (%)')
plt.ylabel('knee joint speed (rad/s)')
utils.no_top_right(ax)
plt.show()
fig.tight_layout(h_pad=-4.0, w_pad=-4.0)
fig.savefig('./Figures/RRA_KneeJointSpeed.pdf',orientation='landscape',bbox_inches='tight')