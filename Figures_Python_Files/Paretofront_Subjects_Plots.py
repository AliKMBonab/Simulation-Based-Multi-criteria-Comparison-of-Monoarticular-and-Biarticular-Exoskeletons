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
ideal_profiles_dataset = utils.csv2numpy('./Data/Ideal/ideal_exos_profiles_dataset.csv') 
# pareto exo torque dataset
directory = './Data/Pareto/*_torque.csv'
files = enumerate(glob.iglob(directory), 1)
exo_torque_dataset = {pathlib.PurePath(f[1]).stem: np.loadtxt(f[1], delimiter=',') for f in files}
# pareto exo power dataset
directory = './Data/Pareto/*_power.csv'
files = enumerate(glob.iglob(directory), 1)
exo_power_dataset = {pathlib.PurePath(f[1]).stem: np.loadtxt(f[1], delimiter=',') for f in files}
# pareto exo energy dataset
directory = './Data/Pareto/*_energy.csv'
files = enumerate(glob.iglob(directory), 1)
assisted_energy_dataset = {pathlib.PurePath(f[1]).stem: np.loadtxt(f[1], delimiter=',') for f in files}
# unassist energy dataset
directory = './Data/Unassist/*_energy.csv'
files = enumerate(glob.iglob(directory), 1)
unassisted_energy_dataset = {pathlib.PurePath(f[1]).stem: np.loadtxt(f[1], delimiter=',') for f in files}
# gls
gl_noload = {'noload_subject{}_trial{}'.format(i,j): utils.construct_gl_mass_side(subjectno=i,trialno=j,loadcond='noload') for i in subjects for j in trials_num}
gl_loaded = {'loaded_subject{}_trial{}'.format(i,j): utils.construct_gl_mass_side(subjectno=i,trialno=j,loadcond='loaded') for i in subjects for j in trials_num}
#####################################################################################
# Processing Data
# toe-off
noload_mean_toe_off,_,loaded_mean_toe_off,_= utils.toe_off_avg_std(gl_noload,gl_loaded)
# metabolics cost reduction percents
bi_loaded_metabolics_percent = utils.pareto_metabolics_reduction(assisted_energy_dataset['biarticular_pareto_load_metabolics_energy'],unassisted_energy_dataset['loaded_metabolics_energy'])
bi_noload_metabolics_percent = utils.pareto_metabolics_reduction(assisted_energy_dataset['biarticular_pareto_noload_metabolics_energy'],unassisted_energy_dataset['noload_metabolics_energy'])
mono_loaded_metabolics_percent = utils.pareto_metabolics_reduction(assisted_energy_dataset['monoarticular_pareto_load_metabolics_energy'],unassisted_energy_dataset['loaded_metabolics_energy'])
mono_noload_metabolics_percent = utils.pareto_metabolics_reduction(assisted_energy_dataset['monoarticular_pareto_noload_metabolics_energy'],unassisted_energy_dataset['noload_metabolics_energy'])

# metabolics cost reduction percents
bi_loaded_metabolics_energy = np.reshape(assisted_energy_dataset['biarticular_pareto_load_metabolics_energy'],(25,21),order='F')
bi_noload_metabolics_energy = np.reshape(assisted_energy_dataset['biarticular_pareto_noload_metabolics_energy'],(25,21),order='F')
mono_loaded_metabolics_energy = np.reshape(assisted_energy_dataset['monoarticular_pareto_load_metabolics_energy'],(25,21),order='F')
mono_noload_metabolics_energy = np.reshape(assisted_energy_dataset['monoarticular_pareto_noload_metabolics_energy'],(25,21),order='F')
unassist_noload_metabolics_energy = unassisted_energy_dataset['noload_metabolics_energy']
unassist_loaded_metabolics_energy = unassisted_energy_dataset['loaded_metabolics_energy']

# actuators energy
bi_loaded_hip_energy= np.reshape(assisted_energy_dataset['biarticular_pareto_load_hipactuator_energy'],(25,21),order='F')
bi_loaded_knee_energy= np.reshape(assisted_energy_dataset['biarticular_pareto_load_kneeactuator_energy'],(25,21),order='F')
bi_loaded_energy = bi_loaded_hip_energy + bi_loaded_knee_energy
bi_noload_hip_energy= np.reshape(assisted_energy_dataset['biarticular_pareto_noload_hipactuator_energy'],(25,21),order='F')
bi_noload_knee_energy= np.reshape(assisted_energy_dataset['biarticular_pareto_noload_kneeactuator_energy'],(25,21),order='F')
bi_noload_energy = bi_noload_hip_energy + bi_noload_knee_energy
mono_loaded_hip_energy= np.reshape(assisted_energy_dataset['monoarticular_pareto_load_hipactuator_energy'],(25,21),order='F')
mono_loaded_knee_energy= np.reshape(assisted_energy_dataset['monoarticular_pareto_load_kneeactuator_energy'],(25,21),order='F')
mono_loaded_energy = mono_loaded_hip_energy + mono_loaded_knee_energy
mono_noload_hip_energy= np.reshape(assisted_energy_dataset['monoarticular_pareto_noload_hipactuator_energy'],(25,21),order='F')
mono_noload_knee_energy= np.reshape(assisted_energy_dataset['monoarticular_pareto_noload_kneeactuator_energy'],(25,21),order='F')
mono_noload_energy = mono_noload_hip_energy + mono_noload_knee_energy

# actuators power profiles
mean_bi_loaded_hip_power, std_bi_loaded_hip_power = utils.pareto_profiles_avg_std(exo_power_dataset['biarticular_pareto_load_hipactuator_power'],gl_noload,change_direction=False)
mean_bi_loaded_knee_power, std_bi_loaded_knee_power = utils.pareto_profiles_avg_std(exo_power_dataset['biarticular_pareto_load_kneeactuator_power'],gl_noload,change_direction=False)
mean_bi_noload_hip_power, std_bi_noload_hip_power = utils.pareto_profiles_avg_std(exo_power_dataset['biarticular_pareto_noload_hipactuator_power'],gl_noload,change_direction=False)
mean_bi_noload_knee_power, std_bi_noload_knee_power = utils.pareto_profiles_avg_std(exo_power_dataset['biarticular_pareto_noload_kneeactuator_power'],gl_noload,change_direction=False)
mean_mono_loaded_hip_power, std_mono_loaded_hip_power = utils.pareto_profiles_avg_std(exo_power_dataset['monoarticular_pareto_load_hipactuator_power'],gl_noload,change_direction=False)
mean_mono_loaded_knee_power, std_mono_loaded_knee_power = utils.pareto_profiles_avg_std(exo_power_dataset['monoarticular_pareto_load_kneeactuator_power'],gl_noload,change_direction=False)
mean_mono_noload_hip_power, std_mono_noload_hip_power = utils.pareto_profiles_avg_std(exo_power_dataset['monoarticular_pareto_noload_hipactuator_power'],gl_noload,change_direction=False)
mean_mono_noload_knee_power, std_mono_noload_knee_power = utils.pareto_profiles_avg_std(exo_power_dataset['monoarticular_pareto_noload_kneeactuator_power'],gl_noload,change_direction=False)

# actuators torque profiles
mean_bi_loaded_hip_torque, std_bi_loaded_hip_torque = utils.pareto_profiles_avg_std(exo_torque_dataset['biarticular_pareto_load_hipactuator_torque'],gl_noload,change_direction=False)
mean_bi_loaded_knee_torque, std_bi_loaded_knee_torque = utils.pareto_profiles_avg_std(exo_torque_dataset['biarticular_pareto_load_kneeactuator_torque'],gl_noload,change_direction=True)
mean_bi_noload_hip_torque, std_bi_noload_hip_torque = utils.pareto_profiles_avg_std(exo_torque_dataset['biarticular_pareto_noload_hipactuator_torque'],gl_noload,change_direction=False)
mean_bi_noload_knee_torque, std_bi_noload_knee_torque = utils.pareto_profiles_avg_std(exo_torque_dataset['biarticular_pareto_noload_kneeactuator_torque'],gl_noload,change_direction=True)
mean_mono_loaded_hip_torque, std_mono_loaded_hip_torque = utils.pareto_profiles_avg_std(exo_torque_dataset['monoarticular_pareto_load_hipactuator_torque'],gl_noload,change_direction=False)
mean_mono_loaded_knee_torque, std_mono_loaded_knee_torque = utils.pareto_profiles_avg_std(exo_torque_dataset['monoarticular_pareto_load_kneeactuator_torque'],gl_noload,change_direction=True)
mean_mono_noload_hip_torque, std_mono_noload_hip_torque = utils.pareto_profiles_avg_std(exo_torque_dataset['monoarticular_pareto_noload_hipactuator_torque'],gl_noload,change_direction=False)
mean_mono_noload_knee_torque, std_mono_noload_knee_torque = utils.pareto_profiles_avg_std(exo_torque_dataset['monoarticular_pareto_noload_kneeactuator_torque'],gl_noload,change_direction=True)

#####################################################################################
# Paretofront
# indices
bi_loaded_indices = np.array([25,24,23,22,21,17,16,13,12,11,6,1])
mono_loaded_indices = np.array([25,20,19,15,10,5,9,4,3,2,1])
bi_noload_indices = np.array([25,24,23,22,21,19,18,17,13,12,11,1])
mono_noload_indices = np.array([25,24,20,15,12,11,7,2,6,1])

bi_loaded_metabolics_percent_Paretofront,\
bi_loaded_energy_Paretofront = utils.paretofront_subjects(bi_loaded_metabolics_energy,bi_loaded_energy,unassist_loaded_metabolics_energy)
bi_noload_metabolics_percent_Paretofront,\
bi_noload_energy_Paretofront = utils.paretofront_subjects(bi_noload_metabolics_energy,bi_noload_energy,unassist_noload_metabolics_energy)
mono_loaded_metabolics_percent_Paretofront,\
mono_loaded_energy_Paretofront = utils.paretofront_subjects(mono_loaded_metabolics_energy,mono_loaded_energy,unassist_loaded_metabolics_energy)
mono_noload_metabolics_percent_Paretofront,\
mono_noload_energy_Paretofront = utils.paretofront_subjects(mono_noload_metabolics_energy,mono_noload_energy,unassist_noload_metabolics_energy)

# loaded biarticular
mean_bi_loaded_hip_torque_paretofront = utils.manual_paretofront_profiles(mean_bi_loaded_hip_torque,bi_loaded_indices)
mean_bi_loaded_knee_torque_paretofront = utils.manual_paretofront_profiles(mean_bi_loaded_knee_torque,bi_loaded_indices)
mean_bi_loaded_hip_power_paretofront = utils.manual_paretofront_profiles(mean_bi_loaded_hip_power,bi_loaded_indices)
mean_bi_loaded_knee_power_paretofront = utils.manual_paretofront_profiles(mean_bi_loaded_knee_power,bi_loaded_indices)

# loaded monoarticular
mean_mono_loaded_hip_torque_paretofront = utils.manual_paretofront_profiles(mean_mono_loaded_hip_torque,mono_loaded_indices)
mean_mono_loaded_knee_torque_paretofront = utils.manual_paretofront_profiles(mean_mono_loaded_knee_torque,mono_loaded_indices)
mean_mono_loaded_hip_power_paretofront = utils.manual_paretofront_profiles(mean_mono_loaded_hip_power,mono_loaded_indices)
mean_mono_loaded_knee_power_paretofront = utils.manual_paretofront_profiles(mean_mono_loaded_knee_power,mono_loaded_indices)

# noload biarticular
mean_bi_noload_hip_torque_paretofront = utils.manual_paretofront_profiles(mean_bi_noload_hip_torque,bi_noload_indices)
mean_bi_noload_knee_torque_paretofront = utils.manual_paretofront_profiles(mean_bi_noload_knee_torque,bi_noload_indices)
mean_bi_noload_hip_power_paretofront = utils.manual_paretofront_profiles(mean_bi_noload_hip_power,bi_noload_indices)
mean_bi_noload_knee_power_paretofront = utils.manual_paretofront_profiles(mean_bi_noload_knee_power,bi_noload_indices)

# noload monoarticular
mean_mono_noload_hip_torque_paretofront = utils.manual_paretofront_profiles(mean_mono_noload_hip_torque,mono_noload_indices)
mean_mono_noload_knee_torque_paretofront = utils.manual_paretofront_profiles(mean_mono_noload_knee_torque,mono_noload_indices)
mean_mono_noload_hip_power_paretofront = utils.manual_paretofront_profiles(mean_mono_noload_hip_power,mono_noload_indices)
mean_mono_noload_knee_power_paretofront = utils.manual_paretofront_profiles(mean_mono_noload_knee_power,mono_noload_indices)

#####################################################################################
# profile plots
#************************************************************************************
# torque profile
fig, axes = plt.subplots(nrows=2,ncols=4,num='Pareto Curve: loaded mono vs bi',figsize=(14.8, 9.6))
# biarticular loaded hip
plot_dic = {'data':utils.smooth(mean_bi_loaded_hip_torque_paretofront,5,multidim=True),
            'joint_data':utils.smooth(rra_dataset['mean_norm_loaded_hipjoint_moment'],5),'joint_color':'k',
            'avg_toeoff':loaded_mean_toe_off,'indices':bi_loaded_indices,'title':'biarticular hip\n torque, loaded',
            'ideal_data':ideal_profiles_dataset['mean_bi_loaded_hip_torque'],'ideal_color':mycolors['crimson red']}
ax = plt.subplot(2,4,1)
utils.plot_paretofront_profile_changes(plot_dic,colormap='tab20',include_colorbar=False,
                                       toeoff_color='k',add_ideal_profile=True,ylabel='flexion/extension\n moment (N.m/kg)')
ax.set_yticks([-1.5,-0.5,0,0.5,1.5])
ax.set_ylim((-1.5,1.5))
# biarticular loaded knee
plot_dic = {'data':utils.smooth(mean_bi_loaded_knee_torque_paretofront,5,multidim=True),
            'joint_data':utils.smooth(rra_dataset['mean_norm_loaded_kneejoint_moment'],5),'joint_color':'k',
            'avg_toeoff':loaded_mean_toe_off,'indices':bi_loaded_indices,'title':'biarticular knee\n torque, loaded',
            'ideal_data':ideal_profiles_dataset['mean_bi_loaded_knee_torque'],'ideal_color':mycolors['crimson red']}
ax = plt.subplot(2,4,2)
utils.plot_paretofront_profile_changes(plot_dic,add_ideal_profile=True,colormap='tab20',toeoff_color='k')
ax.set_yticks([-1.5,-0.5,0,0.5,1.5])
ax.set_ylim((-1.5,1.5))
#****************************************
# monoarticular loaded hip
plot_dic = {'data':utils.smooth(mean_mono_loaded_hip_torque_paretofront,5,multidim=True),
            'joint_data':utils.smooth(rra_dataset['mean_norm_loaded_hipjoint_moment'],5),'joint_color':'k',
            'avg_toeoff':loaded_mean_toe_off,'indices':mono_loaded_indices,'title':'monoarticular hip\n torque, loaded',
            'ideal_data':ideal_profiles_dataset['mean_mono_loaded_hip_torque'],'ideal_color':mycolors['crimson red']}
ax = plt.subplot(2,4,3)
utils.plot_paretofront_profile_changes(plot_dic,add_ideal_profile=True,colormap='tab20',include_colorbar=False,toeoff_color='k')
ax.set_yticks([-1.5,-0.5,0,0.5,1.5])
ax.set_ylim((-1.5,1.5))
# monoarticular loaded knee
plot_dic = {'data':utils.smooth(mean_mono_loaded_knee_torque_paretofront,5,multidim=True),
            'joint_data':utils.smooth(rra_dataset['mean_norm_loaded_kneejoint_moment'],5),'joint_color':'k',
            'avg_toeoff':loaded_mean_toe_off,'indices':mono_loaded_indices,'title':'monoarticular knee\n torque, loaded',
            'ideal_data':ideal_profiles_dataset['mean_mono_loaded_knee_torque'],'ideal_color':mycolors['crimson red']}
ax = plt.subplot(2,4,4)
utils.plot_paretofront_profile_changes(plot_dic,add_ideal_profile=True,colormap='tab20',toeoff_color='k')
ax.set_yticks([-1.5,-0.5,0,0.5,1.5])
ax.set_ylim((-1.5,1.5))
#****************************************
# biarticular noload hip
plot_dic = {'data':utils.smooth(mean_bi_noload_hip_torque_paretofront,5,multidim=True),
            'joint_data':utils.smooth(rra_dataset['mean_norm_noload_hipjoint_moment'],5),'joint_color':'xkcd:shamrock green',
            'avg_toeoff':noload_mean_toe_off,'indices':bi_noload_indices,'title':'biarticular hip\n torque, noload',
            'ideal_data':ideal_profiles_dataset['mean_bi_noload_hip_torque'],'ideal_color':mycolors['french rose']}
ax = plt.subplot(2,4,5)
utils.plot_paretofront_profile_changes(plot_dic,colormap='tab20',include_colorbar=False,add_ideal_profile=True,
                        toeoff_color='xkcd:shamrock green',ylabel='flexion/extension\n moment (N.m/kg)',xlabel=True)
ax.set_yticks([-1.5,-0.5,0,0.5,1.5])
ax.set_ylim((-1.5,1.5))
# biarticular noload knee
plot_dic = {'data':utils.smooth(mean_bi_noload_knee_torque_paretofront,5,multidim=True),
            'joint_data':utils.smooth(rra_dataset['mean_norm_noload_kneejoint_moment'],5),'joint_color':'xkcd:shamrock green',
            'avg_toeoff':noload_mean_toe_off,'indices':bi_noload_indices,'title':'biarticular knee\n torque, noload',
            'ideal_data':ideal_profiles_dataset['mean_bi_noload_knee_torque'],'ideal_color':mycolors['french rose']}
ax = plt.subplot(2,4,6)
utils.plot_paretofront_profile_changes(plot_dic,add_ideal_profile=True,colormap='tab20',toeoff_color='xkcd:shamrock green',xlabel=True)
ax.set_yticks([-1.5,-0.5,0,0.5,1.5])
ax.set_ylim((-1.5,1.5))
#****************************************
# monoarticular noload hip
plot_dic = {'data':utils.smooth(mean_mono_noload_hip_torque_paretofront,5,multidim=True),
            'joint_data':utils.smooth(rra_dataset['mean_norm_noload_hipjoint_moment'],5),'joint_color':'xkcd:shamrock green',
            'avg_toeoff':noload_mean_toe_off,'indices':mono_noload_indices,'title':'monoarticular hip\n torque, noload',
            'ideal_data':ideal_profiles_dataset['mean_mono_noload_hip_torque'],'ideal_color':mycolors['french rose']}
ax = plt.subplot(2,4,7)
utils.plot_paretofront_profile_changes(plot_dic,colormap='tab20',include_colorbar=False,
                                        add_ideal_profile=True,toeoff_color='xkcd:shamrock green',xlabel=True)
ax.set_yticks([-1.5,-0.5,0,0.5,1.5])
ax.set_ylim((-1.5,1.5))
# monoarticular noload knee
plot_dic = {'data':utils.smooth(mean_mono_noload_knee_torque_paretofront,5,multidim=True),
            'joint_data':utils.smooth(rra_dataset['mean_norm_noload_kneejoint_moment'],5),'joint_color':'xkcd:shamrock green',
            'avg_toeoff':noload_mean_toe_off,'indices':mono_noload_indices,'title':'monoarticular knee\n torque, noload',
            'ideal_data':ideal_profiles_dataset['mean_mono_noload_knee_torque'],'ideal_color':mycolors['french rose']}
ax = plt.subplot(2,4,8)
utils.plot_paretofront_profile_changes(plot_dic,colormap='tab20',toeoff_color='xkcd:shamrock green',add_ideal_profile=True,xlabel=True)
ax.set_yticks([-1.5,-0.5,0.5,1.5])
ax.set_ylim((-1.5,1.5))
fig.tight_layout(h_pad=-1.5, w_pad=-1.5)
fig.subplots_adjust(top=0.98, bottom=0.075, left=0.05, right=0.95,hspace=0.20,wspace=0.25)
fig.savefig('./Figures/Paretofront/Subjects_Pareto/PaperFigure_Paretofront_TorqueProfiles.pdf',orientation='landscape',bbox_inches='tight')
plt.show()

#************************************************************************************
#************************************************************************************
# power profile
fig, axes = plt.subplots(nrows=2,ncols=4,num='Pareto Curve: loaded mono vs bi',figsize=(14.8, 9.6))
# biarticular loaded hip
plot_dic = {'data':utils.smooth(mean_bi_loaded_hip_power_paretofront,5,multidim=True),
            'joint_data':utils.smooth(rra_dataset['mean_norm_loaded_hipjoint_power'],5),'joint_color':'k',
            'avg_toeoff':loaded_mean_toe_off,'indices':bi_loaded_indices,'title':'biarticular hip\n power, loaded',
            'ideal_data':ideal_profiles_dataset['mean_bi_loaded_hip_power'],'ideal_color':mycolors['crimson red']}
ax = plt.subplot(2,4,1)
utils.plot_paretofront_profile_changes(plot_dic,colormap='tab20',add_ideal_profile=True,include_colorbar=False,
                                        toeoff_color='k',ylabel='flexion/extension\n power (W/kg)')
ax.set_yticks([-2.5,-1.5,-0.5,0.5,1.5,2.5,3.5])
ax.set_ylim((-2.5,3.5))
# biarticular loaded knee
plot_dic = {'data':utils.smooth(mean_bi_loaded_knee_power_paretofront,5,multidim=True),
            'joint_data':utils.smooth(rra_dataset['mean_norm_loaded_kneejoint_power'],5),'joint_color':'k',
            'avg_toeoff':loaded_mean_toe_off,'indices':bi_loaded_indices,'title':'biarticular knee\n power, loaded',
            'ideal_data':ideal_profiles_dataset['mean_bi_loaded_knee_power'],'ideal_color':mycolors['crimson red']}
ax = plt.subplot(2,4,2)
utils.plot_paretofront_profile_changes(plot_dic,add_ideal_profile=True,colormap='tab20',toeoff_color='k')
ax.set_yticks([-2.5,-1.5,-0.5,0.5,1.5,2.5,3.5])
ax.set_ylim((-2.5,3.5))
#****************************************
# monoarticular loaded hip
plot_dic = {'data':utils.smooth(mean_mono_loaded_hip_power_paretofront,5,multidim=True),
            'joint_data':utils.smooth(rra_dataset['mean_norm_loaded_hipjoint_power'],5),'joint_color':'k',
            'avg_toeoff':loaded_mean_toe_off,'indices':mono_loaded_indices,'title':'monoarticular hip\n power, loaded',
            'ideal_data':ideal_profiles_dataset['mean_mono_loaded_hip_power'],'ideal_color':mycolors['crimson red']}
ax = plt.subplot(2,4,3)
utils.plot_paretofront_profile_changes(plot_dic,add_ideal_profile=True,colormap='tab20',include_colorbar=False,toeoff_color='k')
ax.set_yticks([-2.5,-1.5,-0.5,0.5,1.5,2.5,3.5])
ax.set_ylim((-2.5,3.5))
# monoarticular loaded knee
plot_dic = {'data':utils.smooth(mean_mono_loaded_knee_power_paretofront,5,multidim=True),
            'joint_data':utils.smooth(rra_dataset['mean_norm_loaded_kneejoint_power'],5),'joint_color':'k',
            'avg_toeoff':loaded_mean_toe_off,'indices':mono_loaded_indices,'title':'monoarticular knee\n power, loaded',
            'ideal_data':ideal_profiles_dataset['mean_mono_loaded_knee_power'],'ideal_color':mycolors['crimson red']}
ax = plt.subplot(2,4,4)
utils.plot_paretofront_profile_changes(plot_dic,add_ideal_profile=True,colormap='tab20',toeoff_color='k')
ax.set_yticks([-2.5,-1.5,-0.5,0.5,1.5,2.5,3.5])
ax.set_ylim((-2.5,3.5))
#****************************************
# biarticular noload hip
plot_dic = {'data':utils.smooth(mean_bi_noload_hip_power_paretofront,5,multidim=True),
            'joint_data':utils.smooth(rra_dataset['mean_norm_noload_hipjoint_power'],5),'joint_color':'xkcd:shamrock green',
            'avg_toeoff':noload_mean_toe_off,'indices':bi_noload_indices,'title':'biarticular hip\n power, noload',
            'ideal_data':ideal_profiles_dataset['mean_bi_noload_hip_power'],'ideal_color':mycolors['french rose']}
ax = plt.subplot(2,4,5)
utils.plot_paretofront_profile_changes(plot_dic,colormap='tab20',include_colorbar=False,add_ideal_profile=True,
                        toeoff_color='xkcd:shamrock green',ylabel='flexion/extension\n power (W/kg)',xlabel=True)
ax.set_yticks([-2.5,-1.5,-0.5,0.5,1.5,2.5,3.5])
ax.set_ylim((-2.5,3.5))
# biarticular noload knee
plot_dic = {'data':utils.smooth(mean_bi_noload_knee_power_paretofront,5,multidim=True),
            'joint_data':utils.smooth(rra_dataset['mean_norm_noload_kneejoint_power'],5),'joint_color':'xkcd:shamrock green',
            'avg_toeoff':noload_mean_toe_off,'indices':bi_noload_indices,'title':'biarticular knee\n power, noload',
            'ideal_data':ideal_profiles_dataset['mean_bi_noload_knee_power'],'ideal_color':mycolors['french rose']}
ax = plt.subplot(2,4,6)
utils.plot_paretofront_profile_changes(plot_dic,add_ideal_profile=True,colormap='tab20',toeoff_color='xkcd:shamrock green',xlabel=True)
ax.set_yticks([-2.5,-1.5,-0.5,0.5,1.5,2.5,3.5])
ax.set_ylim((-2.5,3.5))
#****************************************
# monoarticular noload hip
plot_dic = {'data':utils.smooth(mean_mono_noload_hip_power_paretofront,5,multidim=True),
            'joint_data':utils.smooth(rra_dataset['mean_norm_noload_hipjoint_power'],5),'joint_color':'xkcd:shamrock green',
            'avg_toeoff':noload_mean_toe_off,'indices':mono_noload_indices,'title':'monoarticular hip\n power, noload',
            'ideal_data':ideal_profiles_dataset['mean_mono_noload_hip_power'],'ideal_color':mycolors['french rose']}
ax = plt.subplot(2,4,7)
utils.plot_paretofront_profile_changes(plot_dic,add_ideal_profile=True,colormap='tab20',include_colorbar=False,toeoff_color='xkcd:shamrock green',xlabel=True)
ax.set_yticks([-2.5,-1.5,-0.5,0.5,1.5,2.5,3.5])
ax.set_ylim((-2.5,3.5))
# monoarticular noload knee
plot_dic = {'data':utils.smooth(mean_mono_noload_knee_power_paretofront,5,multidim=True),
            'joint_data':utils.smooth(rra_dataset['mean_norm_noload_kneejoint_power'],5),'joint_color':'xkcd:shamrock green',
            'avg_toeoff':noload_mean_toe_off,'indices':mono_noload_indices,'title':'monoarticular knee\n power, noload',
            'ideal_data':ideal_profiles_dataset['mean_mono_noload_knee_power'],'ideal_color':mycolors['french rose']}
ax = plt.subplot(2,4,8)
utils.plot_paretofront_profile_changes(plot_dic,add_ideal_profile=True,colormap='tab20',toeoff_color='xkcd:shamrock green',xlabel=True)
ax.set_yticks([-2.5,-1.5,-0.5,0.5,1.5,2.5,3.5])
ax.set_ylim((-2.5,3.5))
fig.tight_layout(h_pad=-1.5, w_pad=-1.5)
fig.subplots_adjust(top=0.98, bottom=0.075, left=0.05, right=0.95,hspace=0.20,wspace=0.25)
fig.savefig('./Figures/Paretofront/Subjects_Pareto/PaperFigure_Paretofront_PowerProfiles.pdf',orientation='landscape',bbox_inches='tight')
plt.show()

#####################################################################################
# plots front plots
# subjects pareto curve: loaded mono vs biarticular
fig = plt.figure(num='Pareto Curve: loaded mono vs bi',figsize=(12.4, 18.8))
plot_dic = {'x1_data':bi_loaded_metabolics_percent_Paretofront,'x2_data':mono_loaded_metabolics_percent_Paretofront,
          'y1_data':bi_loaded_energy_Paretofront,'y2_data':mono_loaded_energy_Paretofront,
          'color_1':mycolors['crimson red'],'color_2':mycolors['dark purple'],
          'ylabel':'Energy Consumption (W/kg)','xlabel':'Metabolic Reduction (%)'
          }
utils.plot_pareto_curve_subjects (nrows=7,ncols=3,nplot=21,plot_dic=plot_dic,loadcond='loaded')
fig.tight_layout()
fig.savefig('./Figures/Paretofront/Subjects_Pareto/Pareto_Load_Subjects_BiVsMono.pdf',orientation='landscape',bbox_inches='tight')
plt.show()

# subjects pareto curve: noload mono vs biarticular

plot_dic = {'x1_data':bi_noload_metabolics_percent_Paretofront,'x2_data':mono_noload_metabolics_percent_Paretofront,
          'y1_data':bi_noload_energy_Paretofront,'y2_data':mono_noload_energy_Paretofront,
          'color_1':mycolors['crimson red'],'color_2':mycolors['dark purple'],
          'ylabel':'Energy Consumption (W/kg)','xlabel':'Metabolic Reduction (%)'
          }
fig = plt.figure(num='Pareto Curve: noload mono vs bi',figsize=(12.4, 18.8))
utils.plot_pareto_curve_subjects (nrows=7,ncols=3,nplot=21,plot_dic=plot_dic,loadcond='noload')
fig.tight_layout()
fig.savefig('./Figures/Paretofront/Subjects_Pareto/Pareto_Noload_Subjects_BiVsMono.pdf',orientation='landscape',bbox_inches='tight')
plt.show()
