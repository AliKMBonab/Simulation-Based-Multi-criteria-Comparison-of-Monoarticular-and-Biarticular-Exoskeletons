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
import matlab.engine
from Colors import colors as mycolors
#####################################################################################
subjects = ['05','07','09','10','11','12','14']
trials_num = ['01','02','03']
gait_cycle = np.linspace(0,100,1000)
#####################################################################################
# Reading CSV files into a dictionary and constructing gls
rra_dataset = utils.csv2numpy('D:/Ali.K.M.Bonab/Walking_Mass_Inertia_Effect/Data/Data/RRA/rra_final_data.csv') 
unassist_dataset = utils.csv2numpy('D:/Ali.K.M.Bonab/Walking_Mass_Inertia_Effect/Data/Data/Unassist/unassist_final_data.csv') 
# pareto exo torque dataset
directory = 'D:/Ali.K.M.Bonab/Walking_Mass_Inertia_Effect/Data/Data/Pareto/*_torque.csv'
files = enumerate(glob.iglob(directory), 1)
exo_torque_dataset = {pathlib.PurePath(f[1]).stem: np.loadtxt(f[1], delimiter=',') for f in files}
# pareto exo power dataset
directory = 'D:/Ali.K.M.Bonab/Walking_Mass_Inertia_Effect/Data/Data/Pareto/*_power.csv'
files = enumerate(glob.iglob(directory), 1)
exo_power_dataset = {pathlib.PurePath(f[1]).stem: np.loadtxt(f[1], delimiter=',') for f in files}
# pareto exo energy dataset
directory = 'D:/Ali.K.M.Bonab/Walking_Mass_Inertia_Effect/Data/Data/Pareto/*_energy.csv'
files = enumerate(glob.iglob(directory), 1)
assisted_energy_dataset = {pathlib.PurePath(f[1]).stem: np.loadtxt(f[1], delimiter=',') for f in files}
# unassist energy dataset
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
# metabolics cost reduction percents
bi_loaded_metabolics_percent = utils.pareto_metabolics_reduction(assisted_energy_dataset['biarticular_pareto_load_metabolics_energy'],unassisted_energy_dataset['loaded_metabolics_energy'])
bi_noload_metabolics_percent = utils.pareto_metabolics_reduction(assisted_energy_dataset['biarticular_pareto_noload_metabolics_energy'],unassisted_energy_dataset['noload_metabolics_energy'])
mono_loaded_metabolics_percent = utils.pareto_metabolics_reduction(assisted_energy_dataset['monoarticular_pareto_load_metabolics_energy'],unassisted_energy_dataset['loaded_metabolics_energy'])
mono_noload_metabolics_percent = utils.pareto_metabolics_reduction(assisted_energy_dataset['monoarticular_pareto_noload_metabolics_energy'],unassisted_energy_dataset['noload_metabolics_energy'])

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

# actuators energy from processed mean data
proc_bi_loaded_hip_energy= np.reshape(assisted_energy_dataset['biarticular_pareto_load_processed_hipactuator_energy'],(25,21),order='F')
proc_bi_loaded_knee_energy= np.reshape(assisted_energy_dataset['biarticular_pareto_load_processed_kneeactuator_energy'],(25,21),order='F')
proc_bi_loaded_energy = proc_bi_loaded_hip_energy + proc_bi_loaded_knee_energy
proc_bi_noload_hip_energy= np.reshape(assisted_energy_dataset['biarticular_pareto_noload_processed_hipactuator_energy'],(25,21),order='F')
proc_bi_noload_knee_energy= np.reshape(assisted_energy_dataset['biarticular_pareto_noload_processed_kneeactuator_energy'],(25,21),order='F')
proc_bi_noload_energy = proc_bi_noload_hip_energy + proc_bi_noload_knee_energy
proc_mono_loaded_hip_energy= np.reshape(assisted_energy_dataset['monoarticular_pareto_load_processed_hipactuator_energy'],(25,21),order='F')
proc_mono_loaded_knee_energy= np.reshape(assisted_energy_dataset['monoarticular_pareto_load_processed_kneeactuator_energy'],(25,21),order='F')
proc_mono_loaded_energy = proc_mono_loaded_hip_energy + proc_mono_loaded_knee_energy
proc_mono_noload_hip_energy= np.reshape(assisted_energy_dataset['monoarticular_pareto_noload_processed_hipactuator_energy'],(25,21),order='F')
proc_mono_noload_knee_energy= np.reshape(assisted_energy_dataset['monoarticular_pareto_noload_processed_kneeactuator_energy'],(25,21),order='F')
proc_mono_noload_energy = proc_mono_noload_hip_energy + proc_mono_noload_knee_energy
# metabolics cost reduction percents
proc_bi_loaded_metabolics_percent = utils.pareto_metabolics_reduction(assisted_energy_dataset['biarticular_pareto_load_processed_metabolics_energy'],unassisted_energy_dataset['loaded_metabolics_processed_energy'])
proc_bi_noload_metabolics_percent = utils.pareto_metabolics_reduction(assisted_energy_dataset['biarticular_pareto_noload_processed_metabolics_energy'],unassisted_energy_dataset['noload_metabolics_processed_energy'])
proc_mono_loaded_metabolics_percent = utils.pareto_metabolics_reduction(assisted_energy_dataset['monoarticular_pareto_load_processed_metabolics_energy'],unassisted_energy_dataset['loaded_metabolics_processed_energy'])
proc_mono_noload_metabolics_percent = utils.pareto_metabolics_reduction(assisted_energy_dataset['monoarticular_pareto_noload_processed_metabolics_energy'],unassisted_energy_dataset['noload_metabolics_processed_energy'])

#####################################################################################
# plots
# subjects pareto curve: loaded mono vs biarticular

plot_dic = {'x1_data':bi_loaded_metabolics_percent,'x2_data':mono_loaded_metabolics_percent,
          'y1_data':bi_loaded_energy,'y2_data':mono_loaded_energy,
          'color_1':mycolors['crimson red'],'color_2':mycolors['dark purple'],
          'ylabel':'Energy Consumption (W/kg)','xlabel':'Metabolic Reduction (%)'
          }
fig = plt.figure(num='Pareto Curve: loaded mono vs bi',figsize=(10.4, 18.8))
utils.plot_pareto_curve_subjects (nrows=7,ncols=3,nplot=21,plot_dic=plot_dic,loadcond='loaded')
fig.tight_layout()
fig.savefig('./Figures/Pareto/Subjects_Pareto/Pareto_Load_Subjects_BiVsMono.pdf',orientation='landscape',bbox_inches='tight')
plt.show()

# subjects pareto curve: noload mono vs biarticular

plot_dic = {'x1_data':bi_noload_metabolics_percent,'x2_data':mono_noload_metabolics_percent,
          'y1_data':bi_noload_energy,'y2_data':mono_noload_energy,
          'color_1':mycolors['crimson red'],'color_2':mycolors['dark purple'],
          'ylabel':'Energy Consumption (W/kg)','xlabel':'Metabolic Reduction (%)'
          }
fig = plt.figure(num='Pareto Curve: noload mono vs bi',figsize=(10.4, 18.8))
utils.plot_pareto_curve_subjects (nrows=7,ncols=3,nplot=21,plot_dic=plot_dic,loadcond='noload')
fig.tight_layout()
fig.savefig('./Figures/Pareto/Subjects_Pareto/Pareto_Noload_Subjects_BiVsMono.pdf',orientation='landscape',bbox_inches='tight')
plt.show()
# subjects pareto curve: loaded mono vs biarticular

plot_dic = {'x1_data':bi_loaded_metabolics_percent,'x2_data':mono_loaded_metabolics_percent,
          'y1_data':proc_bi_loaded_energy,'y2_data':proc_mono_loaded_energy,
          'color_1':mycolors['crimson red'],'color_2':mycolors['dark purple'],
          'ylabel':'Energy Consumption (W/kg)','xlabel':'Metabolic Reduction (%)'
          }
fig = plt.figure(num='Pareto Curve: loaded mono vs bi - Processed',figsize=(10.4, 18.8))
utils.plot_pareto_curve_subjects (nrows=7,ncols=3,nplot=21,plot_dic=plot_dic,loadcond='loaded')
fig.tight_layout()
fig.savefig('./Figures/Pareto/Subjects_Pareto/Pareto_Load_Subjects_BiVsMono_Processed.pdf',orientation='landscape',bbox_inches='tight')
plt.show()

# subjects pareto curve: noload mono vs biarticular

plot_dic = {'x1_data':bi_noload_metabolics_percent,'x2_data':mono_noload_metabolics_percent,
          'y1_data':proc_bi_noload_energy,'y2_data':proc_mono_noload_energy,
          'color_1':mycolors['crimson red'],'color_2':mycolors['dark purple'],
          'ylabel':'Energy Consumption (W/kg)','xlabel':'Metabolic Reduction (%)'
          }
fig = plt.figure(num='Pareto Curve: noload mono vs bi - Processed',figsize=(10.4, 18.8))
utils.plot_pareto_curve_subjects (nrows=7,ncols=3,nplot=21,plot_dic=plot_dic,loadcond='noload')
fig.tight_layout()
fig.savefig('./Figures/Pareto/Subjects_Pareto/Pareto_Noload_Subjects_BiVsMono_Processed.pdf',orientation='landscape',bbox_inches='tight')
plt.show()

#####################################################################################
# hip loaded power profiles

plot_dic = {'avg_1':mean_bi_loaded_hip_power,'std_1':std_bi_loaded_hip_power,
          'avg_2':mean_mono_loaded_hip_power,'std_2':std_mono_loaded_hip_power,
      'joint_avg':rra_dataset['mean_norm_loaded_hipjoint_power'],'joint_std':rra_dataset['std_norm_loaded_hipjoint_power'],
        'color_1':mycolors['crimson red'], 'color_2':mycolors['dark purple'],
     'avg_toeoff':loaded_mean_toe_off,'ylabel':'power (W/kg)'}

fig = plt.figure(num='Pareto hip power: loaded mono vs bi',figsize=(12.4, 10.8))
utils.plot_pareto_shaded_avg(plot_dic,loadcond='loaded')
fig.tight_layout()
fig.savefig('./Figures/Pareto/Subjects_Pareto/Pareto_loaded_hippower_bivsmono.pdf',orientation='landscape',bbox_inches='tight')
plt.show()

# hip noload power profiles

plot_dic = {'avg_1':mean_bi_noload_hip_power,'std_1':std_bi_noload_hip_power,
          'avg_2':mean_mono_noload_hip_power,'std_2':std_mono_noload_hip_power,
      'joint_avg':rra_dataset['mean_norm_noload_hipjoint_power'],'joint_std':rra_dataset['std_norm_noload_hipjoint_power'],
        'color_1':mycolors['crimson red'], 'color_2':mycolors['dark purple'],
     'avg_toeoff':loaded_mean_toe_off,'ylabel':'power (W/kg)'}

fig = plt.figure(num='Pareto hip power: noload mono vs bi',figsize=(12.4, 10.8))
utils.plot_pareto_shaded_avg(plot_dic,loadcond='loaded')
fig.tight_layout()
fig.savefig('./Figures/Pareto/Subjects_Pareto/Pareto_noload_hippower_bivsmono.pdf',orientation='landscape',bbox_inches='tight')
plt.show()

# knee loaded power profiles

plot_dic = {'avg_1':mean_bi_loaded_knee_power,'std_1':std_bi_loaded_knee_power,
          'avg_2':mean_mono_loaded_knee_power,'std_2':std_mono_loaded_knee_power,
      'joint_avg':rra_dataset['mean_norm_loaded_kneejoint_power'],'joint_std':rra_dataset['std_norm_loaded_kneejoint_power'],
        'color_1':mycolors['crimson red'], 'color_2':mycolors['dark purple'],
     'avg_toeoff':loaded_mean_toe_off,'ylabel':'power (W/kg)'}

fig = plt.figure(num='Pareto knee power: noload mono vs bi',figsize=(12.4, 10.8))
utils.plot_pareto_shaded_avg(plot_dic,loadcond='loaded')
fig.tight_layout()
fig.savefig('./Figures/Pareto/Subjects_Pareto/Pareto_loaded_kneepower_bivsmono.pdf',orientation='landscape',bbox_inches='tight')
plt.show()

# knee noload power profiles

plot_dic = {'avg_1':mean_bi_noload_knee_power,'std_1':std_bi_noload_knee_power,
          'avg_2':mean_mono_noload_knee_power,'std_2':std_mono_noload_knee_power,
      'joint_avg':rra_dataset['mean_norm_noload_kneejoint_power'],'joint_std':rra_dataset['std_norm_noload_kneejoint_power'],
        'color_1':mycolors['crimson red'], 'color_2':mycolors['dark purple'],
     'avg_toeoff':loaded_mean_toe_off,'ylabel':'power (W/kg)'}

fig = plt.figure(num='Pareto knee torque: loaded mono vs bi',figsize=(12.4, 10.8))
utils.plot_pareto_shaded_avg(plot_dic,loadcond='loaded')
fig.tight_layout()
fig.savefig('./Figures/Pareto/Subjects_Pareto/Pareto_noload_kneepower_bivsmono.pdf',orientation='landscape',bbox_inches='tight')
plt.show()

# hip loaded torque profiles

plot_dic = {'avg_1':mean_bi_loaded_hip_torque,'std_1':std_bi_loaded_hip_torque,
          'avg_2':mean_mono_loaded_hip_torque,'std_2':std_mono_loaded_hip_torque,
      'joint_avg':rra_dataset['mean_norm_loaded_hipjoint_moment'],'joint_std':rra_dataset['std_norm_loaded_hipjoint_moment'],
        'color_1':mycolors['crimson red'], 'color_2':mycolors['dark purple'],
     'avg_toeoff':loaded_mean_toe_off,'ylabel':'torque (N-m/kg)'}

fig = plt.figure(num='Pareto hip torque: loaded mono vs bi',figsize=(12.4, 10.8))
utils.plot_pareto_shaded_avg(plot_dic,loadcond='loaded')
fig.tight_layout()
fig.savefig('./Figures/Pareto/Subjects_Pareto/Pareto_loaded_hiptorque_bivsmono.pdf',orientation='landscape',bbox_inches='tight')
plt.show()

# hip noload torque profiles

plot_dic = {'avg_1':mean_bi_noload_hip_torque,'std_1':std_bi_noload_hip_torque,
          'avg_2':mean_mono_noload_hip_torque,'std_2':std_mono_noload_hip_torque,
      'joint_avg':rra_dataset['mean_norm_noload_hipjoint_moment'],'joint_std':rra_dataset['std_norm_noload_hipjoint_moment'],
        'color_1':mycolors['crimson red'], 'color_2':mycolors['dark purple'],
     'avg_toeoff':loaded_mean_toe_off,'ylabel':'torque (N-m/kg)'}

fig = plt.figure(num='Pareto hip torque: loaded mono vs bi',figsize=(12.4, 10.8))
utils.plot_pareto_shaded_avg(plot_dic,loadcond='loaded')
fig.tight_layout()
fig.savefig('./Figures/Pareto/Subjects_Pareto/Pareto_noload_hiptorque_bivsmono.pdf',orientation='landscape',bbox_inches='tight')
plt.show()

# knee loaded torque profiles

plot_dic = {'avg_1':mean_bi_loaded_knee_torque,'std_1':std_bi_loaded_knee_torque,
          'avg_2':mean_mono_loaded_knee_torque,'std_2':std_mono_loaded_knee_torque,
      'joint_avg':rra_dataset['mean_norm_loaded_kneejoint_moment'],'joint_std':rra_dataset['std_norm_loaded_kneejoint_moment'],
        'color_1':mycolors['crimson red'], 'color_2':mycolors['dark purple'],
     'avg_toeoff':loaded_mean_toe_off,'ylabel':'torque (N-m/kg)'}

fig = plt.figure(num='Pareto knee torque: loaded mono vs bi',figsize=(12.4, 10.8))
utils.plot_pareto_shaded_avg(plot_dic,loadcond='loaded')
fig.tight_layout()
fig.savefig('./Figures/Pareto/Subjects_Pareto/Pareto_loaded_kneetorque_bivsmono.pdf',orientation='landscape',bbox_inches='tight')
plt.show()

# knee noload torque profiles

plot_dic = {'avg_1':mean_bi_noload_knee_torque,'std_1':std_bi_noload_knee_torque,
          'avg_2':mean_mono_noload_knee_torque,'std_2':std_mono_noload_knee_torque,
      'joint_avg':rra_dataset['mean_norm_noload_kneejoint_moment'],'joint_std':rra_dataset['std_norm_noload_kneejoint_moment'],
        'color_1':mycolors['crimson red'], 'color_2':mycolors['dark purple'],
     'avg_toeoff':loaded_mean_toe_off,'ylabel':'torque (N-m/kg)'}

fig = plt.figure(num='Pareto knee torque: loaded mono vs bi',figsize=(12.4, 10.8))
utils.plot_pareto_shaded_avg(plot_dic,loadcond='loaded')
fig.tight_layout()
fig.savefig('./Figures/Pareto/Subjects_Pareto/Pareto_noload_kneetorque_bivsmono.pdf',orientation='landscape',bbox_inches='tight')
plt.show()
