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
hip_label = []
knee_label = []
for i in [70,60,50,40,30]:
      for j in [70,60,50,40,30]:
        hip_label.append(i)
        knee_label.append(j)
label = [list(a) for a in zip(hip_label,knee_label)] 
rows = np.arange(1,26,1)
columns = ['hip weight','knee weight']
fig, ax = plt.subplots(figsize=(6.4, 6.8))
table = ax.table(cellText=label,rowLabels=rows.tolist(),colLabels=columns,loc='center')
table.scale(1,1.5)
table.set_fontsize(14)
ax.axis('off')
fig.savefig('./Figures/Labeling_Table.pdf',orientation='landscape',bbox_inches='tight')
plt.show()
#####################################################################################
# Reading CSV files into a dictionary and constructing gls
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
# joint moment dataset
directory = 'D:\Ali.K.M.Bonab\Walking_Mass_Inertia_Effect\Data\Data\RRA\*_torque.csv'
files = enumerate(glob.iglob(directory), 1)
jointmoment_dataset = {pathlib.PurePath(f[1]).stem: np.loadtxt(f[1], delimiter=',') for f in files}
# joint power dataset
directory = 'D:\Ali.K.M.Bonab\Walking_Mass_Inertia_Effect\Data\Data\RRA\*_power.csv'
files = enumerate(glob.iglob(directory), 1)
jointpower_dataset = {pathlib.PurePath(f[1]).stem: np.loadtxt(f[1], delimiter=',') for f in files}
# gls
gl_noload = {'noload_subject{}_trial{}'.format(i,j): utils.construct_gl_mass_side(subjectno=i,trialno=j,loadcond='noload') for i in subjects for j in trials_num}
gl_loaded = {'loaded_subject{}_trial{}'.format(i,j): utils.construct_gl_mass_side(subjectno=i,trialno=j,loadcond='loaded') for i in subjects for j in trials_num}
#####################################################################################
# Processing Data
# toe-off
noload_mean_toe_off,_,loaded_mean_toe_off,_ = utils.toe_off_avg_std(gl_noload,gl_loaded)
# hip joint moment
loaded_hipjoint_moment = utils.normalize_direction_data(jointmoment_dataset['loaded_hipjoint_torque'],gl_noload,direction=True,normalize=False)
noload_hipjoint_moment = utils.normalize_direction_data(jointmoment_dataset['noload_hipjoint_torque'],gl_noload,direction=True,normalize=False)
mean_loaded_hipjoint_moment,std_loaded_hipjoint_moment = utils.mean_std_over_subjects(loaded_hipjoint_moment)
mean_noload_hipjoint_moment,std_noload_hipjoint_moment = utils.mean_std_over_subjects(noload_hipjoint_moment)
# knee joint moment
loaded_kneejoint_moment = utils.normalize_direction_data(jointmoment_dataset['loaded_kneejoint_torque'],gl_noload,direction=True,normalize=False)
noload_kneejoint_moment = utils.normalize_direction_data(jointmoment_dataset['noload_kneejoint_torque'],gl_noload,direction=True,normalize=False)
mean_loaded_kneejoint_moment,std_loaded_kneejoint_moment = utils.mean_std_over_subjects(loaded_kneejoint_moment)
mean_noload_kneejoint_moment,std_noload_kneejoint_moment = utils.mean_std_over_subjects(noload_kneejoint_moment)
# hip joint power
loaded_hipjoint_power = utils.normalize_direction_data(jointpower_dataset['loaded_hipjoint_power'],gl_noload,direction=False,normalize=False)
noload_hipjoint_power = utils.normalize_direction_data(jointpower_dataset['noload_hipjoint_power'],gl_noload,direction=False,normalize=False)
mean_loaded_hipjoint_power,std_loaded_hipjoint_power = utils.mean_std_over_subjects(loaded_hipjoint_power)
mean_noload_hipjoint_power,std_noload_hipjoint_power = utils.mean_std_over_subjects(noload_hipjoint_power)
# knee joint power
loaded_kneejoint_power = utils.normalize_direction_data(jointpower_dataset['loaded_kneejoint_power'],gl_noload,direction=False,normalize=False)
noload_kneejoint_power = utils.normalize_direction_data(jointpower_dataset['noload_kneejoint_power'],gl_noload,direction=False,normalize=False)
mean_loaded_kneejoint_power,std_loaded_kneejoint_power = utils.mean_std_over_subjects(loaded_kneejoint_power)
mean_noload_kneejoint_power,std_noload_kneejoint_power = utils.mean_std_over_subjects(noload_kneejoint_power)
# actuators energy
# actuators energy
bi_loaded_hip_energy= np.reshape(assisted_energy_dataset['biarticular_pareto_load_hipactuator_energy'],(25,7),order='F')
bi_loaded_knee_energy= np.reshape(assisted_energy_dataset['biarticular_pareto_load_kneeactuator_energy'],(25,7),order='F')
bi_loaded_energy = bi_loaded_hip_energy + bi_loaded_knee_energy
bi_noload_hip_energy= np.reshape(assisted_energy_dataset['biarticular_pareto_noload_hipactuator_energy'],(25,7),order='F')
bi_noload_knee_energy= np.reshape(assisted_energy_dataset['biarticular_pareto_noload_kneeactuator_energy'],(25,7),order='F')
bi_noload_energy = bi_noload_hip_energy + bi_noload_knee_energy
mono_loaded_hip_energy= np.reshape(assisted_energy_dataset['monoarticular_pareto_load_hipactuator_energy'],(25,7),order='F')
mono_loaded_knee_energy= np.reshape(assisted_energy_dataset['monoarticular_pareto_load_kneeactuator_energy'],(25,7),order='F')
mono_loaded_energy = mono_loaded_hip_energy + mono_loaded_knee_energy
mono_noload_hip_energy= np.reshape(assisted_energy_dataset['monoarticular_pareto_noload_hipactuator_energy'],(25,7),order='F')
mono_noload_knee_energy= np.reshape(assisted_energy_dataset['monoarticular_pareto_noload_kneeactuator_energy'],(25,7),order='F')
mono_noload_energy = mono_noload_hip_energy + mono_noload_knee_energy
bi_loaded_metabolics_energy = np.reshape(assisted_energy_dataset['biarticular_pareto_load_metabolics_energy'],(25,7),order='F')
mono_loaded_metabolics_energy = np.reshape(assisted_energy_dataset['monoarticular_pareto_load_metabolics_energy'],(25,7),order='F')
bi_noload_metabolics_energy = np.reshape(assisted_energy_dataset['biarticular_pareto_noload_metabolics_energy'],(25,7),order='F')
mono_noload_metabolics_energy = np.reshape(assisted_energy_dataset['monoarticular_pareto_noload_metabolics_energy'],(25,7),order='F')

#####################################################################################
# plots
print('\n')
a = input('subject 05 loaded/noload hip/knee POWER comparison between monoarticular and biarticular (N, Y) :')
print('\n')
if a.lower() == 'y':
  # subject 05 loaded hip power comparison between monoarticular and biarticular
  plot_dic = {'avg_1':exo_power_dataset['biarticular_pareto_load_hipactuator_power'][:,0:25],
              'avg_2':exo_power_dataset['monoarticular_pareto_load_hipactuator_power'][:,0:25],
          'joint_avg':mean_loaded_hipjoint_power,'joint_std':std_loaded_hipjoint_power,
          'color_1':mycolors['crimson red'], 'color_2':mycolors['dark purple'],
      'avg_toeoff':loaded_mean_toe_off,'ylabel':'power (W)'}

  fig = plt.figure(num='Test: Subject 05 Loaded: Hip actuator power',figsize=(12.4, 10.8))
  utils.plot_pareto_shaded_avg(plot_dic,loadcond='loaded',fill_std=False)
  fig.tight_layout()
  fig.savefig('./Figures/Tests/Test_Subject05_Loaded_HipPower.pdf',orientation='landscape',bbox_inches='tight')
  plt.show()

  # subject 05 noload hip power comparison between monoarticular and biarticular
  plot_dic = {'avg_1':exo_power_dataset['biarticular_pareto_noload_hipactuator_power'][:,0:25],
              'avg_2':exo_power_dataset['monoarticular_pareto_noload_hipactuator_power'][:,0:25],
          'joint_avg':mean_noload_hipjoint_power,'joint_std':std_noload_hipjoint_power,
          'color_1':mycolors['crimson red'], 'color_2':mycolors['dark purple'],
      'avg_toeoff':noload_mean_toe_off,'ylabel':'power (W)'}

  fig = plt.figure(num='Test: Subject 05 Noloaded: Hip actuator power',figsize=(12.4, 10.8))
  utils.plot_pareto_shaded_avg(plot_dic,loadcond='noload',fill_std=False)
  fig.tight_layout()
  fig.savefig('./Figures/Tests/Test_Subject05_Noload_HipPower.pdf',orientation='landscape',bbox_inches='tight')
  plt.show()

  # subject 05 loaded knee power comparison between monoarticular and biarticular
  plot_dic = {'avg_1':exo_power_dataset['biarticular_pareto_load_kneeactuator_power'][:,0:25],
              'avg_2':exo_power_dataset['monoarticular_pareto_load_kneeactuator_power'][:,0:25],
          'joint_avg':mean_loaded_kneejoint_power,'joint_std':std_loaded_kneejoint_power,
          'color_1':mycolors['crimson red'], 'color_2':mycolors['dark purple'],
      'avg_toeoff':loaded_mean_toe_off,'ylabel':'power (W)'}

  fig = plt.figure(num='Test: Subject 05 Loaded: Knee actuator power',figsize=(12.4, 10.8))
  utils.plot_pareto_shaded_avg(plot_dic,loadcond='loaded',fill_std=False)
  fig.tight_layout()
  fig.savefig('./Figures/Tests/Test_Subject05_Loaded_KneePower.pdf',orientation='landscape',bbox_inches='tight')
  plt.show()

  # subject 05 noload knee power comparison between monoarticular and biarticular
  plot_dic = {'avg_1':exo_power_dataset['biarticular_pareto_noload_kneeactuator_power'][:,0:25],
              'avg_2':exo_power_dataset['monoarticular_pareto_noload_kneeactuator_power'][:,0:25],
          'joint_avg':mean_noload_kneejoint_power,'joint_std':std_noload_kneejoint_power,
          'color_1':mycolors['crimson red'], 'color_2':mycolors['dark purple'],
      'avg_toeoff':noload_mean_toe_off,'ylabel':'power (W)'}

  fig = plt.figure(num='Test: Subject 05 Noloaded: Knee actuator power',figsize=(12.4, 10.8))
  utils.plot_pareto_shaded_avg(plot_dic,loadcond='noload',fill_std=False)
  fig.tight_layout()
  fig.savefig('./Figures/Tests/Test_Subject05_Noload_KneePower.pdf',orientation='landscape',bbox_inches='tight')
  plt.show()

#####################################################################################
a = input('subject 05 loaded/noload hip/knee TORQUE comparison between monoarticular and biarticular (N, Y) :')
print('\n')
if a.lower() == 'y':
  # subject 05 loaded hip comparison between monoarticular and biarticular
  plot_dic = {'avg_1':exo_torque_dataset['biarticular_pareto_load_hipactuator_torque'][:,0:25],
              'avg_2':exo_torque_dataset['monoarticular_pareto_load_hipactuator_torque'][:,0:25],
          'joint_avg':mean_loaded_hipjoint_moment,'joint_std':std_loaded_hipjoint_moment,
          'color_1':mycolors['crimson red'], 'color_2':mycolors['dark purple'],
      'avg_toeoff':loaded_mean_toe_off,'ylabel':'torque (N-m)'}

  fig = plt.figure(num='Test: Subject 05 Loaded: Hip actuator torque',figsize=(12.4, 10.8))
  utils.plot_pareto_shaded_avg(plot_dic,loadcond='loaded',fill_std=False)
  fig.tight_layout()
  fig.savefig('./Figures/Tests/Test_Subject05_Loaded_HipTorque.pdf',orientation='landscape',bbox_inches='tight')
  plt.show()

  # subject 05 noload hip torque comparison between monoarticular and biarticular
  plot_dic = {'avg_1':exo_torque_dataset['biarticular_pareto_noload_hipactuator_torque'][:,0:25],
              'avg_2':exo_torque_dataset['monoarticular_pareto_noload_hipactuator_torque'][:,0:25],
          'joint_avg':mean_noload_hipjoint_moment,'joint_std':std_noload_hipjoint_moment,
          'color_1':mycolors['crimson red'], 'color_2':mycolors['dark purple'],
      'avg_toeoff':noload_mean_toe_off,'ylabel':'torque (N-m)'}

  fig = plt.figure(num='Test: Subject 05 Noloaded: Hip actuator torque',figsize=(12.4, 10.8))
  utils.plot_pareto_shaded_avg(plot_dic,loadcond='noload',fill_std=False)
  fig.tight_layout()
  fig.savefig('./Figures/Tests/Test_Subject05_Noload_HipTorque.pdf',orientation='landscape',bbox_inches='tight')
  plt.show()

  # subject 05 loaded knee torque comparison between monoarticular and biarticular
  plot_dic = {'avg_1':-exo_torque_dataset['biarticular_pareto_load_kneeactuator_torque'][:,0:25],
              'avg_2':-exo_torque_dataset['monoarticular_pareto_load_kneeactuator_torque'][:,0:25],
          'joint_avg':mean_loaded_kneejoint_moment,'joint_std':std_loaded_kneejoint_moment,
          'color_1':mycolors['crimson red'], 'color_2':mycolors['dark purple'],
      'avg_toeoff':loaded_mean_toe_off,'ylabel':'torque (N-m)'}

  fig = plt.figure(num='Test: Subject 05 Loaded: Knee actuator torque',figsize=(12.4, 10.8))
  utils.plot_pareto_shaded_avg(plot_dic,loadcond='loaded',fill_std=False)
  fig.tight_layout()
  fig.savefig('./Figures/Tests/Test_Subject05_Loaded_KneeTorque.pdf',orientation='landscape',bbox_inches='tight')
  plt.show()

  # subject 05 noload knee torque comparison between monoarticular and biarticular
  plot_dic = {'avg_1':-exo_torque_dataset['biarticular_pareto_noload_kneeactuator_torque'][:,0:25],
              'avg_2':-exo_torque_dataset['monoarticular_pareto_noload_kneeactuator_torque'][:,0:25],
          'joint_avg':mean_noload_kneejoint_moment,'joint_std':std_noload_kneejoint_moment,
          'color_1':mycolors['crimson red'], 'color_2':mycolors['dark purple'],
      'avg_toeoff':noload_mean_toe_off,'ylabel':'torque (N-m)'}

  fig = plt.figure(num='Test: Subject 05 Noloaded: Knee actuator torque',figsize=(12.4, 10.8))
  utils.plot_pareto_shaded_avg(plot_dic,loadcond='noload',fill_std=False)
  fig.tight_layout()
  fig.savefig('./Figures/Tests/Test_Subject05_Noload_KneeTorque.pdf',orientation='landscape',bbox_inches='tight')
  plt.show()

#####################################################################################

a = input('subject 05 loaded/noload hip/knee METABOLIC POWER comparison between monoarticular and biarticular (N, Y) :')
print('\n')
if a.lower() == 'y':
  # subject 05 loaded metabolic power comparison between monoarticular and biarticular
  plot_dic = {'avg_1':exo_power_dataset['biarticular_pareto_load_metabolics_power'][:,0:25],
              'avg_2':exo_power_dataset['monoarticular_pareto_load_metabolics_power'][:,0:25],
              'color_1':mycolors['crimson red'], 'color_2':mycolors['dark purple'],
              'avg_toeoff':loaded_mean_toe_off,'ylabel':'metabolic power (W)'}

  fig = plt.figure(num='Test: Subject 05 Loaded: Metabolic Power',figsize=(12.4, 10.8))
  utils.plot_pareto_shaded_avg(plot_dic, loadcond='loaded', fill_std=False, plot_joint=False)
  fig.tight_layout()
  fig.savefig('./Figures/Tests/Test_Subject05_Loaded_MetabolicPower.pdf',orientation='landscape',bbox_inches='tight')
  plt.show()

  # subject 05 noload hip torque comparison between monoarticular and biarticular
  plot_dic = {'avg_1':exo_power_dataset['biarticular_pareto_noload_metabolics_power'][:,0:25],
              'avg_2':exo_power_dataset['monoarticular_pareto_noload_metabolics_power'][:,0:25],
              'color_1':mycolors['crimson red'], 'color_2':mycolors['dark purple'],
              'avg_toeoff':noload_mean_toe_off,'ylabel':'metabolic power (W)'}

  fig = plt.figure(num='Test: Subject 05 Noloaded: Metabolic Power',figsize=(12.4, 10.8))
  utils.plot_pareto_shaded_avg(plot_dic, loadcond='noload',fill_std=False, plot_joint=False)
  fig.tight_layout()
  fig.savefig('./Figures/Tests/Test_Subject05_Noload_MetabolicPower.pdf',orientation='landscape',bbox_inches='tight')
  plt.show()

#####################################################################################
a = input('Hip/Knee Actuators and Metabolics Energy Subject Comparisons (N, Y):')
print('\n')
if a.lower() == 'y':
  # loaded hip actuator energy
  plot_dic = {'data_1':bi_loaded_hip_energy,'data_2':mono_loaded_hip_energy, 'ylabel':'Energy (W/Kg)',
              'color_1':mycolors['crimson red'], 'color_2':mycolors['dark purple']}

  fig = plt.figure(num='Test: Loaded: Hip Energy',figsize=(12.4, 10.8))
  utils.plot_pareto_comparison(plot_dic,loadcond='loaded',compare='subjects')
  fig.tight_layout()
  fig.savefig('./Figures/Tests/Test_Loaded_Hip_EnergyBarPlot_Subject.pdf',orientation='landscape',bbox_inches='tight')
  plt.show()
  # noload hip actuator energy
  plot_dic = {'data_1':bi_noload_hip_energy,'data_2':mono_noload_hip_energy, 'ylabel':'Energy (W/Kg)',
              'color_1':mycolors['crimson red'], 'color_2':mycolors['dark purple']}

  fig = plt.figure(num='Test: Noloaded: Hip Energy',figsize=(12.4, 10.8))
  utils.plot_pareto_comparison(plot_dic,loadcond='noload',compare='subjects')
  fig.tight_layout()
  fig.savefig('./Figures/Tests/Test_Noloaded_Hip_EnergyBarPlot_Subject.pdf',orientation='landscape',bbox_inches='tight')
  plt.show()
  # loaded knee actuator energy
  plot_dic = {'data_1':bi_loaded_knee_energy,'data_2':mono_loaded_knee_energy, 'ylabel':'Energy (W/Kg)',
              'color_1':mycolors['crimson red'], 'color_2':mycolors['dark purple']}

  fig = plt.figure(num='Test: Loaded: Knee Energy',figsize=(12.4, 10.8))
  utils.plot_pareto_comparison(plot_dic,loadcond='loaded',compare='subjects')
  fig.tight_layout()
  fig.savefig('./Figures/Tests/Test_Loaded_Knee_EnergyBarPlot_Subject.pdf',orientation='landscape',bbox_inches='tight')
  plt.show()
  # noload knee actuator energy
  plot_dic = {'data_1':bi_noload_knee_energy,'data_2':mono_noload_knee_energy, 'ylabel':'Energy (W/Kg)',
              'color_1':mycolors['crimson red'], 'color_2':mycolors['dark purple']}

  fig = plt.figure(num='Test: Noloaded: Knee Energy',figsize=(12.4, 10.8))
  utils.plot_pareto_comparison(plot_dic,loadcond='noload',compare='subjects')
  fig.tight_layout()
  fig.savefig('./Figures/Tests/Test_Noloaded_Knee_EnergyBarPlot_Subject.pdf',orientation='landscape',bbox_inches='tight')
  plt.show()

  # loaded metabolics energy
  plot_dic = {'data_1':bi_loaded_metabolics_energy,'data_2':mono_loaded_metabolics_energy, 'ylabel':'Energy (W/Kg)',
              'color_1':mycolors['crimson red'], 'color_2':mycolors['dark purple']}

  fig = plt.figure(num='Test: Loaded: Metabolic Energy',figsize=(12.4, 10.8)) 
  utils.plot_pareto_comparison(plot_dic,loadcond='loaded',compare='subjects')
  fig.tight_layout()
  fig.savefig('./Figures/Tests/Test_Loaded_Metabolic_energy_Subject.pdf',orientation='landscape',bbox_inches='tight')
  plt.show()

  # loaded metabolics energy
  plot_dic = {'data_1':bi_noload_metabolics_energy,'data_2':mono_noload_metabolics_energy, 'ylabel':'Energy (W/Kg)',
              'color_1':mycolors['crimson red'], 'color_2':mycolors['dark purple']}

  fig = plt.figure(num='Test: Noloaded: Metabolic Energy',figsize=(12.4, 10.8))
  utils.plot_pareto_comparison(plot_dic,loadcond='noload',compare='subjects')
  fig.tight_layout()
  fig.savefig('./Figures/Tests/Test_Noloaded_Metabolic_energy_Subject.pdf',orientation='landscape',bbox_inches='tight')
  plt.show()
  
#####################################################################################
a = input('Hip/Knee Actuators and Metabolics Energy Weight Comparisons (N, Y):')
print('\n')
if a.lower() == 'y':
  # loaded hip actuator energy
  plot_dic = {'data_1':bi_loaded_hip_energy,'data_2':mono_loaded_hip_energy, 'ylabel':'Energy (W/Kg)',
              'color_1':mycolors['crimson red'], 'color_2':mycolors['dark purple']}

  fig = plt.figure(num='Test: Loaded: Hip Energy',figsize=(12.4, 10.8))
  utils.plot_pareto_comparison(plot_dic,loadcond='loaded',compare='weights')
  fig.tight_layout()
  fig.savefig('./Figures/Tests/Test_Loaded_Hip_EnergyBarPlot_Weights.pdf',orientation='landscape',bbox_inches='tight')
  plt.show()
  # noload hip actuator energy
  plot_dic = {'data_1':bi_noload_hip_energy,'data_2':mono_noload_hip_energy, 'ylabel':'Energy (W/Kg)',
              'color_1':mycolors['crimson red'], 'color_2':mycolors['dark purple']}

  fig = plt.figure(num='Test: Noloaded: Hip Energy',figsize=(12.4, 10.8))
  utils.plot_pareto_comparison(plot_dic,loadcond='noload',compare='weights')
  fig.tight_layout()
  fig.savefig('./Figures/Tests/Test_Noloaded_Hip_EnergyBarPlot_Weights.pdf',orientation='landscape',bbox_inches='tight')
  plt.show()
  # loaded knee actuator energy
  plot_dic = {'data_1':bi_loaded_knee_energy,'data_2':mono_loaded_knee_energy, 'ylabel':'Energy (W/Kg)',
              'color_1':mycolors['crimson red'], 'color_2':mycolors['dark purple']}

  fig = plt.figure(num='Test: Loaded: Knee Energy',figsize=(12.4, 10.8))
  utils.plot_pareto_comparison(plot_dic,loadcond='loaded',compare='weights')
  fig.tight_layout()
  fig.savefig('./Figures/Tests/Test_Loaded_Knee_EnergyBarPlot_Weights.pdf',orientation='landscape',bbox_inches='tight')
  plt.show()
  # noload knee actuator energy
  plot_dic = {'data_1':bi_noload_knee_energy,'data_2':mono_noload_knee_energy, 'ylabel':'Energy (W/Kg)',
              'color_1':mycolors['crimson red'], 'color_2':mycolors['dark purple']}

  fig = plt.figure(num='Test: Noloaded: Knee Energy',figsize=(12.4, 10.8))
  utils.plot_pareto_comparison(plot_dic,loadcond='noload',compare='weights')
  fig.tight_layout()
  fig.savefig('./Figures/Tests/Test_Noloaded_Knee_EnergyBarPlot_Weights.pdf',orientation='landscape',bbox_inches='tight')
  plt.show()

  # loaded metabolics energy
  plot_dic = {'data_1':bi_loaded_metabolics_energy,'data_2':mono_loaded_metabolics_energy, 'ylabel':'Energy (W/Kg)',
              'color_1':mycolors['crimson red'], 'color_2':mycolors['dark purple']}

  fig = plt.figure(num='Test: Loaded: Metabolic Energy',figsize=(12.4, 10.8)) 
  utils.plot_pareto_comparison(plot_dic,loadcond='loaded',compare='weights')
  fig.tight_layout()
  fig.savefig('./Figures/Tests/Test_Loaded_Metabolic_energy_Weights.pdf',orientation='landscape',bbox_inches='tight')
  plt.show()

  # loaded metabolics energy
  plot_dic = {'data_1':bi_noload_metabolics_energy,'data_2':mono_noload_metabolics_energy, 'ylabel':'Energy (W/Kg)',
              'color_1':mycolors['crimson red'], 'color_2':mycolors['dark purple']}

  fig = plt.figure(num='Test: Noloaded: Metabolic Energy',figsize=(12.4, 10.8))
  utils.plot_pareto_comparison(plot_dic,loadcond='noload',compare='weights')
  fig.tight_layout()
  fig.savefig('./Figures/Tests/Test_Noloaded_Metabolic_energy_Weights.pdf',orientation='landscape',bbox_inches='tight')
  plt.show()

