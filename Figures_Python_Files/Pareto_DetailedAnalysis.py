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
# unassist energy dataset
directory = 'D:/Ali.K.M.Bonab/Walking_Mass_Inertia_Effect/Data/Data/Unassist/*_energy.csv'
files = enumerate(glob.iglob(directory), 1)
unassisted_energy_dataset = {pathlib.PurePath(f[1]).stem: np.loadtxt(f[1], delimiter=',') for f in files}
# pareto exo energy dataset
directory = 'D:/Ali.K.M.Bonab/Walking_Mass_Inertia_Effect/Data/Data/Pareto/*_energy.csv'
files = enumerate(glob.iglob(directory), 1)
assisted_energy_dataset = {pathlib.PurePath(f[1]).stem: np.loadtxt(f[1], delimiter=',') for f in files}
# gls
gl_noload = {'noload_subject{}_trial{}'.format(i,j): utils.construct_gl_mass_side(subjectno=i,trialno=j,loadcond='noload') for i in subjects for j in trials_num}
gl_loaded = {'loaded_subject{}_trial{}'.format(i,j): utils.construct_gl_mass_side(subjectno=i,trialno=j,loadcond='loaded') for i in subjects for j in trials_num}
#####################################################################################
# Processing Data
# toe-off
noload_mean_toe_off,_,loaded_mean_toe_off,_ = utils.toe_off_avg_std(gl_noload,gl_loaded)
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
bi_loaded_metabolics_energy = np.reshape(assisted_energy_dataset['biarticular_pareto_load_metabolics_energy'],(25,21),order='F')
mono_loaded_metabolics_energy = np.reshape(assisted_energy_dataset['monoarticular_pareto_load_metabolics_energy'],(25,21),order='F')
bi_noload_metabolics_energy = np.reshape(assisted_energy_dataset['biarticular_pareto_noload_metabolics_energy'],(25,21),order='F')
mono_noload_metabolics_energy = np.reshape(assisted_energy_dataset['monoarticular_pareto_noload_metabolics_energy'],(25,21),order='F')
# metabolics cost reduction percents
bi_loaded_metabolics_percent = utils.pareto_metabolics_reduction(assisted_energy_dataset['biarticular_pareto_load_metabolics_energy'],unassisted_energy_dataset['loaded_metabolics_energy'])
bi_noload_metabolics_percent = utils.pareto_metabolics_reduction(assisted_energy_dataset['biarticular_pareto_noload_metabolics_energy'],unassisted_energy_dataset['noload_metabolics_energy'])
mono_loaded_metabolics_percent = utils.pareto_metabolics_reduction(assisted_energy_dataset['monoarticular_pareto_load_metabolics_energy'],unassisted_energy_dataset['loaded_metabolics_energy'])
mono_noload_metabolics_percent = utils.pareto_metabolics_reduction(assisted_energy_dataset['monoarticular_pareto_noload_metabolics_energy'],unassisted_energy_dataset['noload_metabolics_energy'])
# actuators regenrative energy
bi_noload_hip_regen_energy = np.reshape(assisted_energy_dataset['biarticular_pareto_noload_hipregenrative_energy'],(25,21),order='F')
bi_noload_knee_regen_energy = np.reshape(assisted_energy_dataset['biarticular_pareto_noload_kneeregenrative_energy'],(25,21),order='F')
mono_noload_hip_regen_energy = np.reshape(assisted_energy_dataset['monoarticular_pareto_noload_hipregenrative_energy'],(25,21),order='F')
mono_noload_knee_regen_energy = np.reshape(assisted_energy_dataset['monoarticular_pareto_noload_kneeregenrative_energy'],(25,21),order='F')

bi_loaded_hip_regen_energy = np.reshape(assisted_energy_dataset['biarticular_pareto_load_hipregenrative_energy'],(25,21),order='F')
bi_loaded_knee_regen_energy = np.reshape(assisted_energy_dataset['biarticular_pareto_load_kneeregenrative_energy'],(25,21),order='F')
mono_loaded_hip_regen_energy = np.reshape(assisted_energy_dataset['monoarticular_pareto_load_hipregenrative_energy'],(25,21),order='F')
mono_loaded_knee_regen_energy = np.reshape(assisted_energy_dataset['monoarticular_pareto_load_kneeregenrative_energy'],(25,21),order='F')

# actuators energy with regenration energy
bi_noload_hip_withregen_energy = -bi_noload_hip_regen_energy + bi_noload_hip_energy
bi_noload_knee_withregen_energy = -bi_noload_knee_regen_energy + bi_noload_hip_energy
mono_noload_hip_withregen_energy = -mono_noload_hip_regen_energy + mono_noload_hip_energy
mono_noload_knee_withregen_energy = -mono_noload_knee_regen_energy + mono_noload_knee_energy

bi_loaded_hip_withregen_energy = -bi_loaded_hip_regen_energy + bi_loaded_hip_energy
bi_loaded_knee_withregen_energy = -bi_loaded_knee_regen_energy + bi_loaded_hip_energy
mono_loaded_hip_withregen_energy = -mono_loaded_hip_regen_energy + mono_loaded_hip_energy
mono_loaded_knee_withregen_energy = -mono_loaded_knee_regen_energy + mono_loaded_knee_energy

#####################################################################################
a = input('Hip/Knee Actuators and Metabolics Energy Subject Comparisons (N, Y):')
print('\n')
if a.lower() == 'y':
  # loaded hip actuator energy
  plot_dic = {'data_1':bi_loaded_hip_energy,'data_2':mono_loaded_hip_energy, 'ylabel':'Energy (W/Kg)',
              'color_1':mycolors['crimson red'], 'color_2':mycolors['dark purple']}

  fig = plt.figure(num='Loaded: Hip Energy',figsize=(12.4, 18.8))
  utils.plot_pareto_comparison(plot_dic,loadcond='loaded',compare='subjects')
  fig.tight_layout()
  fig.savefig('./Figures/Pareto/Detailed_Analysis/Loaded_Hip_EnergyBarPlot_Subject.pdf',orientation='landscape',bbox_inches='tight')
  plt.show()
  # noload hip actuator energy
  plot_dic = {'data_1':bi_noload_hip_energy,'data_2':mono_noload_hip_energy, 'ylabel':'Energy (W/Kg)',
              'color_1':mycolors['crimson red'], 'color_2':mycolors['dark purple']}

  fig = plt.figure(num='Noloaded: Hip Energy',figsize=(12.4, 18.8))
  utils.plot_pareto_comparison(plot_dic,loadcond='noload',compare='subjects')
  fig.tight_layout()
  fig.savefig('./Figures/Pareto/Detailed_Analysis/Noloaded_Hip_EnergyBarPlot_Subject.pdf',orientation='landscape',bbox_inches='tight')
  plt.show()
  # loaded knee actuator energy
  plot_dic = {'data_1':bi_loaded_knee_energy,'data_2':mono_loaded_knee_energy, 'ylabel':'Energy (W/Kg)',
              'color_1':mycolors['crimson red'], 'color_2':mycolors['dark purple']}

  fig = plt.figure(num='Loaded: Knee Energy',figsize=(12.4, 18.8))
  utils.plot_pareto_comparison(plot_dic,loadcond='loaded',compare='subjects')
  fig.tight_layout()
  fig.savefig('./Figures/Pareto/Detailed_Analysis/Loaded_Knee_EnergyBarPlot_Subject.pdf',orientation='landscape',bbox_inches='tight')
  plt.show()

  # noload knee actuator energy
  plot_dic = {'data_1':bi_noload_knee_energy,'data_2':mono_noload_knee_energy, 'ylabel':'Energy (W/Kg)',
              'color_1':mycolors['crimson red'], 'color_2':mycolors['dark purple']}

  fig = plt.figure(num='Noloaded: Knee Energy',figsize=(12.4, 18.8))
  utils.plot_pareto_comparison(plot_dic,loadcond='noload',compare='subjects')
  fig.tight_layout()
  fig.savefig('./Figures/Pareto/Detailed_Analysis/Noloaded_Knee_EnergyBarPlot_Subject.pdf',orientation='landscape',bbox_inches='tight')
  plt.show()

  # loaded metabolics energy
  plot_dic = {'data_1':bi_loaded_metabolics_percent,'data_2':mono_loaded_metabolics_percent, 'ylabel':'Energy (%)',
              'color_1':mycolors['crimson red'], 'color_2':mycolors['dark purple']}

  fig = plt.figure(num='Loaded: Metabolic Energy',figsize=(12.4, 18.8)) 
  utils.plot_pareto_comparison(plot_dic,loadcond='loaded',compare='subjects')
  fig.tight_layout()
  fig.savefig('./Figures/Pareto/Detailed_Analysis/Loaded_Metabolic_energy_Subject.pdf',orientation='landscape',bbox_inches='tight')
  plt.show()

  # noload metabolics energy
  plot_dic = {'data_1':bi_noload_metabolics_percent,'data_2':mono_noload_metabolics_percent, 'ylabel':'Energy (%)',
              'color_1':mycolors['crimson red'], 'color_2':mycolors['dark purple']}

  fig = plt.figure(num='Noloaded: Metabolic Energy',figsize=(12.4, 18.8))
  utils.plot_pareto_comparison(plot_dic,loadcond='noload',compare='subjects')
  fig.tight_layout()
  fig.savefig('./Figures/Pareto/Detailed_Analysis/Noloaded_Metabolic_energy_Subject.pdf',orientation='landscape',bbox_inches='tight')
  plt.show()
  #**********************************************
  # regenerated energies mono vs bi
  # loaded hip actuator energy
  plot_dic = {'data_1':bi_loaded_hip_withregen_energy,'data_2':mono_loaded_hip_withregen_energy, 'ylabel':'Energy (W/Kg)',
              'color_1':mycolors['crimson red'], 'color_2':mycolors['dark purple']}

  fig = plt.figure(num='Loaded: Hip Energy',figsize=(12.4, 18.8))
  utils.plot_pareto_comparison(plot_dic,loadcond='loaded',compare='subjects')
  fig.tight_layout()
  fig.savefig('./Figures/Pareto/Detailed_Analysis/Loaded_Hip_Regenerated_EnergyBarPlot_Subject.pdf',orientation='landscape',bbox_inches='tight')
  plt.show()
  # noload hip actuator energy
  plot_dic = {'data_1':bi_noload_hip_withregen_energy,'data_2':mono_noload_hip_withregen_energy, 'ylabel':'Energy (W/Kg)',
              'color_1':mycolors['crimson red'], 'color_2':mycolors['dark purple']}

  fig = plt.figure(num='Noloaded: Hip Energy',figsize=(12.4, 18.8))
  utils.plot_pareto_comparison(plot_dic,loadcond='noload',compare='subjects')
  fig.tight_layout()
  fig.savefig('./Figures/Pareto/Detailed_Analysis/Noloaded_Hip_Regenerated_EnergyBarPlot_Subject.pdf',orientation='landscape',bbox_inches='tight')
  plt.show()
  # loaded knee actuator energy
  plot_dic = {'data_1':bi_loaded_knee_withregen_energy,'data_2':mono_loaded_knee_withregen_energy, 'ylabel':'Energy (W/Kg)',
              'color_1':mycolors['crimson red'], 'color_2':mycolors['dark purple']}

  fig = plt.figure(num='Loaded: Knee Energy',figsize=(12.4, 18.8))
  utils.plot_pareto_comparison(plot_dic,loadcond='loaded',compare='subjects')
  fig.tight_layout()
  fig.savefig('./Figures/Pareto/Detailed_Analysis/Loaded_Knee_Regenerated_EnergyBarPlot_Subject.pdf',orientation='landscape',bbox_inches='tight')
  plt.show()

  # noload knee actuator energy
  plot_dic = {'data_1':bi_noload_knee_withregen_energy,'data_2':mono_noload_knee_withregen_energy, 'ylabel':'Energy (W/Kg)',
              'color_1':mycolors['crimson red'], 'color_2':mycolors['dark purple']}

  fig = plt.figure(num='Noloaded: Knee Energy',figsize=(12.4, 18.8))
  utils.plot_pareto_comparison(plot_dic,loadcond='noload',compare='subjects')
  fig.tight_layout()
  fig.savefig('./Figures/Pareto/Detailed_Analysis/Noloaded_Knee_Regenerated_EnergyBarPlot_Subject.pdf',orientation='landscape',bbox_inches='tight')
  plt.show()
  #**********************************************
  # regeneration energy
  # loaded hip actuator energy
  plot_dic = {'data_1':bi_loaded_hip_regen_energy,'data_2':mono_loaded_hip_regen_energy, 'ylabel':'Energy (W/Kg)',
              'color_1':mycolors['crimson red'], 'color_2':mycolors['dark purple']}

  fig = plt.figure(num='Loaded: Hip Energy',figsize=(12.4, 18.8))
  utils.plot_pareto_comparison(plot_dic,loadcond='loaded',compare='subjects')
  fig.tight_layout()
  fig.savefig('./Figures/Pareto/Detailed_Analysis/Loaded_Hip_Regenertable_Energy_Subject.pdf',orientation='landscape',bbox_inches='tight')
  plt.show()

  # noload hip actuator energy
  plot_dic = {'data_1':bi_noload_hip_regen_energy,'data_2':mono_noload_hip_regen_energy, 'ylabel':'Energy (W/Kg)',
              'color_1':mycolors['crimson red'], 'color_2':mycolors['dark purple']}

  fig = plt.figure(num='Noloaded: Hip Energy',figsize=(12.4, 18.8))
  utils.plot_pareto_comparison(plot_dic,loadcond='noload',compare='subjects')
  fig.tight_layout()
  fig.savefig('./Figures/Pareto/Detailed_Analysis/Noloaded_Hip_Regenertable_Energy_Subject.pdf',orientation='landscape',bbox_inches='tight')
  plt.show()

  # loaded knee actuator energy
  plot_dic = {'data_1':bi_loaded_knee_regen_energy,'data_2':mono_loaded_knee_regen_energy, 'ylabel':'Energy (W/Kg)',
              'color_1':mycolors['crimson red'], 'color_2':mycolors['dark purple']}

  fig = plt.figure(num='Loaded: Knee Energy',figsize=(12.4, 18.8))
  utils.plot_pareto_comparison(plot_dic,loadcond='loaded',compare='subjects')
  fig.tight_layout()
  fig.savefig('./Figures/Pareto/Detailed_Analysis/Loaded_Knee_Regenertable_Energy_Subject.pdf',orientation='landscape',bbox_inches='tight')
  plt.show()

  # noload knee actuator energy
  plot_dic = {'data_1':bi_noload_knee_regen_energy,'data_2':mono_noload_knee_regen_energy, 'ylabel':'Energy (W/Kg)',
              'color_1':mycolors['crimson red'], 'color_2':mycolors['dark purple']}

  fig = plt.figure(num='Noloaded: Knee Energy',figsize=(12.4, 18.8))
  utils.plot_pareto_comparison(plot_dic,loadcond='noload',compare='subjects')
  fig.tight_layout()
  fig.savefig('./Figures/Pareto/Detailed_Analysis/Noloaded_Knee_Regenertable_Energy_Subject.pdf',orientation='landscape',bbox_inches='tight')
  plt.show()
  #**********************************************
  # loaded regenerated vs non-regenerated
  # loaded bi hip actuator energy
  plot_dic = {'data_1':bi_loaded_hip_withregen_energy,'data_2':bi_loaded_hip_energy, 'ylabel':'Energy (W/Kg)',
              'color_2':mycolors['crimson red'], 'color_1':mycolors['gold'],
              'legend_1':'bi regenerated','legend_2':'bi non-regenerated'}

  fig = plt.figure(num='Loaded: Hip Energy',figsize=(12.4, 18.8))
  utils.plot_pareto_comparison(plot_dic,loadcond='loaded',compare='subjects')
  fig.tight_layout()
  fig.savefig('./Figures/Pareto/Detailed_Analysis/Loaded_Hip_Biarticular_Rengenration_Subject.pdf',orientation='landscape',bbox_inches='tight')
  plt.show()
  # loaded mono hip actuator energy
  plot_dic = {'data_1':mono_loaded_hip_withregen_energy,'data_2':mono_loaded_hip_energy, 'ylabel':'Energy (W/Kg)',
              'color_2':mycolors['royal blue'], 'color_1':mycolors['lavender purple'],
              'legend_1':'mono regenerated','legend_2':'mono non-regenerated'}

  fig = plt.figure(num='Loaded: Hip Energy',figsize=(12.4, 18.8))
  utils.plot_pareto_comparison(plot_dic,loadcond='loaded',compare='subjects')
  fig.tight_layout()
  fig.savefig('./Figures/Pareto/Detailed_Analysis/Loaded_Hip_Monoarticular_Rengenration_Subject.pdf',orientation='landscape',bbox_inches='tight')
  plt.show()
  # loaded bi knee actuator energy
  plot_dic = {'data_1':bi_loaded_knee_withregen_energy,'data_2':bi_loaded_knee_energy, 'ylabel':'Energy (W/Kg)',
              'color_2':mycolors['crimson red'], 'color_1':mycolors['gold'],
              'legend_1':'bi regenerated','legend_2':'bi non-regenerated'}

  fig = plt.figure(num='Loaded: Knee Energy',figsize=(12.4, 18.8))
  utils.plot_pareto_comparison(plot_dic,loadcond='loaded',compare='subjects')
  fig.tight_layout()
  fig.savefig('./Figures/Pareto/Detailed_Analysis/Loaded_Knee_Biarticular_Rengenration_Subject.pdf',orientation='landscape',bbox_inches='tight')
  plt.show()
  # loaded mono knee actuator energy
  plot_dic = {'data_1':mono_loaded_knee_withregen_energy,'data_2':mono_loaded_knee_energy, 'ylabel':'Energy (W/Kg)',
              'color_2':mycolors['royal blue'], 'color_1':mycolors['lavender purple'],
              'legend_1':'mono regenerated','legend_2':'mono non-regenerated'}

  fig = plt.figure(num='Loaded: Knee Energy',figsize=(12.4, 18.8))
  utils.plot_pareto_comparison(plot_dic,loadcond='loaded',compare='subjects')
  fig.tight_layout()
  fig.savefig('./Figures/Pareto/Detailed_Analysis/Loaded_Knee_Monoarticular_Rengenration_Subject.pdf',orientation='landscape',bbox_inches='tight')
  plt.show()
  #**********************************************
  # noload regenerated vs non-regenerated
  # noload bi hip actuator energy
  plot_dic = {'data_1':bi_noload_hip_withregen_energy,'data_2':bi_noload_hip_energy, 'ylabel':'Energy (W/Kg)',
              'color_2':mycolors['crimson red'], 'color_1':mycolors['gold'],
              'legend_1':'bi regenerated','legend_2':'bi non-regenerated'}

  fig = plt.figure(num='Noloaded: Hip Energy',figsize=(12.4, 18.8))
  utils.plot_pareto_comparison(plot_dic,loadcond='noload',compare='subjects')
  fig.tight_layout()
  fig.savefig('./Figures/Pareto/Detailed_Analysis/Noload_Hip_Biarticular_Rengenration_Subject.pdf',orientation='landscape',bbox_inches='tight')
  plt.show()

  # noload mono hip actuator energy
  plot_dic = {'data_1':mono_noload_hip_withregen_energy,'data_2':mono_noload_hip_energy, 'ylabel':'Energy (W/Kg)',
              'color_2':mycolors['royal blue'], 'color_1':mycolors['lavender purple'],
              'legend_1':'mono regenerated','legend_2':'mono non-regenerated'}

  fig = plt.figure(num='Noloaded: Hip Energy',figsize=(12.4, 18.8))
  utils.plot_pareto_comparison(plot_dic,loadcond='noload',compare='subjects')
  fig.tight_layout()
  fig.savefig('./Figures/Pareto/Detailed_Analysis/Noload_Hip_Monoarticular_Rengenration_Subject.pdf',orientation='landscape',bbox_inches='tight')
  plt.show()

  # noload bi knee actuator energy
  plot_dic = {'data_1':bi_noload_knee_withregen_energy,'data_2':bi_noload_knee_energy, 'ylabel':'Energy (W/Kg)',
              'color_2':mycolors['crimson red'], 'color_1':mycolors['gold'],
              'legend_1':'bi regenerated','legend_2':'bi non-regenerated'}

  fig = plt.figure(num='Noloaded: Knee Energy',figsize=(12.4, 18.8))
  utils.plot_pareto_comparison(plot_dic,loadcond='noload',compare='subjects')
  fig.tight_layout()
  fig.savefig('./Figures/Pareto/Detailed_Analysis/Noload_Knee_Biarticular_Rengenration_Subject.pdf',orientation='landscape',bbox_inches='tight')
  plt.show()
  # noload mono knee actuator energy
  plot_dic = {'data_1':mono_noload_knee_withregen_energy,'data_2':mono_noload_knee_energy, 'ylabel':'Energy (W/Kg)',
              'color_2':mycolors['royal blue'], 'color_1':mycolors['lavender purple'],
              'legend_1':'mono regenerated','legend_2':'mono non-regenerated'}

  fig = plt.figure(num='Noloaded: Knee Energy',figsize=(12.4, 18.8))
  utils.plot_pareto_comparison(plot_dic,loadcond='noload',compare='subjects')
  fig.tight_layout()
  fig.savefig('./Figures/Pareto/Detailed_Analysis/Noload_Knee_Monoarticular_Rengenration_Subject.pdf',orientation='landscape',bbox_inches='tight')
  plt.show()
#####################################################################################
a = input('Hip/Knee Actuators and Metabolics Energy Weight Comparisons (N, Y):')
print('\n')
if a.lower() == 'y':
  # loaded hip actuator energy
  plot_dic = {'data_1':bi_loaded_hip_energy,'data_2':mono_loaded_hip_energy, 'ylabel':'Energy (W/Kg)',
              'color_1':mycolors['crimson red'], 'color_2':mycolors['dark purple']}

  fig = plt.figure(num='Loaded: Hip Energy',figsize=(12.4, 10.8))
  utils.plot_pareto_comparison(plot_dic,loadcond='loaded',compare='weights')
  fig.tight_layout()
  fig.savefig('./Figures/Pareto/Detailed_Analysis/Loaded_Hip_EnergyBarPlot_Weights.pdf',orientation='landscape',bbox_inches='tight')
  plt.show()
  # noload hip actuator energy
  plot_dic = {'data_1':bi_noload_hip_energy,'data_2':mono_noload_hip_energy, 'ylabel':'Energy (W/Kg)',
              'color_1':mycolors['crimson red'], 'color_2':mycolors['dark purple']}

  fig = plt.figure(num='Noloaded: Hip Energy',figsize=(12.4, 10.8))
  utils.plot_pareto_comparison(plot_dic,loadcond='noload',compare='weights')
  fig.tight_layout()
  fig.savefig('./Figures/Pareto/Detailed_Analysis/Noloaded_Hip_EnergyBarPlot_Weights.pdf',orientation='landscape',bbox_inches='tight')
  plt.show()
  # loaded knee actuator energy
  plot_dic = {'data_1':bi_loaded_knee_energy,'data_2':mono_loaded_knee_energy, 'ylabel':'Energy (W/Kg)',
              'color_1':mycolors['crimson red'], 'color_2':mycolors['dark purple']}

  fig = plt.figure(num='Loaded: Knee Energy',figsize=(12.4, 10.8))
  utils.plot_pareto_comparison(plot_dic,loadcond='loaded',compare='weights')
  fig.tight_layout()
  fig.savefig('./Figures/Pareto/Detailed_Analysis/Loaded_Knee_EnergyBarPlot_Weights.pdf',orientation='landscape',bbox_inches='tight')
  plt.show()
  # noload knee actuator energy
  plot_dic = {'data_1':bi_noload_knee_energy,'data_2':mono_noload_knee_energy, 'ylabel':'Energy (W/Kg)',
              'color_1':mycolors['crimson red'], 'color_2':mycolors['dark purple']}

  fig = plt.figure(num='Noloaded: Knee Energy',figsize=(12.4, 10.8))
  utils.plot_pareto_comparison(plot_dic,loadcond='noload',compare='weights')
  fig.tight_layout()
  fig.savefig('./Figures/Pareto/Detailed_Analysis/Noloaded_Knee_EnergyBarPlot_Weights.pdf',orientation='landscape',bbox_inches='tight')
  plt.show()

  # loaded metabolics energy
  plot_dic = {'data_1':bi_loaded_metabolics_percent,'data_2':mono_loaded_metabolics_percent, 'ylabel':'Energy (%)',
              'color_1':mycolors['crimson red'], 'color_2':mycolors['dark purple']}

  fig = plt.figure(num='Loaded: Metabolic Energy',figsize=(12.4, 10.8)) 
  utils.plot_pareto_comparison(plot_dic,loadcond='loaded',compare='weights')
  fig.tight_layout()
  fig.savefig('./Figures/Pareto/Detailed_Analysis/Loaded_Metabolic_energy_Weights.pdf',orientation='landscape',bbox_inches='tight')
  plt.show()

  # noload metabolics energy
  plot_dic = {'data_1':bi_noload_metabolics_percent,'data_2':mono_noload_metabolics_percent, 'ylabel':'Energy (%)',
              'color_1':mycolors['crimson red'], 'color_2':mycolors['dark purple']}

  fig = plt.figure(num='Noloaded: Metabolic Energy',figsize=(12.4, 10.8))
  utils.plot_pareto_comparison(plot_dic,loadcond='noload',compare='weights')
  fig.tight_layout()
  fig.savefig('./Figures/Pareto/Detailed_Analysis/Noloaded_Metabolic_energy_Weights.pdf',orientation='landscape',bbox_inches='tight')
  plt.show()
  #**********************************************
  # regeneration energy
  # loaded hip actuator energy
  plot_dic = {'data_1':bi_loaded_hip_regen_energy,'data_2':mono_loaded_hip_regen_energy, 'ylabel':'Energy (W/Kg)',
              'color_1':mycolors['crimson red'], 'color_2':mycolors['dark purple']}

  fig = plt.figure(num='Loaded: Hip Energy',figsize=(12.4, 18.8))
  utils.plot_pareto_comparison(plot_dic,loadcond='loaded',compare='weights')
  fig.tight_layout()
  fig.savefig('./Figures/Pareto/Detailed_Analysis/Loaded_Hip_Regenertable_Energy_Weights.pdf',orientation='landscape',bbox_inches='tight')
  plt.show()
  
  # noload hip actuator energy
  plot_dic = {'data_1':bi_noload_hip_regen_energy,'data_2':mono_noload_hip_regen_energy, 'ylabel':'Energy (W/Kg)',
              'color_1':mycolors['crimson red'], 'color_2':mycolors['dark purple']}

  fig = plt.figure(num='Noloaded: Hip Energy',figsize=(12.4, 18.8))
  utils.plot_pareto_comparison(plot_dic,loadcond='noload',compare='weights')
  fig.tight_layout()
  fig.savefig('./Figures/Pareto/Detailed_Analysis/Noloaded_Hip_Regenertable_Energy_Weights.pdf',orientation='landscape',bbox_inches='tight')
  plt.show()

  # loaded knee actuator energy
  plot_dic = {'data_1':bi_loaded_knee_regen_energy,'data_2':mono_loaded_knee_regen_energy, 'ylabel':'Energy (W/Kg)',
              'color_1':mycolors['crimson red'], 'color_2':mycolors['dark purple']}

  fig = plt.figure(num='Loaded: Knee Energy',figsize=(12.4, 18.8))
  utils.plot_pareto_comparison(plot_dic,loadcond='loaded',compare='weights')
  fig.tight_layout()
  fig.savefig('./Figures/Pareto/Detailed_Analysis/Loaded_Knee_Regenertable_Energy_Weights.pdf',orientation='landscape',bbox_inches='tight')
  plt.show()

  # noload knee actuator energy
  plot_dic = {'data_1':bi_noload_knee_regen_energy,'data_2':mono_noload_knee_regen_energy, 'ylabel':'Energy (W/Kg)',
              'color_1':mycolors['crimson red'], 'color_2':mycolors['dark purple']}

  fig = plt.figure(num='Noloaded: Knee Energy',figsize=(12.4, 18.8))
  utils.plot_pareto_comparison(plot_dic,loadcond='noload',compare='weights')
  fig.tight_layout()
  fig.savefig('./Figures/Pareto/Detailed_Analysis/Noloaded_Knee_Regenertable_Energy_Weights.pdf',orientation='landscape',bbox_inches='tight')
  plt.show()
