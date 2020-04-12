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
# mean & std metabolics cost reduction percents
mean_bi_loaded_metabolics_percent, std_bi_loaded_metabolics_percent = utils.pareto_avg_std_energy(bi_loaded_metabolics_percent,reshape=False)
mean_bi_noload_metabolics_percent, std_bi_noload_metabolics_percent = utils.pareto_avg_std_energy(bi_noload_metabolics_percent,reshape=False)
mean_mono_loaded_metabolics_percent, std_mono_loaded_metabolics_percent = utils.pareto_avg_std_energy(mono_loaded_metabolics_percent,reshape=False)
mean_mono_noload_metabolics_percent, std_mono_noload_metabolics_percent = utils.pareto_avg_std_energy(mono_noload_metabolics_percent,reshape=False)
# mean & std metabolics cost reduction
mean_bi_loaded_metabolics, std_bi_loaded_metabolics = utils.pareto_avg_std_energy(assisted_energy_dataset['biarticular_pareto_load_metabolics_energy'],reshape=True)
mean_bi_noload_metabolics, std_bi_noload_metabolics = utils.pareto_avg_std_energy(assisted_energy_dataset['biarticular_pareto_noload_metabolics_energy'],reshape=True)
mean_mono_loaded_metabolics, std_mono_loaded_metabolics = utils.pareto_avg_std_energy(assisted_energy_dataset['monoarticular_pareto_load_metabolics_energy'],reshape=True)
mean_mono_noload_metabolics, std_mono_noload_metabolics = utils.pareto_avg_std_energy(assisted_energy_dataset['monoarticular_pareto_noload_metabolics_energy'],reshape=True)
# mean & std actuators energy
# loaded bi
mean_bi_loaded_hip_energy, std_bi_loaded_hip_energy = utils.pareto_avg_std_energy(assisted_energy_dataset['biarticular_pareto_load_hipactuator_energy'],reshape=True)
mean_bi_loaded_knee_energy, std_bi_loaded_knee_energy = utils.pareto_avg_std_energy(assisted_energy_dataset['biarticular_pareto_load_kneeactuator_energy'],reshape=True)
mean_bi_loaded_energy, std_bi_loaded_energy = utils.pareto_avg_std_energy(assisted_energy_dataset['biarticular_pareto_load_kneeactuator_energy']+assisted_energy_dataset['biarticular_pareto_load_hipactuator_energy'],reshape=True)
# noload bi
mean_bi_noload_hip_energy, std_bi_noload_hip_energy = utils.pareto_avg_std_energy(assisted_energy_dataset['biarticular_pareto_noload_hipactuator_energy'],reshape=True)
mean_bi_noload_knee_energy, std_bi_noload_knee_energy = utils.pareto_avg_std_energy(assisted_energy_dataset['biarticular_pareto_noload_kneeactuator_energy'],reshape=True)
mean_bi_noload_energy, std_bi_noload_energy = utils.pareto_avg_std_energy(assisted_energy_dataset['biarticular_pareto_noload_kneeactuator_energy']+assisted_energy_dataset['biarticular_pareto_noload_hipactuator_energy'],reshape=True)
# loaded mono
mean_mono_loaded_hip_energy, std_mono_loaded_hip_energy = utils.pareto_avg_std_energy(assisted_energy_dataset['monoarticular_pareto_load_hipactuator_energy'],reshape=True)
mean_mono_loaded_knee_energy, std_mono_loaded_knee_energy = utils.pareto_avg_std_energy(assisted_energy_dataset['monoarticular_pareto_load_kneeactuator_energy'],reshape=True)
mean_mono_loaded_energy, std_mono_loaded_energy = utils.pareto_avg_std_energy(assisted_energy_dataset['monoarticular_pareto_load_kneeactuator_energy']+assisted_energy_dataset['monoarticular_pareto_load_hipactuator_energy'],reshape=True)
# noload mono
mean_mono_noload_hip_energy, std_mono_noload_hip_energy = utils.pareto_avg_std_energy(assisted_energy_dataset['monoarticular_pareto_noload_hipactuator_energy'],reshape=True)
mean_mono_noload_knee_energy, std_mono_noload_knee_energy = utils.pareto_avg_std_energy(assisted_energy_dataset['monoarticular_pareto_noload_kneeactuator_energy'],reshape=True)
mean_mono_noload_energy, std_mono_noload_energy = utils.pareto_avg_std_energy(assisted_energy_dataset['monoarticular_pareto_noload_kneeactuator_energy']+assisted_energy_dataset['monoarticular_pareto_noload_hipactuator_energy'],reshape=True)
#####################################################################################
# Paretofront data
# mean & std metabolics cost reduction percents
bi_loaded_indices = np.array([])
mean_bi_loaded_paretofront = utils.manual_paretofront(mean_bi_loaded_metabolics_percent,mean_bi_loaded_energy)
std_bi_loaded_paretofront = utils.manual_paretofront(std_bi_loaded_metabolics_percent,std_bi_loaded_energy)


#####################################################################################
# plots
# average pareto curve: loaded mono vs biarticular

plot_dic = {'x1_data':mean_bi_loaded_metabolics_percent,'x1err_data':std_bi_loaded_metabolics_percent,
          'x2_data':mean_mono_loaded_metabolics_percent,'x2err_data':std_mono_loaded_metabolics_percent,
          'y1_data':mean_bi_loaded_energy,'y1err_data':std_bi_loaded_energy,
          'y2_data':mean_mono_loaded_energy,'y2err_data':std_mono_loaded_energy,
          'color_1':mycolors['crimson red'],'color_2':mycolors['dark purple']
          }
fig = plt.figure(num='Pareto Curve: loaded mono vs bi',figsize=(10.4, 8.8))
utils.plot_pareto_avg_curve (plot_dic,loadcond='loaded')
plt.xlabel('Metabolic Cost Reduction (%)')
plt.ylabel('Exoskeleton Energy Consumption (W/kg)')
ax = plt.gca()
utils.no_top_right(ax)
plt.legend(loc='best',frameon=False)
fig.tight_layout()
fig.savefig('./Figures/Pareto/Mean_Pareto/Pareto_Load_BiVsMono.pdf',orientation='landscape',bbox_inches='tight')
plt.show()

# average pareto curve: noload mono vs biarticular

plot_dic = {'x1_data':mean_bi_noload_metabolics_percent,'x1err_data':std_bi_noload_metabolics_percent,
          'x2_data':mean_mono_noload_metabolics_percent,'x2err_data':std_mono_noload_metabolics_percent,
          'y1_data':mean_bi_noload_energy,'y1err_data':std_bi_noload_energy,
          'y2_data':mean_mono_noload_energy,'y2err_data':std_mono_noload_energy,
          'color_1':mycolors['magenta pink'],'color_2':mycolors['lavender purple']
          }
fig = plt.figure(num='Pareto Curve: noload mono vs bi',figsize=(10.4, 8.8))
utils.plot_pareto_avg_curve (plot_dic,loadcond='loaded')
plt.xlabel('Metabolic Cost Reduction (%)')
plt.ylabel('Exoskeleton Energy Consumption (W/kg)')
ax = plt.gca()
utils.no_top_right(ax)
plt.legend(loc='best',frameon=False)
fig.tight_layout()
fig.savefig('./Figures/Pareto/Mean_Pareto/Pareto_Noload_BiVsMono.pdf',orientation='landscape',bbox_inches='tight')
plt.show()

# average pareto curve: biarticular noload vs loaded

plot_dic = {'x1_data':mean_bi_loaded_metabolics_percent,'x1err_data':std_bi_loaded_metabolics_percent,
          'x2_data':mean_bi_noload_metabolics_percent,'x2err_data':std_bi_noload_metabolics_percent,
          'y1_data':mean_bi_loaded_energy,'y1err_data':std_bi_loaded_energy,
          'y2_data':mean_bi_noload_energy,'y2err_data':std_bi_noload_energy,
          'color_1':mycolors['crimson red'],'color_2':mycolors['olympic blue'],
          'legend_1':'loaded biarticular','legend_2':'noload biarticular'
          }
fig = plt.figure(num='Pareto Curve: biarticular loaded vs noload',figsize=(10.4, 8.8))
utils.plot_pareto_avg_curve (plot_dic,loadcond='loaded')
plt.xlabel('Metabolic Cost Reduction (%)')
plt.ylabel('Exoskeleton Energy Consumption (W/kg)')
ax = plt.gca()
utils.no_top_right(ax)
plt.legend(loc='best',frameon=False)
fig.tight_layout()
fig.savefig('./Figures/Pareto/Mean_Pareto/Pareto_Biarticular_LoadedvsNoload.pdf',orientation='landscape',bbox_inches='tight')
plt.show()

# average pareto curve: monoarticular noload vs loaded

plot_dic = {'x1_data':mean_mono_loaded_metabolics_percent,'x1err_data':std_mono_loaded_metabolics_percent,
          'x2_data':mean_mono_noload_metabolics_percent,'x2err_data':std_mono_noload_metabolics_percent,
          'y1_data':mean_mono_loaded_energy,'y1err_data':std_mono_loaded_energy,
          'y2_data':mean_mono_noload_energy,'y2err_data':std_mono_noload_energy,
          'color_1':mycolors['crimson red'],'color_2':mycolors['olympic blue'],
          'legend_1':'loaded monoarticular','legend_2':'noload monoarticular'
          }
fig = plt.figure(num='Pareto Curve: monoarticular loaded vs noload',figsize=(10.4, 8.8))
utils.plot_pareto_avg_curve (plot_dic,loadcond='loaded')
plt.xlabel('Metabolic Cost Reduction (%)')
plt.ylabel('Exoskeleton Energy Consumption (W/kg)')
ax = plt.gca()
utils.no_top_right(ax)
plt.legend(loc='best',frameon=False)
fig.tight_layout()
fig.savefig('./Figures/Pareto/Mean_Pareto/Pareto_Monoarticular_LoadedvsNoload.pdf',orientation='landscape',bbox_inches='tight')
plt.show()

#####################################################################################
# plots
# average pareto curve: loaded mono vs biarticular

plot_dic = {'x1_data':mean_bi_loaded_metabolics,'x1err_data':std_bi_loaded_metabolics,
          'x2_data':mean_mono_loaded_metabolics,'x2err_data':std_mono_loaded_metabolics,
          'y1_data':mean_bi_loaded_energy,'y1err_data':std_bi_loaded_energy,
          'y2_data':mean_mono_loaded_energy,'y2err_data':std_mono_loaded_energy,
          'color_1':mycolors['crimson red'],'color_2':mycolors['dark purple']
          }
fig = plt.figure(num='Pareto Curve: loaded mono vs bi',figsize=(10.4, 8.8))
utils.plot_pareto_avg_curve (plot_dic,loadcond='loaded')
plt.xlabel('Metabolic Cost (W/kg)')
plt.ylabel('Exoskeleton Energy Consumption (W/kg)')
ax = plt.gca()
utils.no_top_right(ax)
plt.legend(loc='best',frameon=False)
fig.tight_layout()
fig.savefig('./Figures/Pareto/Mean_Pareto/Pareto_Load_BiVsMono_Energy.pdf',orientation='landscape',bbox_inches='tight')
plt.show()

# average pareto curve: noload mono vs biarticular

plot_dic = {'x1_data':mean_bi_noload_metabolics,'x1err_data':std_bi_noload_metabolics,
          'x2_data':mean_mono_noload_metabolics,'x2err_data':std_mono_noload_metabolics,
          'y1_data':mean_bi_noload_energy,'y1err_data':std_bi_noload_energy,
          'y2_data':mean_mono_noload_energy,'y2err_data':std_mono_noload_energy,
          'color_1':mycolors['magenta pink'],'color_2':mycolors['lavender purple']
          }
fig = plt.figure(num='Pareto Curve: noload mono vs bi',figsize=(10.4, 8.8))
utils.plot_pareto_avg_curve (plot_dic,loadcond='loaded')
plt.xlabel('Metabolic Cost  (W/kg)')
plt.ylabel('Exoskeleton Energy Consumption (W/kg)')
ax = plt.gca()
utils.no_top_right(ax)
plt.legend(loc='best',frameon=False)
fig.tight_layout()
fig.savefig('./Figures/Pareto/Mean_Pareto/Pareto_Noload_BiVsMono_Energy.pdf',orientation='landscape',bbox_inches='tight')
plt.show()

# average pareto curve: biarticular noload vs loaded

plot_dic = {'x1_data':mean_bi_loaded_metabolics,'x1err_data':std_bi_loaded_metabolics,
          'x2_data':mean_bi_noload_metabolics,'x2err_data':std_bi_noload_metabolics,
          'y1_data':mean_bi_loaded_energy,'y1err_data':std_bi_loaded_energy,
          'y2_data':mean_bi_noload_energy,'y2err_data':std_bi_noload_energy,
          'color_1':mycolors['crimson red'],'color_2':mycolors['olympic blue'],
          'legend_1':'loaded biarticular','legend_2':'noload biarticular'
          }
fig = plt.figure(num='Pareto Curve: biarticular loaded vs noload',figsize=(10.4, 8.8))
utils.plot_pareto_avg_curve (plot_dic,loadcond='loaded')
plt.xlabel('Metabolic Cost (W/kg)')
plt.ylabel('Exoskeleton Energy Consumption (W/kg)')
ax = plt.gca()
utils.no_top_right(ax)
plt.legend(loc='best',frameon=False)
fig.tight_layout()
fig.savefig('./Figures/Pareto/Mean_Pareto/Pareto_Biarticular_LoadedvsNoload_Energy.pdf',orientation='landscape',bbox_inches='tight')
plt.show()

# average pareto curve: monoarticular noload vs loaded

plot_dic = {'x1_data':mean_mono_loaded_metabolics,'x1err_data':std_mono_loaded_metabolics,
          'x2_data':mean_mono_noload_metabolics,'x2err_data':std_mono_noload_metabolics,
          'y1_data':mean_mono_loaded_energy,'y1err_data':std_mono_loaded_energy,
          'y2_data':mean_mono_noload_energy,'y2err_data':std_mono_noload_energy,
          'color_1':mycolors['crimson red'],'color_2':mycolors['olympic blue'],
          'legend_1':'loaded monoarticular','legend_2':'noload monoarticular'
          }
fig = plt.figure(num='Pareto Curve : monoarticular loaded vs noload',figsize=(10.4, 8.8))
utils.plot_pareto_avg_curve (plot_dic,loadcond='loaded')
plt.xlabel('Metabolic Cost (W/kg)')
plt.ylabel('Exoskeleton Energy Consumption (W/kg)')
ax = plt.gca()
utils.no_top_right(ax)
plt.legend(loc='best',frameon=False)
fig.tight_layout()
fig.savefig('./Figures/Pareto/Mean_Pareto/Pareto_Monoarticular_LoadedvsNoload_Energy.pdf',orientation='landscape',bbox_inches='tight')
plt.show()

