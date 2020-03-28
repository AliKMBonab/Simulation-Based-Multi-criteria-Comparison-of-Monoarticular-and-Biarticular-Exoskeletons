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
# metabolic cost
bi_noload_metabolics = np.reshape(assisted_energy_dataset['biarticular_pareto_noload_metabolics_energy'],(25,21),order='F')
mono_noload_metabolics = np.reshape(assisted_energy_dataset['monoarticular_pareto_noload_metabolics_energy'],(25,21),order='F')
# metabolics cost reduction percents
bi_noload_metabolics_percent = utils.pareto_metabolics_reduction(assisted_energy_dataset['biarticular_pareto_noload_metabolics_energy'],unassisted_energy_dataset['noload_metabolics_energy'])
mono_noload_metabolics_percent = utils.pareto_metabolics_reduction(assisted_energy_dataset['monoarticular_pareto_noload_metabolics_energy'],unassisted_energy_dataset['noload_metabolics_energy'])
# mean & std metabolics cost reduction percents
mean_bi_noload_metabolics_percent, std_bi_noload_metabolics_percent = utils.pareto_avg_std_energy(bi_noload_metabolics_percent,reshape=False)
mean_mono_noload_metabolics_percent, std_mono_noload_metabolics_percent = utils.pareto_avg_std_energy(mono_noload_metabolics_percent,reshape=False)
# mean & std metabolics cost
mean_bi_noload_metabolics, std_bi_noload_metabolics = utils.pareto_avg_std_energy(bi_noload_metabolics,reshape=False)
mean_mono_noload_metabolics, std_mono_noload_metabolics = utils.pareto_avg_std_energy(mono_noload_metabolics,reshape=False)
# actuators energy
bi_noload_hip_energy= np.reshape(assisted_energy_dataset['biarticular_pareto_noload_hipactuator_energy'],(25,21),order='F')
bi_noload_knee_energy= np.reshape(assisted_energy_dataset['biarticular_pareto_noload_kneeactuator_energy'],(25,21),order='F')
bi_noload_energy = bi_noload_hip_energy + bi_noload_knee_energy
mono_noload_hip_energy= np.reshape(assisted_energy_dataset['monoarticular_pareto_noload_hipactuator_energy'],(25,21),order='F')
mono_noload_knee_energy= np.reshape(assisted_energy_dataset['monoarticular_pareto_noload_kneeactuator_energy'],(25,21),order='F')
mono_noload_energy = mono_noload_hip_energy + mono_noload_knee_energy
# mean & std actuators energy
# noload bi
mean_bi_noload_hip_energy, std_bi_noload_hip_energy = utils.pareto_avg_std_energy(assisted_energy_dataset['biarticular_pareto_noload_hipactuator_energy'],reshape=True)
mean_bi_noload_knee_energy, std_bi_noload_knee_energy = utils.pareto_avg_std_energy(assisted_energy_dataset['biarticular_pareto_noload_kneeactuator_energy'],reshape=True)
mean_bi_noload_energy, std_bi_noload_energy = utils.pareto_avg_std_energy(assisted_energy_dataset['biarticular_pareto_noload_kneeactuator_energy']+assisted_energy_dataset['biarticular_pareto_noload_hipactuator_energy'],reshape=True)
# noload mono
mean_mono_noload_hip_energy, std_mono_noload_hip_energy = utils.pareto_avg_std_energy(assisted_energy_dataset['monoarticular_pareto_noload_hipactuator_energy'],reshape=True)
mean_mono_noload_knee_energy, std_mono_noload_knee_energy = utils.pareto_avg_std_energy(assisted_energy_dataset['monoarticular_pareto_noload_kneeactuator_energy'],reshape=True)
mean_mono_noload_energy, std_mono_noload_energy = utils.pareto_avg_std_energy(assisted_energy_dataset['monoarticular_pareto_noload_kneeactuator_energy']+assisted_energy_dataset['monoarticular_pareto_noload_hipactuator_energy'],reshape=True)
#####################################################################################
# Processing Data For Adding Mass
biarticular_exoskeleton_dic = {'m_waist':6, 'm_thigh':1, 'm_shank':0.9, 'motor_max_torque':2, 'motor_inertia':0.000506, 'thigh_com':0.23, 'shank_com':0.18, 'leg_inertia':2.52}
monoarticular_exoskeleton_dic = {'m_waist':3, 'm_thigh':4, 'm_shank':0.9, 'motor_max_torque':2, 'motor_inertia':0.000506, 'thigh_com':0.3, 'shank_com':0.18, 'leg_inertia':2.52}
biarticular_out = utils.addingmass_metabolics_pareto(unassisted_energy_dataset['noload_metabolics_energy'],bi_noload_metabolics,biarticular_exoskeleton_dic)
monoarticular_out = utils.addingmass_metabolics_pareto(unassisted_energy_dataset['noload_metabolics_energy'],mono_noload_metabolics,monoarticular_exoskeleton_dic)
# Metabolic cost reduction after adding mass
a = unassisted_energy_dataset['noload_metabolics_energy'][np.newaxis][0,:]
noload_metabolics_energy_biarticular_mass_added = np.tile(unassisted_energy_dataset['noload_metabolics_energy'][np.newaxis][0,:],(25,1))  + biarticular_out[4]
noload_metabolics_energy_monoarticular_mass_added = np.tile(unassisted_energy_dataset['noload_metabolics_energy'][np.newaxis][0,:],(25,1)) + monoarticular_out[4]
bi_noload_metabolics_addedmass_percent = utils.addingmass_metabolics_reduction(biarticular_out[0],noload_metabolics_energy_biarticular_mass_added)
mono_noload_metabolics_addedmass_percent = utils.addingmass_metabolics_reduction(monoarticular_out[0],noload_metabolics_energy_monoarticular_mass_added)
# mean & std metabolics cost reduction percents after adding mass
mean_bi_noload_metabolics_addedmass_percent, std_bi_noload_metabolics_addedmass_percent = utils.pareto_avg_std_energy(bi_noload_metabolics_addedmass_percent,reshape=False)
mean_mono_noload_metabolics_addedmass_percent, std_mono_noload_metabolics_addedmass_percent = utils.pareto_avg_std_energy(mono_noload_metabolics_addedmass_percent,reshape=False)

#####################################################################################
# plots

# subjects pareto curve: noload mono vs biarticular

plot_dic = {'x1_data':bi_noload_metabolics_percent,'x2_data':mono_noload_metabolics_percent,
          'y1_data':bi_noload_energy,'y2_data':mono_noload_energy,
          'color_1':mycolors['magenta pink'],'color_2':mycolors['lavender purple'],
          'ylabel':'Energy Consumption (W/kg)','xlabel':'Metabolic Reduction (%)'
          }
fig = plt.figure(num='Pareto Curve: noload mono vs bi',figsize=(10.4, 18.8))
utils.plot_pareto_curve_subjects (nrows=7,ncols=3,nplot=21,plot_dic=plot_dic,loadcond='noload')
fig.tight_layout()
fig.savefig('./Figures/Pareto/Adding_Mass_Pareto/Pareto_Noload_Subjects_BiVsMono_NoMass.pdf',orientation='landscape',bbox_inches='tight')
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
fig.savefig('./Figures/Pareto/Adding_Mass_Pareto/Pareto_Noload_BiVsMono_NoMass.pdf',orientation='landscape',bbox_inches='tight')
plt.show()

#####################################################################################
# plots with masses

# subjects pareto curve: noload mono vs biarticular

plot_dic = {'x1_data':bi_noload_metabolics_addedmass_percent,'x2_data':mono_noload_metabolics_addedmass_percent,
          'y1_data':bi_noload_energy,'y2_data':mono_noload_energy,
          'color_1':mycolors['burgundy red'],'color_2':mycolors['royal blue'],
          'ylabel':'Energy Consumption (W/kg)','xlabel':'Metabolic Reduction (%)'
          }
fig = plt.figure(num='Pareto Curve: noload mono vs bi',figsize=(10.4, 18.8))
utils.plot_pareto_curve_subjects (nrows=7,ncols=3,nplot=21,plot_dic=plot_dic,loadcond='noload')
fig.tight_layout()
fig.savefig('./Figures/Pareto/Adding_Mass_Pareto/Pareto_Noload_Subjects_BiVsMono_MassAdded.pdf',orientation='landscape',bbox_inches='tight')
plt.show()

# average pareto curve: noload mono vs biarticular

plot_dic = {'x1_data':mean_bi_noload_metabolics_addedmass_percent,'x1err_data':std_bi_noload_metabolics_addedmass_percent,
          'x2_data':mean_mono_noload_metabolics_addedmass_percent,'x2err_data':std_mono_noload_metabolics_addedmass_percent,
          'y1_data':mean_bi_noload_energy,'y1err_data':std_bi_noload_energy,
          'y2_data':mean_mono_noload_energy,'y2err_data':std_mono_noload_energy,
          'color_1':mycolors['burgundy red'],'color_2':mycolors['royal blue']
          }
fig = plt.figure(num='Pareto Curve: noload mono vs bi',figsize=(10.4, 8.8))
utils.plot_pareto_avg_curve (plot_dic,loadcond='loaded')
plt.xlabel('Metabolic Cost Reduction (%)')
plt.ylabel('Exoskeleton Energy Consumption (W/kg)')
ax = plt.gca()
utils.no_top_right(ax)
plt.legend(loc='best',frameon=False)
fig.tight_layout()
fig.savefig('./Figures/Pareto/Adding_Mass_Pareto/Pareto_Noload_BiVsMono_MassAdded.pdf',orientation='landscape',bbox_inches='tight')
plt.show()
