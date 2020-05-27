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
# actuators regenrative energy
bi_noload_hip_regen_energy= np.reshape(assisted_energy_dataset['biarticular_pareto_noload_hipregenrative_energy'],(25,21),order='F')
bi_noload_knee_regen_energy= np.reshape(assisted_energy_dataset['biarticular_pareto_noload_kneeregenrative_energy'],(25,21),order='F')
bi_noload_regen_energy = -bi_noload_hip_regen_energy - bi_noload_knee_regen_energy + bi_noload_hip_energy + bi_noload_knee_energy
mono_noload_hip_regen_energy= np.reshape(assisted_energy_dataset['monoarticular_pareto_noload_hipregenrative_energy'],(25,21),order='F')
mono_noload_knee_regen_energy= np.reshape(assisted_energy_dataset['monoarticular_pareto_noload_kneeregenrative_energy'],(25,21),order='F')
mono_noload_regen_energy = -mono_noload_hip_regen_energy - mono_noload_knee_regen_energy + mono_noload_hip_energy + mono_noload_knee_energy

# mean & std actuators regenrative energy
# noload bi
mean_bi_noload_hip_regen_energy, std_bi_noload_hip_regen_energy = utils.pareto_avg_std_energy(-0.65*assisted_energy_dataset['biarticular_pareto_noload_hipregenrative_energy']+\
                                                                                                  assisted_energy_dataset['biarticular_pareto_noload_hipactuator_energy'],reshape=True)
mean_bi_noload_knee_regen_energy, std_bi_noload_knee_regen_energy = utils.pareto_avg_std_energy(-0.65*assisted_energy_dataset['biarticular_pareto_noload_kneeregenrative_energy']+\
                                                                                                    assisted_energy_dataset['biarticular_pareto_noload_kneeactuator_energy'],reshape=True)
mean_bi_noload_regen_energy, std_bi_noload_regen_energy = utils.pareto_avg_std_energy(-0.65*assisted_energy_dataset['biarticular_pareto_noload_kneeregenrative_energy']-\
                                                                                          0.65*assisted_energy_dataset['biarticular_pareto_noload_hipregenrative_energy']+\
                                                                                          assisted_energy_dataset['biarticular_pareto_noload_kneeactuator_energy']+\
                                                                                          assisted_energy_dataset['biarticular_pareto_noload_hipactuator_energy'],reshape=True)

# noload mono
mean_mono_noload_hip_regen_energy, std_mono_noload_hip_regen_energy = utils.pareto_avg_std_energy(-0.65*assisted_energy_dataset['monoarticular_pareto_noload_hipregenrative_energy']+\
                                                                                                  assisted_energy_dataset['monoarticular_pareto_noload_hipactuator_energy'],reshape=True)
mean_mono_noload_knee_regen_energy, std_mono_noload_knee_regen_energy = utils.pareto_avg_std_energy(-0.65*assisted_energy_dataset['monoarticular_pareto_noload_kneeregenrative_energy']+\
                                                                                                    assisted_energy_dataset['monoarticular_pareto_noload_kneeactuator_energy'],reshape=True)
mean_mono_noload_regen_energy, std_mono_noload_regen_energy = utils.pareto_avg_std_energy(-0.65*assisted_energy_dataset['monoarticular_pareto_noload_kneeregenrative_energy']-\
                                                                                          0.65*assisted_energy_dataset['monoarticular_pareto_noload_hipregenrative_energy']+\
                                                                                          assisted_energy_dataset['monoarticular_pareto_noload_kneeactuator_energy']+\
                                                                                          assisted_energy_dataset['monoarticular_pareto_noload_hipactuator_energy'],reshape=True)

#####################################################################################
# Processing Data For Adding Mass
biarticular_exoskeleton_dic = {'m_waist':4.5, 'm_thigh':1, 'm_shank':0.9, 'motor_max_torque':2, 'motor_inertia':0.000506, 'thigh_com':0.23, 'shank_com':0.18, 'leg_inertia':2.52}
monoarticular_exoskeleton_dic = {'m_waist':3, 'm_thigh':2.5, 'm_shank':0.9, 'motor_max_torque':2, 'motor_inertia':0.000506, 'thigh_com':0.3, 'shank_com':0.18, 'leg_inertia':2.52}
monoarticular_actuator_near_waist_exoskeleton_dic = {'m_waist':3, 'm_thigh':2.5, 'm_shank':0.9, 'motor_max_torque':2, 'motor_inertia':0.000506, 'thigh_com':0.18, 'shank_com':0.18, 'leg_inertia':2.52}
monoarticular_actuator_on_shank_exoskeleton_dic = {'m_waist':3, 'm_thigh':1, 'm_shank':0.9+1.5, 'motor_max_torque':2, 'motor_inertia':0.000506, 'thigh_com':0.23, 'shank_com':0.13, 'leg_inertia':2.52}

biarticular_out = utils.addingmass_metabolics_pareto(unassisted_energy_dataset['noload_metabolics_energy'],bi_noload_metabolics,biarticular_exoskeleton_dic)
monoarticular_out = utils.addingmass_metabolics_pareto(unassisted_energy_dataset['noload_metabolics_energy'],mono_noload_metabolics,monoarticular_exoskeleton_dic)
monoarticular_actuator_near_waist_out = utils.addingmass_metabolics_pareto(unassisted_energy_dataset['noload_metabolics_energy'],mono_noload_metabolics,monoarticular_actuator_near_waist_exoskeleton_dic)
monoarticular_actuator_on_shank_out = utils.addingmass_metabolics_pareto(unassisted_energy_dataset['noload_metabolics_energy'],mono_noload_metabolics,monoarticular_actuator_on_shank_exoskeleton_dic)
# Metabolic cost reduction after adding mass
noload_metabolics_energy_biarticular_mass_added = np.tile(unassisted_energy_dataset['noload_metabolics_energy'][np.newaxis][0,:],(25,1))  + biarticular_out[4]
noload_metabolics_energy_monoarticular_mass_added = np.tile(unassisted_energy_dataset['noload_metabolics_energy'][np.newaxis][0,:],(25,1)) + monoarticular_out[4]
noload_metabolics_energy_monoarticular_actuator_near_waist_mass_added = np.tile(unassisted_energy_dataset['noload_metabolics_energy'][np.newaxis][0,:],(25,1)) + monoarticular_actuator_near_waist_out[4]
noload_metabolics_energy_monoarticular_actuator_on_shank_mass_added = np.tile(unassisted_energy_dataset['noload_metabolics_energy'][np.newaxis][0,:],(25,1)) + monoarticular_actuator_on_shank_out[4]

#**************************
bi_noload_metabolics_addedmass_percent = utils.addingmass_metabolics_reduction(biarticular_out[0],noload_metabolics_energy_biarticular_mass_added)
mono_noload_metabolics_addedmass_percent = utils.addingmass_metabolics_reduction(monoarticular_out[0],noload_metabolics_energy_monoarticular_mass_added)
mono_noload_actuator_near_waist_metabolics_addedmass_percent = utils.addingmass_metabolics_reduction(monoarticular_actuator_near_waist_out[0],noload_metabolics_energy_monoarticular_actuator_near_waist_mass_added)
mono_noload_actuator_on_shank_metabolics_addedmass_percent = utils.addingmass_metabolics_reduction(monoarticular_actuator_on_shank_out[0],noload_metabolics_energy_monoarticular_actuator_on_shank_mass_added)

# mean & std metabolics cost reduction percents after adding mass
mean_bi_noload_metabolics_addedmass_percent, std_bi_noload_metabolics_addedmass_percent = utils.pareto_avg_std_energy(bi_noload_metabolics_addedmass_percent,reshape=False)
mean_mono_noload_metabolics_addedmass_percent, std_mono_noload_metabolics_addedmass_percent = utils.pareto_avg_std_energy(mono_noload_metabolics_addedmass_percent,reshape=False)
mean_mono_noload_actuator_near_waist_metabolics_addedmass_percent, std_mono_noload_actuator_near_waist_metabolics_addedmass_percent = utils.pareto_avg_std_energy(mono_noload_actuator_near_waist_metabolics_addedmass_percent,reshape=False)
mean_mono_noload_actuator_on_shank_metabolics_addedmass_percent, std_mono_noload_actuator_on_shank_metabolics_addedmass_percent = utils.pareto_avg_std_energy(mono_noload_actuator_on_shank_metabolics_addedmass_percent,reshape=False)
#####################################################################################
# Pareto filtering
unassist_noload_metabolics = unassisted_energy_dataset['noload_metabolics_energy']
bi_noload_metabolics_percent_Paretofront,\
bi_noload_energy_Paretofront = utils.paretofront_subjects(bi_noload_metabolics,bi_noload_energy,unassist_noload_metabolics)
mono_noload_metabolics_percent_Paretofront,\
mono_noload_energy_Paretofront = utils.paretofront_subjects(mono_noload_metabolics,mono_noload_energy,unassist_noload_metabolics)
#added mass
bi_noload_metabolics_addedmass_percent_Paretofront,\
bi_noload_energy_addedmass_Paretofront = utils.paretofront_subjects(biarticular_out[0],bi_noload_energy,noload_metabolics_energy_biarticular_mass_added,adding_mass_case=True)
mono_noload_metabolics_addedmass_percent_Paretofront,\
mono_noload_energy_addedmass_Paretofront = utils.paretofront_subjects(monoarticular_out[0],mono_noload_energy,noload_metabolics_energy_monoarticular_mass_added,adding_mass_case=True)
# regenerated
bi_noload_metabolics_addedmass_regen_percent_Paretofront,\
bi_noload_energy_regen_Paretofront = utils.paretofront_subjects(biarticular_out[0],bi_noload_regen_energy,noload_metabolics_energy_biarticular_mass_added,adding_mass_case=True)
mono_noload_metabolics_addedmass_regen_percent_Paretofront,\
mono_noload_energy_regen_Paretofront = utils.paretofront_subjects(monoarticular_out[0],mono_noload_regen_energy,noload_metabolics_energy_monoarticular_mass_added,adding_mass_case=True)
#####################################################################################
# noload biarticular
bi_noload_indices = np.array([25,24,23,22,21,19,18,17,13,12,11,1])
mean_bi_noload_paretofront = utils.manual_paretofront(mean_bi_noload_metabolics_percent,mean_bi_noload_energy,bi_noload_indices)
std_bi_noload_paretofront = utils.manual_paretofront(std_bi_noload_metabolics_percent,std_bi_noload_energy,bi_noload_indices)

# noload monoarticular
mono_noload_indices = np.array([25,20,15,14,13,8,7,6,2,1])
mean_mono_noload_paretofront = utils.manual_paretofront(mean_mono_noload_metabolics_percent,mean_mono_noload_energy,mono_noload_indices)
std_mono_noload_paretofront = utils.manual_paretofront(std_mono_noload_metabolics_percent,std_mono_noload_energy,mono_noload_indices)

# biarticular noload mass added
bi_noload_indices = np.array([25,24,23,22,20,19,18,13,12])
mean_bi_noload_addedmass_paretofront = utils.manual_paretofront(mean_bi_noload_metabolics_addedmass_percent,mean_bi_noload_energy,bi_noload_indices)
std_bi_noload_addedmass_paretofront = utils.manual_paretofront(std_bi_noload_metabolics_addedmass_percent,std_bi_noload_energy,bi_noload_indices)

# monoarticular noload mass added
mono_noload_indices = np.array([25,24,20,18,14,13,12])
mean_mono_noload_addedmass_paretofront = utils.manual_paretofront(mean_mono_noload_metabolics_addedmass_percent,mean_mono_noload_energy,mono_noload_indices)
std_mono_noload_addedmass_paretofront = utils.manual_paretofront(std_mono_noload_metabolics_addedmass_percent,std_mono_noload_energy,mono_noload_indices)

# monoarticular noload mass added - actuator near waist
mono_noload_actuator_near_waist_indices = np.array([25,24,20,19,18,14,13,12])
mean_mono_noload_actuator_near_waist_addedmass_paretofront = utils.manual_paretofront(mean_mono_noload_actuator_near_waist_metabolics_addedmass_percent,mean_mono_noload_energy,mono_noload_actuator_near_waist_indices)
std_mono_noload_actuator_near_waist_addedmass_paretofront = utils.manual_paretofront(std_mono_noload_actuator_near_waist_metabolics_addedmass_percent,std_mono_noload_energy,mono_noload_actuator_near_waist_indices)

# monoarticular noload mass added - actuator near shank
mono_noload_actuator_on_shank_indices = np.array([25,24,20,19,18,14,13,12])
mean_mono_noload_actuator_on_shank_addedmass_paretofront = utils.manual_paretofront(mean_mono_noload_actuator_on_shank_metabolics_addedmass_percent,mean_mono_noload_energy,mono_noload_actuator_on_shank_indices)
std_mono_noload_actuator_on_shank_addedmass_paretofront = utils.manual_paretofront(std_mono_noload_actuator_on_shank_metabolics_addedmass_percent,std_mono_noload_energy,mono_noload_actuator_on_shank_indices)

# noload biarticular added mass with regeneration
bi_noload_indices = np.array([25,24,23,20,19,18,17,12])
mean_bi_noload_regen_addedmass_paretofront = utils.manual_paretofront(mean_bi_noload_metabolics_addedmass_percent,mean_bi_noload_regen_energy,bi_noload_indices)
std_bi_noload_regen_addedmass_paretofront = utils.manual_paretofront(std_bi_noload_metabolics_addedmass_percent,std_bi_noload_regen_energy,bi_noload_indices)

# noload monoarticular added mass with regeneration
mono_noload_indices = np.array([25,24,20,19,18,14,13,12])
mean_mono_noload_regen_addedmass_paretofront = utils.manual_paretofront(mean_mono_noload_metabolics_addedmass_percent,mean_mono_noload_regen_energy,mono_noload_indices)
std_mono_noload_regen_addedmass_paretofront = utils.manual_paretofront(std_mono_noload_metabolics_addedmass_percent,std_mono_noload_regen_energy,mono_noload_indices)

# noload biarticular with regeneration
bi_noload_indices = np.array([25,24,21,20,19,18,17,13,12,11,6,1])
mean_bi_noload_regen_paretofront = utils.manual_paretofront(mean_bi_noload_metabolics_percent,mean_bi_noload_regen_energy,bi_noload_indices)
std_bi_noload_regen_paretofront = utils.manual_paretofront(std_bi_noload_metabolics_percent,std_bi_noload_regen_energy,bi_noload_indices)

# noload monoarticular with regeneration
mono_noload_indices = np.array([25,24,20,19,15,14,13,12,11])
mean_mono_noload_regen_paretofront = utils.manual_paretofront(mean_mono_noload_metabolics_percent,mean_mono_noload_regen_energy,mono_noload_indices)
std_mono_noload_regen_paretofront = utils.manual_paretofront(std_mono_noload_metabolics_percent,std_mono_noload_regen_energy,mono_noload_indices)

#####################################################################################
# PAPER FIGURE
# average pareto curve: noload mono three types
# monoarticular with typical design
plot_dic = {'x1_data':mean_mono_noload_paretofront[:,0],'x1err_data':std_mono_noload_paretofront[:,0],
          'x2_data':mean_mono_noload_addedmass_paretofront[:,0],'x2err_data':std_mono_noload_addedmass_paretofront[:,0],
          'y1_data':mean_mono_noload_paretofront[:,1],'y1err_data':std_mono_noload_paretofront[:,1],
          'y2_data':mean_mono_noload_addedmass_paretofront[:,1],'y2err_data':std_mono_noload_addedmass_paretofront[:,1],
          'color_1':'grey','color_2':mycolors['royal blue'],
          'legend_1':'monoarticular ideal','legend_2':'monoarticular mass added\n typical design'
          }
fig = plt.figure(num='Monoarticular exoskeleton types',figsize=(12.8, 9.6))
plt.subplot(2,2,1)
utils.plot_pareto_avg_curve (plot_dic,loadcond='noload',line=True)
plt.ylabel('Exoskeleton power\n consumption (W/kg)')
ax = plt.gca()
utils.no_top_right(ax)
ax.set_xticks([10, 15, 20, 25, 30])
ax.set_yticks([1, 1.5, 2, 2.5,2.7])
plt.title('Monoarticular exoskeleton\nTypical Design')
plt.legend(loc='best',frameon=False)
plt.tick_params(axis='both',direction='in')
# monoarticular with knee actuator near waist
plot_dic = {'x1_data':mean_mono_noload_paretofront[:,0],'x1err_data':std_mono_noload_paretofront[:,0],
          'x2_data':mean_mono_noload_actuator_near_waist_addedmass_paretofront[:,0],'x2err_data':std_mono_noload_actuator_near_waist_addedmass_paretofront[:,0],
          'y1_data':mean_mono_noload_paretofront[:,1],'y1err_data':std_mono_noload_paretofront[:,1],
          'y2_data':mean_mono_noload_actuator_near_waist_addedmass_paretofront[:,1],'y2err_data':std_mono_noload_actuator_near_waist_addedmass_paretofront[:,1],
          'color_1':'grey','color_2':mycolors['pine green'],
          'legend_1':'monoarticular ideal','legend_2':'monoarticular mass added\n knee actuator near waist'
          }
plt.subplot(2,2,2)
utils.plot_pareto_avg_curve (plot_dic,loadcond='noload',line=True)
ax = plt.gca()
utils.no_top_right(ax)
ax.set_xticks([10, 15, 20, 25, 30])
ax.set_yticks([1, 1.5, 2, 2.5,2.7])
plt.title('Monoarticular exoskeleton\nKnee actuator near waist')
plt.legend(loc='best',frameon=False)
plt.tick_params(axis='both',direction='in')
# monoarticular with knee actuator on shank
plot_dic = {'x1_data':mean_mono_noload_paretofront[:,0],'x1err_data':std_mono_noload_paretofront[:,0],
          'x2_data':mean_mono_noload_actuator_on_shank_addedmass_paretofront[:,0],'x2err_data':std_mono_noload_actuator_on_shank_addedmass_paretofront[:,0],
          'y1_data':mean_mono_noload_paretofront[:,1],'y1err_data':std_mono_noload_paretofront[:,1],
          'y2_data':mean_mono_noload_actuator_on_shank_addedmass_paretofront[:,1],'y2err_data':std_mono_noload_actuator_on_shank_addedmass_paretofront[:,1],
          'color_1':'grey','color_2':mycolors['french sky blue'],
          'legend_1':'monoarticular ideal','legend_2':'monoarticular mass added\n knee actuator on shank'
          }
plt.subplot(2,2,3)
utils.plot_pareto_avg_curve (plot_dic,loadcond='noload',line=True)
plt.xlabel('Metabolic cost\n reduction (%)')
plt.ylabel('Exoskeleton power\n consumption (W/kg)')
ax = plt.gca()
utils.no_top_right(ax)
ax.set_xticks([10, 15, 20, 25, 30])
ax.set_yticks([1, 1.5, 2, 2.5,2.7])
plt.title('Monoarticular exoskeleton\nKnee actuator on shank')
plt.legend(loc='best',frameon=False)
plt.tick_params(axis='both',direction='in')
# average pareto curve: noload mono three types
plot_dic = {'x1_data':mean_mono_noload_addedmass_paretofront[:,0],'x1err_data':std_mono_noload_addedmass_paretofront[:,0],
          'x2_data':mean_mono_noload_actuator_near_waist_addedmass_paretofront[:,0],'x2err_data':std_mono_noload_actuator_near_waist_addedmass_paretofront[:,0],
          'x3_data':mean_mono_noload_actuator_on_shank_addedmass_paretofront[:,0],'x3err_data':std_mono_noload_actuator_on_shank_addedmass_paretofront[:,0],
          'y1_data':mean_mono_noload_addedmass_paretofront[:,1],'y1err_data':std_mono_noload_addedmass_paretofront[:,1],
          'y2_data':mean_mono_noload_actuator_near_waist_addedmass_paretofront[:,1],'y2err_data':std_mono_noload_actuator_near_waist_addedmass_paretofront[:,1],
          'y3_data':mean_mono_noload_actuator_on_shank_addedmass_paretofront[:,1],'y3err_data':std_mono_noload_actuator_on_shank_addedmass_paretofront[:,1],
          'color_1':mycolors['royal blue'],'color_2':mycolors['pine green'],'color_3':mycolors['french sky blue'],
          'legend_1':'Monoarticular typical design','legend_2':'Monoarticular with knee\n actuator near waist',
          'legend_3':'Monoarticular with knee\n actuator mounted on shank'}
plt.subplot(2,2,4)
utils.plot_pareto_avg_curve (plot_dic,loadcond='noload',third_plot=True,line=True)
plt.xlabel('Metabolic cost\n reduction (%)')
ax = plt.gca()
utils.no_top_right(ax)
ax.set_xticks([10, 15, 20])
ax.set_yticks([1, 1.5, 2, 2.5])
plt.title('Monoarticular exoskeleton\nDifferent designs')
plt.legend(loc='best',frameon=False)
plt.tick_params(axis='both',direction='in')
fig.tight_layout()
fig.tight_layout(h_pad=-1, w_pad=-1.5)
fig.subplots_adjust(top=0.98, bottom=0.075, left=0.100, right=0.975,hspace=0.25,wspace=0.15)
fig.savefig('./Figures/Paretofront/Adding_Mass_Pareto/PaperFigure_MonoarticularExoskeleton_DifferentConfigs.pdf',orientation='landscape',bbox_inches='tight')
plt.show()
#####################################################################################
# PAPER FIGURE
# plots
# average pareto front: noload mono vs biarticular: ideal devices

plot_dic = {'x1_data':mean_bi_noload_paretofront[:,0],'x1err_data':std_bi_noload_paretofront[:,0],
          'x2_data':mean_mono_noload_paretofront[:,0],'x2err_data':std_mono_noload_paretofront[:,0],
          'y1_data':mean_bi_noload_paretofront[:,1],'y1err_data':std_bi_noload_paretofront[:,1],
          'y2_data':mean_mono_noload_paretofront[:,1],'y2err_data':std_mono_noload_paretofront[:,1],
          'color_1':mycolors['french rose'],'color_2':mycolors['lavender purple']
          }
fig, axes = plt.subplots(nrows=2,ncols=2,num='PaperFigure_Paretofront',figsize=(12.8, 9.6))
plt.subplot(2,2,1)
utils.plot_pareto_avg_curve (plot_dic,loadcond='noload',line=True)
plt.ylabel('Exoskeleton Energy\n Consumption (W/kg)')
plt.title('noload, ideal devices')
ax = plt.gca()
ax.set_xticks([10, 15, 20, 25, 30])
ax.set_yticks([1, 1.5, 2, 2.5, 3, 3.5])
plt.tick_params(axis='both',direction='in')
utils.no_top_right(ax)
plt.legend(loc='best',frameon=False)

# average pareto front: noload mono vs biarticular : under mass/inertia effect

plot_dic = {'x1_data':mean_bi_noload_addedmass_paretofront[:,0],'x1err_data':std_bi_noload_addedmass_paretofront[:,0],
          'x2_data':mean_mono_noload_addedmass_paretofront[:,0],'x2err_data':std_mono_noload_addedmass_paretofront[:,0],
          'y1_data':mean_bi_noload_addedmass_paretofront[:,1],'y1err_data':std_bi_noload_addedmass_paretofront[:,1],
          'y2_data':mean_mono_noload_addedmass_paretofront[:,1],'y2err_data':std_mono_noload_addedmass_paretofront[:,1],
          'color_1':mycolors['crimson red'],'color_2':mycolors['dark purple']
          }
plt.subplot(2,2,2)
utils.plot_pareto_avg_curve (plot_dic,loadcond='noload',line=True)
plt.title('noload, devices under\n mass/inertia effect')
ax = plt.gca()
ax.set_xticks([5, 10, 15, 20])
ax.set_yticks([1, 1.5, 2, 2.5, 3])
plt.tick_params(axis='both',direction='in')
utils.no_top_right(ax)
plt.legend(loc='best',frameon=False)

# average pareto front: noload mono vs biarticular : under regeneration effect

plot_dic = {'x1_data':mean_bi_noload_regen_paretofront[:,0],'x1err_data':std_bi_noload_regen_paretofront[:,0],
          'x2_data':mean_mono_noload_regen_paretofront[:,0],'x2err_data':std_mono_noload_regen_paretofront[:,0],
          'y1_data':mean_bi_noload_regen_paretofront[:,1],'y1err_data':std_bi_noload_regen_paretofront[:,1],
          'y2_data':mean_mono_noload_regen_paretofront[:,1],'y2err_data':std_mono_noload_regen_paretofront[:,1],
          'color_1':mycolors['salmon'],'color_2':mycolors['olympic blue']
          }
plt.subplot(2,2,3)
utils.plot_pareto_avg_curve (plot_dic,loadcond='noload',line=True)
plt.ylabel('Exoskeleton Energy\n Consumption (W/kg)')
plt.xlabel('metabolic cost\n reduction (%)')
plt.title('noload, devices under\n regenration effect')
ax = plt.gca()
ax.set_xticks([10, 15, 20, 25, 30])
ax.set_yticks([0.5, 1, 1.5, 2, 2.5])
plt.tick_params(axis='both',direction='in')
utils.no_top_right(ax)
plt.legend(loc='best',frameon=False)


# average pareto front: noload mono vs biarticular : under mass/inertia and regenration effect

plot_dic = {'x1_data':mean_bi_noload_regen_addedmass_paretofront[:,0],'x1err_data':std_bi_noload_regen_addedmass_paretofront[:,0],
          'x2_data':mean_mono_noload_regen_addedmass_paretofront[:,0],'x2err_data':std_mono_noload_regen_addedmass_paretofront[:,0],
          'y1_data':mean_bi_noload_regen_addedmass_paretofront[:,1],'y1err_data':std_bi_noload_regen_addedmass_paretofront[:,1],
          'y2_data':mean_mono_noload_regen_addedmass_paretofront[:,1],'y2err_data':std_mono_noload_regen_addedmass_paretofront[:,1],
          'color_1':mycolors['imperial red'],'color_2':mycolors['teal']
          }
plt.subplot(2,2,4)
utils.plot_pareto_avg_curve (plot_dic,loadcond='noload',line=True)
plt.xlabel('metabolic cost\n reduction (%)')
plt.title('noload, devices under\n mass/inertia and regenration effect')
ax = plt.gca()
ax.set_xticks([5, 10, 15, 20])
ax.set_yticks([0.5, 1, 1.5, 2, 2.5])
plt.tick_params(axis='both',direction='in')
utils.no_top_right(ax)
plt.legend(loc='best',frameon=False)
fig.tight_layout(h_pad=-1, w_pad=-1.5)
fig.subplots_adjust(top=0.98, bottom=0.075, left=0.100, right=0.975,hspace=0.25,wspace=0.15)
fig.savefig('./Figures/Paretofront/Adding_Mass_Pareto/PaperFigure_AddingMass_Pareto.pdf',orientation='landscape',bbox_inches='tight')
plt.show()

#####################################################################################
# PAPER FIGURE
# Pareto comparison to ideal exo
# average pareto front: noload mono vs biarticular: ideal devices
# monoarticular ideal vs addedmass
plot_dic = {'x1_data':mean_mono_noload_paretofront[:,0],'x1err_data':std_mono_noload_paretofront[:,0],
          'x2_data':mean_mono_noload_addedmass_paretofront[:,0],'x2err_data':std_mono_noload_addedmass_paretofront[:,0],
          'y1_data':mean_mono_noload_paretofront[:,1],'y1err_data':std_mono_noload_paretofront[:,1],
          'y2_data':mean_mono_noload_addedmass_paretofront[:,1],'y2err_data':std_mono_noload_addedmass_paretofront[:,1],
          'color_1':mycolors['lavender purple'],'color_2':mycolors['dark purple'],
          'legend_1':'ideal device','legend_2':'device with mass/inertia'
          }
fig, axes = plt.subplots(nrows=2,ncols=2,num='PaperFigure_Paretofront',figsize=(14.8, 11.6))
plt.subplot(2,3,1)
utils.plot_pareto_avg_curve (plot_dic,loadcond='noload',line=True)
plt.ylabel('Exoskeleton Energy\n Consumption (W/kg)')
plt.title('monoarticular exoskeleton')
ax = plt.gca()
ax.set_xticks([5,10, 15, 20, 25, 30])
ax.set_yticks([1, 1.5, 2, 2.5, 3])
plt.tick_params(axis='both',direction='in')
utils.no_top_right(ax)
plt.legend(loc='best',frameon=False)

# monoarticular ideal vs regeneration
plot_dic = {'x1_data':mean_mono_noload_paretofront[:,0],'x1err_data':std_mono_noload_paretofront[:,0],
          'x2_data':mean_mono_noload_regen_paretofront[:,0],'x2err_data':std_mono_noload_regen_paretofront[:,0],
          'y1_data':mean_mono_noload_paretofront[:,1],'y1err_data':std_mono_noload_paretofront[:,1],
          'y2_data':mean_mono_noload_regen_paretofront[:,1],'y2err_data':std_mono_noload_regen_paretofront[:,1],
          'color_1':mycolors['lavender purple'],'color_2':mycolors['olympic blue'],
          'legend_1':'ideal device','legend_2':'device with regeneration'
          }
plt.subplot(2,3,2)
utils.plot_pareto_avg_curve (plot_dic,loadcond='noload',line=True)
plt.title('monoarticular exoskeleton')
ax = plt.gca()
ax.set_xticks([5,10, 15, 20, 25, 30])
ax.set_yticks([1, 1.5, 2, 2.5, 3])
plt.tick_params(axis='both',direction='in')
utils.no_top_right(ax)
plt.legend(loc='best',frameon=False)

# monoarticular ideal vs regeneration+mass_added
plot_dic = {'x1_data':mean_mono_noload_paretofront[:,0],'x1err_data':std_mono_noload_paretofront[:,0],
          'x2_data':mean_mono_noload_regen_addedmass_paretofront[:,0],'x2err_data':std_mono_noload_regen_addedmass_paretofront[:,0],
          'y1_data':mean_mono_noload_paretofront[:,1],'y1err_data':std_mono_noload_paretofront[:,1],
          'y2_data':mean_mono_noload_regen_addedmass_paretofront[:,1],'y2err_data':std_mono_noload_regen_addedmass_paretofront[:,1],
          'color_1':mycolors['lavender purple'],'color_2':mycolors['teal'],
          'legend_1':'ideal device','legend_2':'device with regeneration\n and mass/inertia'
          }
plt.subplot(2,3,3)
utils.plot_pareto_avg_curve (plot_dic,loadcond='noload',line=True)
plt.title('monoarticular exoskeleton')
ax = plt.gca()
ax.set_xticks([5,10, 15, 20, 25, 30])
ax.set_yticks([1, 1.5, 2, 2.5, 3])
plt.tick_params(axis='both',direction='in')
utils.no_top_right(ax)
plt.legend(loc='best',frameon=False)

# biarticular ideal vs addedmass
plot_dic = {'x1_data':mean_bi_noload_paretofront[:,0],'x1err_data':std_bi_noload_paretofront[:,0],
          'x2_data':mean_bi_noload_addedmass_paretofront[:,0],'x2err_data':std_bi_noload_addedmass_paretofront[:,0],
          'y1_data':mean_bi_noload_paretofront[:,1],'y1err_data':std_bi_noload_paretofront[:,1],
          'y2_data':mean_bi_noload_addedmass_paretofront[:,1],'y2err_data':std_bi_noload_addedmass_paretofront[:,1],
          'color_1':mycolors['french rose'],'color_2':mycolors['crimson red'],
          'legend_1':'ideal device','legend_2':'device with mass/inertia'
          }
plt.subplot(2,3,4)
utils.plot_pareto_avg_curve (plot_dic,loadcond='noload',line=True)
plt.ylabel('Exoskeleton Energy\n Consumption (W/kg)')
plt.xlabel('metabolic cost\n reduction (%)')
plt.title('biarticular exoskeleton')
ax = plt.gca()
ax.set_xticks([5,10, 15, 20, 25, 30])
ax.set_yticks([1, 1.5, 2, 2.5, 3, 3.5])
plt.tick_params(axis='both',direction='in')
utils.no_top_right(ax)
plt.legend(loc='best',frameon=False)

# biarticular ideal vs regeneration
plot_dic = {'x1_data':mean_bi_noload_paretofront[:,0],'x1err_data':std_bi_noload_paretofront[:,0],
          'x2_data':mean_bi_noload_regen_paretofront[:,0],'x2err_data':std_bi_noload_regen_paretofront[:,0],
          'y1_data':mean_bi_noload_paretofront[:,1],'y1err_data':std_bi_noload_paretofront[:,1],
          'y2_data':mean_bi_noload_regen_paretofront[:,1],'y2err_data':std_bi_noload_regen_paretofront[:,1],
          'color_1':mycolors['french rose'],'color_2':mycolors['salmon'],
          'legend_1':'ideal device','legend_2':'device with regeneration'
          }
plt.subplot(2,3,5)
utils.plot_pareto_avg_curve (plot_dic,loadcond='noload',line=True)
plt.xlabel('metabolic cost\n reduction (%)')
plt.title('biarticular exoskeleton')
ax = plt.gca()
ax.set_xticks([5,10, 15, 20, 25, 30])
ax.set_yticks([1, 1.5, 2, 2.5, 3, 3.5])
plt.tick_params(axis='both',direction='in')
utils.no_top_right(ax)
plt.legend(loc='best',frameon=False)

# biarticular ideal vs regeneration and mass/inertia
plot_dic = {'x1_data':mean_bi_noload_paretofront[:,0],'x1err_data':std_bi_noload_paretofront[:,0],
          'x2_data':mean_bi_noload_regen_addedmass_paretofront[:,0],'x2err_data':std_bi_noload_regen_addedmass_paretofront[:,0],
          'y1_data':mean_bi_noload_paretofront[:,1],'y1err_data':std_bi_noload_paretofront[:,1],
          'y2_data':mean_bi_noload_regen_addedmass_paretofront[:,1],'y2err_data':std_bi_noload_regen_addedmass_paretofront[:,1],
          'color_1':mycolors['french rose'],'color_2':mycolors['imperial red'],
          'legend_1':'ideal device','legend_2':'device with regeneration\n and mass/inertia'
          }
plt.subplot(2,3,6)
utils.plot_pareto_avg_curve (plot_dic,loadcond='noload',line=True)
plt.xlabel('metabolic cost\n reduction (%)')
plt.title('biarticular exoskeleton')
ax = plt.gca()
ax.set_xticks([5,10, 15, 20, 25, 30])
ax.set_yticks([1, 1.5, 2, 2.5, 3, 3.5])
plt.tick_params(axis='both',direction='in')
utils.no_top_right(ax)
plt.legend(loc='best',frameon=False)
fig.tight_layout(h_pad=-1, w_pad=-1.5)
fig.subplots_adjust(top=0.98, bottom=0.075, left=0.100, right=0.975,hspace=0.25,wspace=0.15)
fig.savefig('./Figures/Paretofront/Adding_Mass_Pareto/PaperFigure_Pareto_Comparison.pdf',orientation='landscape',bbox_inches='tight')
plt.show()

#####################################################################################

# plots with masses

# subjects pareto front: noload mono vs biarticular mass added

plot_dic = {'x1_data':bi_noload_metabolics_addedmass_percent_Paretofront,'x2_data':mono_noload_metabolics_addedmass_percent_Paretofront,
          'y1_data':bi_noload_energy_addedmass_Paretofront,'y2_data':mono_noload_energy_addedmass_Paretofront,
          'color_1':mycolors['burgundy red'],'color_2':mycolors['royal blue'],
          'ylabel':'Energy Consumption (W/kg)','xlabel':'Metabolic Reduction (%)'
          }
fig = plt.figure(num='Pareto Front: noload mono vs bi',figsize=(10.4, 18.8))
utils.plot_pareto_curve_subjects (nrows=7,ncols=3,nplot=21,plot_dic=plot_dic,loadcond='noload')
fig.tight_layout()
fig.savefig('./Figures/Paretofront/Adding_Mass_Pareto/Pareto_Noload_Subjects_BiVsMono_MassAdded.pdf',orientation='landscape',bbox_inches='tight')
plt.show()

# subjects pareto front: biarticular mass added vs biarticular noload

plot_dic = {'x1_data':bi_noload_metabolics_addedmass_percent_Paretofront,'x2_data':bi_noload_metabolics_percent_Paretofront,
          'y1_data':bi_noload_energy_addedmass_Paretofront,'y2_data':bi_noload_energy_Paretofront,
          'color_1':mycolors['burgundy red'],'color_2':mycolors['salmon'],
          'ylabel':'Energy Consumption (W/kg)','xlabel':'Metabolic Reduction (%)',
          'legend_1': 'bi non-ideal','legend_2': 'bi ideal'}
fig = plt.figure(num='Pareto Front: noload bi non-ideal vs ideal',figsize=(10.4, 18.8))
utils.plot_pareto_curve_subjects (nrows=7,ncols=3,nplot=21,plot_dic=plot_dic,loadcond='noload')
fig.tight_layout()
fig.savefig('./Figures/Paretofront/Adding_Mass_Pareto/Pareto_Noload_Subjects_Bi_MassAddedVSIdeal.pdf',orientation='landscape',bbox_inches='tight')
plt.show()

# subjects pareto front: monoarticular mass added vs monoarticular noload

plot_dic = {'x1_data':mono_noload_metabolics_addedmass_percent_Paretofront,'x2_data':mono_noload_metabolics_percent_Paretofront,
          'y1_data':mono_noload_energy_addedmass_Paretofront,'y2_data':mono_noload_energy_Paretofront,
          'color_1':mycolors['royal blue'],'color_2':mycolors['lavender purple'],
          'ylabel':'Energy Consumption (W/kg)','xlabel':'Metabolic Reduction (%)',
          'legend_1': 'mono non-ideal','legend_2': 'mono ideal'}
fig = plt.figure(num='Pareto Front: noload mono  non-ideal vs ideal',figsize=(10.4, 18.8))
utils.plot_pareto_curve_subjects (nrows=7,ncols=3,nplot=21,plot_dic=plot_dic,loadcond='noload')
fig.tight_layout()
fig.savefig('./Figures/Paretofront/Adding_Mass_Pareto/Pareto_Noload_Subjects_Mono_MassAddedVSIdeal.pdf',orientation='landscape',bbox_inches='tight')
plt.show()

# average pareto front: noload mono vs biarticular mass added

plot_dic = {'x1_data':mean_bi_noload_addedmass_paretofront[:,0],'x1err_data':std_bi_noload_addedmass_paretofront[:,0],
          'x2_data':mean_mono_noload_addedmass_paretofront[:,0],'x2err_data':std_mono_noload_addedmass_paretofront[:,0],
          'y1_data':mean_bi_noload_addedmass_paretofront[:,1],'y1err_data':std_bi_noload_addedmass_paretofront[:,1],
          'y2_data':mean_mono_noload_addedmass_paretofront[:,1],'y2err_data':std_mono_noload_addedmass_paretofront[:,1],
          'color_1':mycolors['burgundy red'],'color_2':mycolors['royal blue']
          }
fig = plt.figure(num='Pareto Front: noload mono vs bi',figsize=(10.4, 8.8))
utils.plot_pareto_avg_curve (plot_dic,loadcond='noload',line=True)
plt.xlabel('Metabolic Cost Reduction (%)')
plt.ylabel('Exoskeleton Energy Consumption (W/kg)')
ax = plt.gca()
utils.no_top_right(ax)
plt.legend(loc='best',frameon=False)
fig.tight_layout()
fig.savefig('./Figures/Paretofront/Adding_Mass_Pareto/Pareto_Noload_BiVsMono_MassAdded.pdf',orientation='landscape',bbox_inches='tight')
plt.show()

# average pareto curve: biarticular ideal vs mass added

plot_dic = {'x1_data':mean_bi_noload_addedmass_paretofront[:,0],'x1err_data':std_bi_noload_addedmass_paretofront[:,0],
          'x2_data':mean_bi_noload_paretofront[:,0],'x2err_data':std_bi_noload_paretofront[:,0],
          'y1_data':mean_bi_noload_addedmass_paretofront[:,1],'y1err_data':std_bi_noload_addedmass_paretofront[:,1],
          'y2_data':mean_bi_noload_paretofront[:,1],'y2err_data':std_bi_noload_paretofront[:,1],
          'color_1':mycolors['burgundy red'],'color_2':mycolors['salmon'],
          'legend_1': 'bi non-ideal','legend_2': 'bi ideal'}

fig = plt.figure(num='Pareto Front: noload bi non-ideal vs ideal',figsize=(10.4, 8.8))
utils.plot_pareto_avg_curve (plot_dic,loadcond='noload',line=True)
plt.xlabel('Metabolic Cost Reduction (%)')
plt.ylabel('Exoskeleton Energy Consumption (W/kg)')
ax = plt.gca()
utils.no_top_right(ax)
plt.legend(loc='best',frameon=False)
fig.tight_layout()
fig.savefig('./Figures/Paretofront/Adding_Mass_Pareto/Pareto_Noload_Bi_NonidealVsIdeal.pdf',orientation='landscape',bbox_inches='tight')
plt.show()

# average pareto front: monoarticular ideal vs mass added

plot_dic = {'x1_data':mean_mono_noload_addedmass_paretofront[:,0],'x1err_data':std_mono_noload_addedmass_paretofront[:,0],
          'x2_data':mean_mono_noload_paretofront[:,0],'x2err_data':std_mono_noload_paretofront[:,0],
          'y1_data':mean_mono_noload_addedmass_paretofront[:,1],'y1err_data':std_mono_noload_addedmass_paretofront[:,1],
          'y2_data':mean_mono_noload_paretofront[:,1],'y2err_data':std_mono_noload_paretofront[:,1],
          'color_1':mycolors['royal blue'],'color_2':mycolors['lavender purple'],
          'legend_1': 'mono non-ideal','legend_2': 'mono ideal'}
        
fig = plt.figure(num='Pareto Front: noload mono non-ideal vs ideal',figsize=(10.4, 8.8))
utils.plot_pareto_avg_curve (plot_dic,loadcond='noload',line=True)
plt.xlabel('Metabolic Cost Reduction (%)')
plt.ylabel('Exoskeleton Energy Consumption (W/kg)')
ax = plt.gca()
utils.no_top_right(ax)
plt.legend(loc='best',frameon=False)
fig.tight_layout()
fig.savefig('./Figures/Paretofront/Adding_Mass_Pareto/Pareto_Noload_Mono_NonidealVsIdeal.pdf',orientation='landscape',bbox_inches='tight')
plt.show()

#####################################################################################
#####################################################################################
# plots with regenration and mass

# subjects pareto front: noload mono vs biarticular mass added

plot_dic = {'x1_data':bi_noload_metabolics_addedmass_regen_percent_Paretofront,'x2_data':mono_noload_metabolics_addedmass_regen_percent_Paretofront,
          'y1_data':bi_noload_energy_regen_Paretofront,'y2_data':mono_noload_energy_regen_Paretofront,
          'color_1':mycolors['burgundy red'],'color_2':mycolors['royal blue'],
          'ylabel':'Energy Consumption (W/kg)','xlabel':'Metabolic Reduction (%)',
          'legend_1':'bi-regenerated','legend_2':'mono-regenerated'
          }
fig = plt.figure(num='Pareto Front: noload mono vs bi',figsize=(10.4, 18.8))
utils.plot_pareto_curve_subjects (nrows=7,ncols=3,nplot=21,plot_dic=plot_dic,loadcond='noload')
fig.tight_layout()
fig.savefig('./Figures/Paretofront/Adding_Mass_Pareto/Pareto_Noload_Subjects_BiVsMono_MassAdded_Regenerated.pdf',orientation='landscape',bbox_inches='tight')
plt.show()

# subjects pareto front: biarticular mass added vs biarticular noload

plot_dic = {'x1_data':bi_noload_metabolics_addedmass_percent_Paretofront,'x2_data':bi_noload_metabolics_addedmass_percent_Paretofront,
          'y1_data':bi_noload_energy_regen_Paretofront,'y2_data':bi_noload_energy_Paretofront,
          'color_1':mycolors['burgundy red'],'color_2':mycolors['salmon'],
          'ylabel':'Energy Consumption (W/kg)','xlabel':'Metabolic Reduction (%)',
          'legend_1': 'bi regenerated','legend_2': 'bi non-regenerated'}
fig = plt.figure(num='Pareto Front: noload bi regenerated vs no regeneration',figsize=(10.4, 18.8))
utils.plot_pareto_curve_subjects (nrows=7,ncols=3,nplot=21,plot_dic=plot_dic,loadcond='noload')
fig.tight_layout()
fig.savefig('./Figures/Paretofront/Adding_Mass_Pareto/Pareto_Noload_Subjects_Bi_GenVSNoGen.pdf',orientation='landscape',bbox_inches='tight')
plt.show()

# subjects pareto front: monoarticular mass added vs monoarticular noload

plot_dic = {'x1_data':mono_noload_metabolics_addedmass_percent_Paretofront,'x2_data':mono_noload_metabolics_addedmass_percent_Paretofront,
          'y1_data':mono_noload_energy_regen_Paretofront,'y2_data':mono_noload_energy_Paretofront,
          'color_1':mycolors['royal blue'],'color_2':mycolors['lavender purple'],
          'ylabel':'Energy Consumption (W/kg)','xlabel':'Metabolic Reduction (%)',
          'legend_1': 'mono non-regenerated','legend_2': 'mono regenerated'}
fig = plt.figure(num='Pareto Front: noload mono  non-ideal vs ideal',figsize=(10.4, 18.8))
utils.plot_pareto_curve_subjects (nrows=7,ncols=3,nplot=21,plot_dic=plot_dic,loadcond='noload')
fig.tight_layout()
fig.savefig('./Figures/Paretofront/Adding_Mass_Pareto/Pareto_Noload_Subjects_Mono_GenVsNonGen.pdf',orientation='landscape',bbox_inches='tight')
plt.show()


# average pareto curve: biarticular ideal vs mass added

plot_dic = {'x1_data':mean_bi_noload_regen_addedmass_paretofront[:,0],'x1err_data':std_bi_noload_regen_addedmass_paretofront[:,0],
          'x2_data':mean_bi_noload_addedmass_paretofront[:,0],'x2err_data':std_bi_noload_addedmass_paretofront[:,0],
          'y1_data':mean_bi_noload_regen_addedmass_paretofront[:,1],'y1err_data':std_bi_noload_regen_addedmass_paretofront[:,1],
          'y2_data':mean_bi_noload_addedmass_paretofront[:,1],'y2err_data':std_bi_noload_addedmass_paretofront[:,1],
          'color_1':mycolors['burgundy red'],'color_2':mycolors['salmon'],
          'legend_2': 'bi no-regenrated','legend_1': 'bi regenrated'}

fig = plt.figure(num='Pareto Curve: noload bi non-regenerated vs regenerated',figsize=(10.4, 8.8))
utils.plot_pareto_avg_curve (plot_dic,loadcond='noload',line=True)
plt.xlabel('Metabolic Cost Reduction (%)')
plt.ylabel('Exoskeleton Energy Consumption (W/kg)')
ax = plt.gca()
utils.no_top_right(ax)
plt.legend(loc='best',frameon=False)
fig.tight_layout()
fig.savefig('./Figures/Paretofront/Adding_Mass_Pareto/Pareto_Noload_Bi_NoGenVsGen.pdf',orientation='landscape',bbox_inches='tight')
plt.show()

# average pareto curve: monoarticular ideal vs mass added

plot_dic = {'x1_data':mean_mono_noload_regen_addedmass_paretofront[:,0],'x1err_data':std_mono_noload_regen_addedmass_paretofront[:,0],
          'x2_data':mean_mono_noload_addedmass_paretofront[:,0],'x2err_data':std_mono_noload_addedmass_paretofront[:,0],
          'y1_data':mean_mono_noload_regen_addedmass_paretofront[:,1],'y1err_data':std_mono_noload_regen_addedmass_paretofront[:,1],
          'y2_data':mean_mono_noload_addedmass_paretofront[:,1],'y2err_data':std_mono_noload_addedmass_paretofront[:,1],
          'color_1':mycolors['royal blue'],'color_2':mycolors['lavender purple'],
          'legend_2': 'mono non-regenerated','legend_1': 'mono regenerated'}
        
fig = plt.figure(num='Pareto Curve: noload mono non-regenerated vs regenerated',figsize=(10.4, 8.8))
utils.plot_pareto_avg_curve (plot_dic,loadcond='noload',line=True)
plt.xlabel('Metabolic Cost Reduction (%)')
plt.ylabel('Exoskeleton Energy Consumption (W/kg)')
ax = plt.gca()
utils.no_top_right(ax)
plt.legend(loc='best',frameon=False)
fig.tight_layout()
fig.savefig('./Figures/Paretofront/Adding_Mass_Pareto/Pareto_Noload_Mono_NonGenVsGen.pdf',orientation='landscape',bbox_inches='tight')
plt.show()