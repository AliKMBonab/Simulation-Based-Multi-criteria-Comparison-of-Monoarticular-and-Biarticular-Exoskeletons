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
from mpl_toolkits.mplot3d import Axes3D
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
ideal_dataset = utils.csv2numpy('./Data/Ideal/ideal_exos_dataset.csv') 
rra_dataset = utils.csv2numpy('./Data/RRA/rra_final_data.csv') 
unassist_dataset = utils.csv2numpy('./Data/Unassist/unassist_final_data.csv') 
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
# TODO: this part could be written more better than this. This was a fast analysis for me.
# mean & std actuators regenrative energy
mu_list = [0,0.30,0.37,0.50,0.65]
mean_bi_noload_regen_energy = np.zeros((25,len(mu_list)))
std_bi_noload_regen_energy = np.zeros((25,len(mu_list)))
mean_mono_noload_regen_energy = np.zeros((25,len(mu_list)))
std_mono_noload_regen_energy = np.zeros((25,len(mu_list)))
mean_bi_loaded_regen_energy = np.zeros((25,len(mu_list)))
std_bi_loaded_regen_energy = np.zeros((25,len(mu_list)))
mean_mono_loaded_regen_energy = np.zeros((25,len(mu_list)))
std_mono_loaded_regen_energy = np.zeros((25,len(mu_list)))
# indices
bi_loaded_indices = np.array([25,24,23,22,21,17,16,13,12,11,6,1])
mono_loaded_indices = np.array([25,20,15,10,5,4,3,2,1])
bi_noload_indices = np.array([25,24,23,22,21,19,18,17,13,12,11,1])
mono_noload_indices = np.array([25,20,15,14,13,8,7,6,2,1])
# regenerated
bi_loaded_regen_indices = np.array([25,24,23,22,21,17,13,12,11,6,1])
mono_loaded_regen_indices = np.array([25,20,19,15,10,5,9,4,3,2,1])
bi_noload_regen_indices = np.array([25,24,23,22,21,19,18,17,16,13,12,11,6,1])
mono_noload_regen_indices = np.array([25,24,20,15,10,5,4,1])

# paretofront dataset
# energy
mean_bi_noload_regen_paretofront_energy = np.zeros((25,len(mu_list)))
std_bi_noload_regen_paretofront_energy = np.zeros((25,len(mu_list)))
mean_mono_noload_regen_paretofront_energy = np.zeros((25,len(mu_list)))
std_mono_noload_regen_paretofront_energy = np.zeros((25,len(mu_list)))
mean_bi_loaded_regen_paretofront_energy = np.zeros((25,len(mu_list)))
std_bi_loaded_regen_paretofront_energy = np.zeros((25,len(mu_list)))
mean_mono_loaded_regen_paretofront_energy = np.zeros((25,len(mu_list)))
std_mono_loaded_regen_paretofront_energy = np.zeros((25,len(mu_list)))
# metabolics
mean_bi_noload_regen_paretofront_metabolics = np.zeros((25,len(mu_list)))
std_bi_noload_regen_paretofront_metabolics = np.zeros((25,len(mu_list)))
mean_mono_noload_regen_paretofront_metabolics = np.zeros((25,len(mu_list)))
std_mono_noload_regen_paretofront_metabolics = np.zeros((25,len(mu_list)))
mean_bi_loaded_regen_paretofront_metabolics = np.zeros((25,len(mu_list)))
std_bi_loaded_regen_paretofront_metabolics = np.zeros((25,len(mu_list)))
mean_mono_loaded_regen_paretofront_metabolics = np.zeros((25,len(mu_list)))
std_mono_loaded_regen_paretofront_metabolics = np.zeros((25,len(mu_list)))

c=0
for mu in mu_list:
    # noload bi 
    mean_bi_noload_regen_energy[:,c], std_bi_noload_regen_energy[:,c] = utils.pareto_avg_std_energy(-mu*assisted_energy_dataset['biarticular_pareto_noload_kneeregenrative_energy']-\
                                                                                        mu*assisted_energy_dataset['biarticular_pareto_noload_hipregenrative_energy']+\
                                                                                            assisted_energy_dataset['biarticular_pareto_noload_kneeactuator_energy']+\
                                                                                            assisted_energy_dataset['biarticular_pareto_noload_hipactuator_energy'],reshape=True)
    # noload mono
    mean_mono_noload_regen_energy[:,c], std_mono_noload_regen_energy[:,c] = utils.pareto_avg_std_energy(-mu*assisted_energy_dataset['monoarticular_pareto_noload_kneeregenrative_energy']-\
                                                                                            mu*assisted_energy_dataset['monoarticular_pareto_noload_hipregenrative_energy']+\
                                                                                            assisted_energy_dataset['monoarticular_pareto_noload_kneeactuator_energy']+\
                                                                                            assisted_energy_dataset['monoarticular_pareto_noload_hipactuator_energy'],reshape=True)
    # loaded bi
    mean_bi_loaded_regen_energy[:,c], std_bi_loaded_regen_energy[:,c] = utils.pareto_avg_std_energy(-mu*assisted_energy_dataset['biarticular_pareto_load_kneeregenrative_energy']-\
                                                                                        mu*assisted_energy_dataset['biarticular_pareto_load_hipregenrative_energy']+\
                                                                                            assisted_energy_dataset['biarticular_pareto_load_kneeactuator_energy']+\
                                                                                            assisted_energy_dataset['biarticular_pareto_load_hipactuator_energy'],reshape=True)
    # loaded mono
    mean_mono_loaded_regen_energy[:,c], std_mono_loaded_regen_energy[:,c] = utils.pareto_avg_std_energy(-mu*assisted_energy_dataset['monoarticular_pareto_load_kneeregenrative_energy']-\
                                                                                            mu*assisted_energy_dataset['monoarticular_pareto_load_hipregenrative_energy']+\
                                                                                            assisted_energy_dataset['monoarticular_pareto_load_kneeactuator_energy']+\
                                                                                            assisted_energy_dataset['monoarticular_pareto_load_hipactuator_energy'],reshape=True)
    # loaded biarticular regenerated
    mean_bi_loaded_regen_paretofront = utils.manual_paretofront(mean_bi_loaded_metabolics_percent,mean_bi_loaded_regen_energy[:,c],bi_loaded_regen_indices)
    std_bi_loaded_regen_paretofront = utils.manual_paretofront(std_bi_loaded_metabolics_percent,std_bi_loaded_regen_energy[:,c],bi_loaded_regen_indices)
    # metabolics
    mean_bi_loaded_regen_paretofront_metabolics[:,c] = mean_bi_loaded_regen_paretofront[:,0]
    std_bi_loaded_regen_paretofront_metabolics[:,c] = std_bi_loaded_regen_paretofront[:,0]
    # energy
    mean_bi_loaded_regen_paretofront_energy[:,c] = mean_bi_loaded_regen_paretofront[:,1]
    std_bi_loaded_regen_paretofront_energy[:,c] = std_bi_loaded_regen_paretofront[:,1]
    # loaded monoarticular regenerated
    mean_mono_loaded_regen_paretofront = utils.manual_paretofront(mean_mono_loaded_metabolics_percent,mean_mono_loaded_regen_energy[:,c],mono_loaded_regen_indices)
    std_mono_loaded_regen_paretofront = utils.manual_paretofront(std_mono_loaded_metabolics_percent,std_mono_loaded_regen_energy[:,c],mono_loaded_regen_indices)
    # metabolics
    mean_mono_loaded_regen_paretofront_metabolics[:,c] = mean_mono_loaded_regen_paretofront[:,0]
    std_mono_loaded_regen_paretofront_metabolics[:,c] = std_mono_loaded_regen_paretofront[:,0]
    # energy
    mean_mono_loaded_regen_paretofront_energy[:,c] = mean_mono_loaded_regen_paretofront[:,1]
    std_mono_loaded_regen_paretofront_energy[:,c] = std_mono_loaded_regen_paretofront[:,1]
    # noload biarticular regenerated
    mean_bi_noload_regen_paretofront = utils.manual_paretofront(mean_bi_noload_metabolics_percent,mean_bi_noload_regen_energy[:,c],bi_noload_regen_indices)
    std_bi_noload_regen_paretofront = utils.manual_paretofront(std_bi_noload_metabolics_percent,std_bi_noload_regen_energy[:,c],bi_noload_regen_indices)
    # metabolics
    mean_bi_noload_regen_paretofront_metabolics[:,c] = mean_bi_noload_regen_paretofront[:,0]
    std_bi_noload_regen_paretofront_metabolics[:,c] = std_bi_noload_regen_paretofront[:,0]
    # energy
    mean_bi_noload_regen_paretofront_energy[:,c] = mean_bi_noload_regen_paretofront[:,1]
    std_bi_noload_regen_paretofront_energy[:,c] = std_bi_noload_regen_paretofront[:,1]
    # noload monoarticular regenerated
    mean_mono_noload_regen_paretofront = utils.manual_paretofront(mean_mono_noload_metabolics_percent,mean_mono_noload_regen_energy[:,c],mono_noload_regen_indices)
    std_mono_noload_regen_paretofront = utils.manual_paretofront(std_mono_noload_metabolics_percent,std_mono_noload_regen_energy[:,c],mono_noload_regen_indices)
    # metabolics
    mean_mono_noload_regen_paretofront_metabolics[:,c] = mean_mono_noload_regen_paretofront[:,0]
    std_mono_noload_regen_paretofront_metabolics[:,c] = std_mono_noload_regen_paretofront[:,0]
    # energy
    mean_mono_noload_regen_paretofront_energy[:,c] = mean_mono_noload_regen_paretofront[:,1]
    std_mono_noload_regen_paretofront_energy[:,c] = std_mono_noload_regen_paretofront[:,1]
    c+=1
#####################################################################################
# Paretofront data
# mean & std metabolics cost reduction percents
# loaded biarticular
mean_bi_loaded_paretofront = utils.manual_paretofront(mean_bi_loaded_metabolics_percent,mean_bi_loaded_energy,bi_loaded_indices)
std_bi_loaded_paretofront = utils.manual_paretofront(std_bi_loaded_metabolics_percent,std_bi_loaded_energy,bi_loaded_indices)
#hip
mean_bi_loaded_hip_actuator_paretofront = utils.manual_paretofront(mean_bi_loaded_metabolics_percent,mean_bi_loaded_hip_energy,bi_loaded_indices)
std_bi_loaded_hip_actuator_paretofront = utils.manual_paretofront(std_bi_loaded_metabolics_percent,std_bi_loaded_hip_energy,bi_loaded_indices)
#knee
mean_bi_loaded_knee_actuator_paretofront = utils.manual_paretofront(mean_bi_loaded_metabolics_percent,mean_bi_loaded_knee_energy,bi_loaded_indices)
std_bi_loaded_knee_actuator_paretofront = utils.manual_paretofront(std_bi_loaded_metabolics_percent,std_bi_loaded_knee_energy,bi_loaded_indices)

# loaded monoarticular
mean_mono_loaded_paretofront = utils.manual_paretofront(mean_mono_loaded_metabolics_percent,mean_mono_loaded_energy,mono_loaded_indices)
std_mono_loaded_paretofront = utils.manual_paretofront(std_mono_loaded_metabolics_percent,std_mono_loaded_energy,mono_loaded_indices)
#hip
mean_mono_loaded_hip_actuator_paretofront = utils.manual_paretofront(mean_mono_loaded_metabolics_percent,mean_mono_loaded_hip_energy,mono_loaded_indices)
std_mono_loaded_hip_actuator_paretofront = utils.manual_paretofront(std_mono_loaded_metabolics_percent,std_mono_loaded_hip_energy,mono_loaded_indices)
#knee
mean_mono_loaded_knee_actuator_paretofront = utils.manual_paretofront(mean_mono_loaded_metabolics_percent,mean_mono_loaded_knee_energy,mono_loaded_indices)
std_mono_loaded_knee_actuator_paretofront = utils.manual_paretofront(std_mono_loaded_metabolics_percent,std_mono_loaded_knee_energy,mono_loaded_indices)

# noload biarticular
mean_bi_noload_paretofront = utils.manual_paretofront(mean_bi_noload_metabolics_percent,mean_bi_noload_energy,bi_noload_indices)
std_bi_noload_paretofront = utils.manual_paretofront(std_bi_noload_metabolics_percent,std_bi_noload_energy,bi_noload_indices)
#hip
mean_bi_noload_hip_actuator_paretofront = utils.manual_paretofront(mean_bi_noload_metabolics_percent,mean_bi_noload_hip_energy,bi_noload_indices)
std_bi_noload_hip_actuator_paretofront = utils.manual_paretofront(std_bi_noload_metabolics_percent,std_bi_noload_hip_energy,bi_noload_indices)
#knee
mean_bi_noload_knee_actuator_paretofront = utils.manual_paretofront(mean_bi_noload_metabolics_percent,mean_bi_noload_knee_energy,bi_noload_indices)
std_bi_noload_knee_actuator_paretofront = utils.manual_paretofront(std_bi_noload_metabolics_percent,std_bi_noload_knee_energy,bi_noload_indices)

# noload monoarticular
mean_mono_noload_paretofront = utils.manual_paretofront(mean_mono_noload_metabolics_percent,mean_mono_noload_energy,mono_noload_indices)
std_mono_noload_paretofront = utils.manual_paretofront(std_mono_noload_metabolics_percent,std_mono_noload_energy,mono_noload_indices)
#hip
mean_mono_noload_hip_actuator_paretofront = utils.manual_paretofront(mean_mono_noload_metabolics_percent,mean_mono_noload_hip_energy,mono_noload_indices)
std_mono_noload_hip_actuator_paretofront = utils.manual_paretofront(std_mono_noload_metabolics_percent,std_mono_noload_hip_energy,mono_noload_indices)
#knee
mean_mono_noload_knee_actuator_paretofront = utils.manual_paretofront(mean_mono_noload_metabolics_percent,mean_mono_noload_knee_energy,mono_noload_indices)
std_mono_noload_knee_actuator_paretofront = utils.manual_paretofront(std_mono_noload_metabolics_percent,std_mono_noload_knee_energy,mono_noload_indices)

# loaded biarticular regenerated
mean_bi_loaded_regen_paretofront = utils.manual_paretofront(mean_bi_loaded_metabolics_percent,mean_bi_loaded_regen_energy[:,-1],bi_loaded_regen_indices)
std_bi_loaded_regen_paretofront = utils.manual_paretofront(std_bi_loaded_metabolics_percent,std_bi_loaded_regen_energy[:,-1],bi_loaded_regen_indices)
# loaded monoarticular regenerated
mean_mono_loaded_regen_paretofront = utils.manual_paretofront(mean_mono_loaded_metabolics_percent,mean_mono_loaded_regen_energy[:,-1],mono_loaded_regen_indices)
std_mono_loaded_regen_paretofront = utils.manual_paretofront(std_mono_loaded_metabolics_percent,std_mono_loaded_regen_energy[:,-1],mono_loaded_regen_indices)
# noload biarticular regenerated
mean_bi_noload_regen_paretofront = utils.manual_paretofront(mean_bi_noload_metabolics_percent,mean_bi_noload_regen_energy[:,-1],bi_noload_regen_indices)
std_bi_noload_regen_paretofront = utils.manual_paretofront(std_bi_noload_metabolics_percent,std_bi_noload_regen_energy[:,-1],bi_noload_regen_indices)
# noload monoarticular regenerated
mean_mono_noload_regen_paretofront = utils.manual_paretofront(mean_mono_noload_metabolics_percent,mean_mono_noload_regen_energy[:,-1],mono_noload_regen_indices)
std_mono_noload_regen_paretofront = utils.manual_paretofront(std_mono_noload_metabolics_percent,std_mono_noload_regen_energy[:,-1],mono_noload_regen_indices)
#####################################################################################
# paretocurve for different regenerations
plt.rcParams.update({'font.size': 14})
'''
# loaded biarticular
plot_dic = {'x_values':np.transpose(np.tile(mean_bi_loaded_metabolics_percent,(len(mu_list),1))),
         'xerr_values':np.transpose(np.tile(std_bi_loaded_metabolics_percent,(len(mu_list),1))),
          'y_values':mean_bi_loaded_regen_energy,'yerr_values':std_bi_loaded_regen_energy,
          'legends':['0%','30%','37%','50%','65%'],'weights':np.transpose(np.tile(np.arange(1,26,1),(len(mu_list),1)))}
fig, axes = plt.subplots(nrows=2,ncols=2,num='PaperFigure_Paretofront',figsize=(12.8, 9.6))
plt.subplot(2,2,1)
utils.plot_regeneration_efficiency (plot_dic,line=False,label_on=False,ideal_color=mycolors['crimson red'])
plt.ylabel('Exoskeleton Energy\n Consumption (W/kg)')
plt.title('loaded: biarticular regenerated')
ax = plt.gca()
plt.tick_params(axis='both',direction='in')
utils.no_top_right(ax)
plt.legend(loc='best',frameon=False)
# loaded monoarticular
plot_dic = {'x_values':np.transpose(np.tile(mean_mono_loaded_metabolics_percent,(len(mu_list),1))),
         'xerr_values':np.transpose(np.tile(std_mono_loaded_metabolics_percent,(len(mu_list),1))),
          'y_values':mean_mono_loaded_regen_energy,'yerr_values':std_mono_loaded_regen_energy,
          'legends':['0%','30%','37%','50%','65%'],'weights':np.transpose(np.tile(np.arange(1,26,1),(len(mu_list),1)))}
plt.subplot(2,2,2)
utils.plot_regeneration_efficiency (plot_dic,line=False,label_on=False,ideal_color=mycolors['dark purple'])
plt.title('loaded: monoarticular regenerated')
ax = plt.gca()
plt.tick_params(axis='both',direction='in')
utils.no_top_right(ax)
# noload biarticular
plot_dic = {'x_values':np.transpose(np.tile(mean_bi_noload_metabolics_percent,(len(mu_list),1))),
         'xerr_values':np.transpose(np.tile(std_bi_noload_metabolics_percent,(len(mu_list),1))),
          'y_values':mean_bi_noload_regen_energy,'yerr_values':std_bi_noload_regen_energy,
          'legends':['0%','30%','37%','50%','65%'],'weights':np.transpose(np.tile(np.arange(1,26,1),(len(mu_list),1)))}
plt.subplot(2,2,3)
utils.plot_regeneration_efficiency (plot_dic,line=False,label_on=False,ideal_color=mycolors['french rose'])
plt.title('noload: biarticular regenerated')
plt.ylabel('Exoskeleton Energy\n Consumption (W/kg)')
ax.set_xlabel('metabolic cost\nreduction (%)')
ax = plt.gca()
plt.tick_params(axis='both',direction='in')
utils.no_top_right(ax)
# noload monoarticular
plot_dic = {'x_values':np.transpose(np.tile(mean_mono_noload_metabolics_percent,(len(mu_list),1))),
         'xerr_values':np.transpose(np.tile(std_mono_noload_metabolics_percent,(len(mu_list),1))),
          'y_values':mean_mono_noload_regen_energy,'yerr_values':std_mono_noload_regen_energy,
          'legends':['0%','30%','37%','50%','65%'],'weights':np.transpose(np.tile(np.arange(1,26,1),(len(mu_list),1)))}
plt.subplot(2,2,4)
utils.plot_regeneration_efficiency (plot_dic,line=False,label_on=False,ideal_color=mycolors['lavender purple'])
plt.title('noload: monoarticular regenerated')
ax = plt.gca()
plt.tick_params(axis='both',direction='in')
utils.no_top_right(ax)
ax.set_xlabel('metabolic cost\nreduction (%)')
plt.tick_params(axis='both',direction='in')
utils.no_top_right(ax)
fig.tight_layout(h_pad=-1, w_pad=-1.5)
fig.subplots_adjust(top=0.98, bottom=0.075, left=0.100, right=0.975,hspace=0.25,wspace=0.15)
fig.savefig('./Figures/Paretofront/Analyses_Pareto/Paretocurve_Regeneration_Efficiency.pdf',orientation='landscape',bbox_inches='tight')
plt.show()
'''
#####################################################################################
# paretofront for different regenerations
# loaded biarticular
plot_dic = {'x_values':mean_bi_loaded_regen_paretofront_metabolics,'xerr_values':std_bi_loaded_regen_paretofront_metabolics,
          'y_values':mean_bi_loaded_regen_paretofront_energy,'yerr_values':std_bi_loaded_regen_paretofront_energy,
          'legends':['0%','30%','37%','50%','65%'],'weights':np.transpose(np.tile(bi_loaded_indices,(len(mu_list),1)))}
fig, axes = plt.subplots(nrows=2,ncols=2,num='PaperFigure_Paretofront',figsize=(12.8, 9.6))
plt.subplot(2,2,4)
utils.plot_regeneration_efficiency (plot_dic,line=True,label_on=False,ideal_color=mycolors['crimson red'])
plt.title('loaded: biarticular regenerated')
ax = plt.gca()
ax.set_xlabel('metabolic cost\nreduction (%)')
ax.set_xticks([5, 10, 15, 20, 25])
ax.set_yticks([1, 1.5, 2, 2.5,3])
plt.tick_params(axis='both',direction='in')
utils.no_top_right(ax)
# loaded monoarticular
plot_dic = {'x_values':mean_mono_loaded_regen_paretofront_metabolics,'xerr_values':std_mono_loaded_regen_paretofront_metabolics,
          'y_values':mean_mono_loaded_regen_paretofront_energy,'yerr_values':std_mono_loaded_regen_paretofront_energy,
          'legends':['0%','30%','37%','50%','65%'],'weights':np.transpose(np.tile(mono_loaded_indices,(len(mu_list),1)))}
plt.subplot(2,2,3)
utils.plot_regeneration_efficiency (plot_dic,line=True,label_on=False,ideal_color=mycolors['dark purple'])
plt.title('loaded: monoarticular regenerated')
plt.ylabel('exoskeleton power\n consumption (W/kg)')
ax = plt.gca()
ax.set_xlabel('metabolic cost\nreduction (%)')
ax.set_xticks([5, 10, 15, 20, 25])
ax.set_yticks([1, 1.5, 2, 2.5,3])
plt.tick_params(axis='both',direction='in')
utils.no_top_right(ax)
# noload biarticular
plot_dic = {'x_values':mean_bi_noload_regen_paretofront_metabolics,'xerr_values':std_bi_noload_regen_paretofront_metabolics,
          'y_values':mean_bi_noload_regen_paretofront_energy,'yerr_values':std_bi_noload_regen_paretofront_energy,
          'legends':['0%','30%','37%','50%','65%'],'weights':np.transpose(np.tile(bi_noload_indices,(len(mu_list),1)))}
plt.subplot(2,2,2)
utils.plot_regeneration_efficiency (plot_dic,line=True,label_on=False,ideal_color=mycolors['french rose'])
plt.title('noload: biarticular regenerated')
ax = plt.gca()
ax.set_xticks([5, 10, 15, 20, 25])
ax.set_yticks([1, 1.5, 2, 2.5,3])
plt.tick_params(axis='both',direction='in')
utils.no_top_right(ax)
# noload monoarticular
plot_dic = {'x_values':mean_mono_noload_regen_paretofront_metabolics,'xerr_values':std_mono_noload_regen_paretofront_metabolics,
          'y_values':mean_mono_noload_regen_paretofront_energy,'yerr_values':std_mono_noload_regen_paretofront_energy,
          'legends':['0%','30%','37%','50%','65%'],'weights':np.transpose(np.tile(bi_noload_indices,(len(mu_list),1)))}
plt.subplot(2,2,1)
utils.plot_regeneration_efficiency (plot_dic,line=True,label_on=False,ideal_color=mycolors['lavender purple'])
plt.title('noload: monoarticular regenerated')
plt.ylabel('Exoskeleton Energy\n Consumption (W/kg)')
ax = plt.gca()
ax.set_xticks([5, 10, 15, 20, 25])
ax.set_yticks([1, 1.5, 2, 2.5,3])
plt.tick_params(axis='both',direction='in')
plt.legend(loc='best',frameon=False)
utils.no_top_right(ax)
fig.tight_layout(h_pad=-1, w_pad=-1.5)
fig.subplots_adjust(top=0.98, bottom=0.075, left=0.100, right=0.975,hspace=0.25,wspace=0.15)
fig.savefig('./Figures/Paretofront/Analyses_Pareto/Paretofront_Regeneration_Efficiency.pdf',orientation='landscape',bbox_inches='tight')
plt.show()

#####################################################################################
# PAPER FIGURE
# average pareto curve: loaded biarticular REGENERATED

plot_dic = {'x1_data':mean_bi_loaded_regen_paretofront[:,0],'x1err_data':std_bi_loaded_regen_paretofront[:,0],
          'x2_data':mean_bi_loaded_paretofront[:,0],'x2err_data':std_bi_loaded_paretofront[:,0],
          'y1_data':mean_bi_loaded_regen_paretofront[:,1],'y1err_data':std_bi_loaded_regen_paretofront[:,1],
          'y2_data':mean_bi_loaded_paretofront[:,1],'y2err_data':std_bi_loaded_paretofront[:,1],
          'color_2':mycolors['salmon'],'color_1':mycolors['burgundy red'],
          'legend_2':'biarticular, loaded','legend_1':'regenerated biarticular, loaded',
          }
fig, axes = plt.subplots(nrows=2,ncols=2,num='PaperFigure_Paretofront',figsize=(12.8, 9.6))
plt.subplot(2,2,1)
utils.plot_pareto_avg_curve (plot_dic,loadcond='loaded',line=True)
plt.ylabel('Exoskeleton Energy\n Consumption (W/kg)')
plt.title('loaded: biarticular regenerated')
ax = plt.gca()
ax.set_xticks([5, 10, 15, 20, 25, 30])
ax.set_yticks([1, 1.5, 2, 2.5,3.10])
plt.tick_params(axis='both',direction='in')
utils.no_top_right(ax)
plt.legend(loc='best',frameon=False)

# average pareto curve: loaded monoarticular

plot_dic = {'x1_data':mean_mono_loaded_regen_paretofront[:,0],'x1err_data':std_mono_loaded_regen_paretofront[:,0],
          'x2_data':mean_mono_loaded_paretofront[:,0],'x2err_data':std_mono_loaded_paretofront[:,0],
          'y1_data':mean_mono_loaded_regen_paretofront[:,1],'y1err_data':std_mono_loaded_regen_paretofront[:,1],
          'y2_data':mean_mono_loaded_paretofront[:,1],'y2err_data':std_mono_loaded_paretofront[:,1],
          'color_2':mycolors['pastel blue'],'color_1':mycolors['dark purple'],
          'legend_2':'monoarticular, loaded','legend_1':'regenerated monoarticular, loaded',
          }
plt.subplot(2,2,2)
utils.plot_pareto_avg_curve (plot_dic,loadcond='loaded',line=True)
plt.title('loaded: mononarticular regenerated')
plt.tick_params(axis='both',direction='in')
ax = plt.gca()
ax.set_xticks([5, 10, 15, 20, 25, 30])
ax.set_yticks([1, 1.5, 2, 2.5,3.10])
utils.no_top_right(ax)
plt.legend(loc='best',frameon=False)

# average pareto curve: noload biarticular 

plot_dic = {'x1_data':mean_bi_noload_regen_paretofront[:,0],'x1err_data':std_bi_noload_regen_paretofront[:,0],
          'x2_data':mean_bi_noload_paretofront[:,0],'x2err_data':std_bi_noload_paretofront[:,0],
          'y1_data':mean_bi_noload_regen_paretofront[:,1],'y1err_data':std_bi_noload_regen_paretofront[:,1],
          'y2_data':mean_bi_noload_paretofront[:,1],'y2err_data':std_bi_noload_paretofront[:,1],
          'color_2':mycolors['salmon'],'color_1':mycolors['french rose'],
          'legend_2':'biarticular, noload','legend_1':'regenerated biarticular, noload',
          }
plt.subplot(2,2,3)
utils.plot_pareto_avg_curve (plot_dic,loadcond='noload',line=True)
plt.title('noload: biarticular regenerated')
plt.ylabel('Exoskeleton Energy\n Consumption (W/kg)')
plt.tick_params(axis='both',direction='in')
ax = plt.gca()
ax.set_xticks([5, 10, 15, 20, 25, 30])
ax.set_yticks([1, 1.5, 2, 2.5,3.10])
ax.set_xlabel('metabolic cost\nreduction (%)')
utils.no_top_right(ax)
plt.legend(loc='best',frameon=False)

# average pareto curve: noload monoarticular

plot_dic = {'x1_data':mean_mono_noload_regen_paretofront[:,0],'x1err_data':std_mono_noload_regen_paretofront[:,0],
          'x2_data':mean_mono_noload_paretofront[:,0],'x2err_data':std_mono_noload_paretofront[:,0],
          'y1_data':mean_mono_noload_regen_paretofront[:,1],'y1err_data':std_mono_noload_regen_paretofront[:,1],
          'y2_data':mean_mono_noload_paretofront[:,1],'y2err_data':std_mono_noload_paretofront[:,1],
          'color_2':mycolors['pastel blue'],'color_1':mycolors['olympic blue'],
          'legend_2':'monoarticular, noload','legend_1':'regenerated monoarticular, noload',
          }
plt.subplot(2,2,4)
utils.plot_pareto_avg_curve (plot_dic,loadcond='noload',line=True)
plt.title('noload: monoarticular regenerated')
plt.tick_params(axis='both',direction='in')
ax = plt.gca()
ax.set_xticks([5, 10, 15, 20, 25, 30])
ax.set_yticks([1, 1.5, 2, 2.5,3.10])
ax.set_xlabel('metabolic cost\nreduction (%)')
utils.no_top_right(ax)
plt.legend(loc='best',frameon=False)
fig.tight_layout(h_pad=-1, w_pad=-1.5)
fig.subplots_adjust(top=0.98, bottom=0.075, left=0.100, right=0.975,hspace=0.25,wspace=0.15)
fig.savefig('./Figures/Paretofront/Mean_Pareto/PaperFigure_Main_Pareto_RegeneratedVsNonRegenerated.pdf',orientation='landscape',bbox_inches='tight')
plt.show()
