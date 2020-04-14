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
mean_bi_noload_hip_regen_energy, std_bi_noload_hip_regen_energy = utils.pareto_avg_std_energy(-assisted_energy_dataset['biarticular_pareto_noload_hipregenrative_energy']+\
                                                                                                  assisted_energy_dataset['biarticular_pareto_noload_hipactuator_energy'],reshape=True)
mean_bi_noload_knee_regen_energy, std_bi_noload_knee_regen_energy = utils.pareto_avg_std_energy(-assisted_energy_dataset['biarticular_pareto_noload_kneeregenrative_energy']+\
                                                                                                    assisted_energy_dataset['biarticular_pareto_noload_kneeactuator_energy'],reshape=True)
mean_bi_noload_regen_energy, std_bi_noload_regen_energy = utils.pareto_avg_std_energy(-assisted_energy_dataset['biarticular_pareto_noload_kneeregenrative_energy']-\
                                                                                          assisted_energy_dataset['biarticular_pareto_noload_hipregenrative_energy']+\
                                                                                          assisted_energy_dataset['biarticular_pareto_noload_kneeactuator_energy']+\
                                                                                          assisted_energy_dataset['biarticular_pareto_noload_hipactuator_energy'],reshape=True)
# noload mono
mean_mono_noload_hip_regen_energy, std_mono_noload_hip_regen_energy = utils.pareto_avg_std_energy(-assisted_energy_dataset['monoarticular_pareto_noload_hipregenrative_energy']+\
                                                                                                  assisted_energy_dataset['monoarticular_pareto_noload_hipactuator_energy'],reshape=True)
mean_mono_noload_knee_regen_energy, std_mono_noload_knee_regen_energy = utils.pareto_avg_std_energy(-assisted_energy_dataset['monoarticular_pareto_noload_kneeregenrative_energy']+\
                                                                                                    assisted_energy_dataset['monoarticular_pareto_noload_kneeactuator_energy'],reshape=True)
mean_mono_noload_regen_energy, std_mono_noload_regen_energy = utils.pareto_avg_std_energy(-assisted_energy_dataset['monoarticular_pareto_noload_kneeregenrative_energy']-\
                                                                                          assisted_energy_dataset['monoarticular_pareto_noload_hipregenrative_energy']+\
                                                                                          assisted_energy_dataset['monoarticular_pareto_noload_kneeactuator_energy']+\
                                                                                          assisted_energy_dataset['monoarticular_pareto_noload_hipactuator_energy'],reshape=True)

#####################################################################################
#####################################################################################

# Processing Data For Adding Mass
biarticular_exoskeleton_dic = {'m_waist':6, 'm_thigh':1, 'm_shank':0.9, 'motor_max_torque':2, 'motor_inertia':0.000506, 'thigh_com':0.23, 'shank_com':0.18, 'leg_inertia':2.52}
monoarticular_exoskeleton_dic = {'m_waist':3, 'm_thigh':4, 'm_shank':0.9, 'motor_max_torque':2, 'motor_inertia':0.000506, 'thigh_com':0.3, 'shank_com':0.18, 'leg_inertia':2.52}
biarticular_out = utils.addingmass_metabolics_pareto(unassisted_energy_dataset['noload_metabolics_energy'],bi_noload_metabolics,biarticular_exoskeleton_dic)
monoarticular_out = utils.addingmass_metabolics_pareto(unassisted_energy_dataset['noload_metabolics_energy'],mono_noload_metabolics,monoarticular_exoskeleton_dic)
# Metabolic cost reduction after adding mass
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
utils.plot_pareto_avg_curve (plot_dic,loadcond='noload')
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

# subjects pareto curve: noload mono vs biarticular mass added

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

# subjects pareto curve: biarticular mass added vs biarticular noload

plot_dic = {'x1_data':bi_noload_metabolics_addedmass_percent,'x2_data':bi_noload_metabolics_percent,
          'y1_data':bi_noload_energy,'y2_data':bi_noload_energy,
          'color_1':mycolors['burgundy red'],'color_2':mycolors['gold'],
          'ylabel':'Energy Consumption (W/kg)','xlabel':'Metabolic Reduction (%)',
          'legend_1': 'bi non-ideal','legend_2': 'bi ideal'}
fig = plt.figure(num='Pareto Curve: noload bi non-ideal vs ideal',figsize=(10.4, 18.8))
utils.plot_pareto_curve_subjects (nrows=7,ncols=3,nplot=21,plot_dic=plot_dic,loadcond='noload')
fig.tight_layout()
fig.savefig('./Figures/Pareto/Adding_Mass_Pareto/Pareto_Noload_Subjects_Bi_MassAddedVSIdeal.pdf',orientation='landscape',bbox_inches='tight')
plt.show()

# subjects pareto curve: monoarticular mass added vs monoarticular noload

plot_dic = {'x1_data':mono_noload_metabolics_addedmass_percent,'x2_data':mono_noload_metabolics_percent,
          'y1_data':mono_noload_energy,'y2_data':mono_noload_energy,
          'color_1':mycolors['royal blue'],'color_2':mycolors['lavender purple'],
          'ylabel':'Energy Consumption (W/kg)','xlabel':'Metabolic Reduction (%)',
          'legend_1': 'mono non-ideal','legend_2': 'mono ideal'}
fig = plt.figure(num='Pareto Curve: noload mono  non-ideal vs ideal',figsize=(10.4, 18.8))
utils.plot_pareto_curve_subjects (nrows=7,ncols=3,nplot=21,plot_dic=plot_dic,loadcond='noload')
fig.tight_layout()
fig.savefig('./Figures/Pareto/Adding_Mass_Pareto/Pareto_Noload_Subjects_Mono_MassAddedVSIdeal.pdf',orientation='landscape',bbox_inches='tight')
plt.show()

# average pareto curve: noload mono vs biarticular mass added

plot_dic = {'x1_data':mean_bi_noload_metabolics_addedmass_percent,'x1err_data':std_bi_noload_metabolics_addedmass_percent,
          'x2_data':mean_mono_noload_metabolics_addedmass_percent,'x2err_data':std_mono_noload_metabolics_addedmass_percent,
          'y1_data':mean_bi_noload_energy,'y1err_data':std_bi_noload_energy,
          'y2_data':mean_mono_noload_energy,'y2err_data':std_mono_noload_energy,
          'color_1':mycolors['burgundy red'],'color_2':mycolors['royal blue']
          }
fig = plt.figure(num='Pareto Curve: noload mono vs bi',figsize=(10.4, 8.8))
utils.plot_pareto_avg_curve (plot_dic,loadcond='noload')
plt.xlabel('Metabolic Cost Reduction (%)')
plt.ylabel('Exoskeleton Energy Consumption (W/kg)')
ax = plt.gca()
utils.no_top_right(ax)
plt.legend(loc='best',frameon=False)
fig.tight_layout()
fig.savefig('./Figures/Pareto/Adding_Mass_Pareto/Pareto_Noload_BiVsMono_MassAdded.pdf',orientation='landscape',bbox_inches='tight')
plt.show()

# average pareto curve: biarticular ideal vs mass added

plot_dic = {'x1_data':mean_bi_noload_metabolics_addedmass_percent,'x1err_data':std_bi_noload_metabolics_addedmass_percent,
          'x2_data':mean_bi_noload_metabolics_percent,'x2err_data':std_bi_noload_metabolics_percent,
          'y1_data':mean_bi_noload_energy,'y1err_data':std_bi_noload_energy,
          'y2_data':mean_bi_noload_energy,'y2err_data':std_bi_noload_energy,
          'color_1':mycolors['burgundy red'],'color_2':mycolors['gold'],
          'legend_1': 'bi non-ideal','legend_2': 'bi ideal'}

fig = plt.figure(num='Pareto Curve: noload bi non-ideal vs ideal',figsize=(10.4, 8.8))
utils.plot_pareto_avg_curve (plot_dic,loadcond='noload')
plt.xlabel('Metabolic Cost Reduction (%)')
plt.ylabel('Exoskeleton Energy Consumption (W/kg)')
ax = plt.gca()
utils.no_top_right(ax)
plt.legend(loc='best',frameon=False)
fig.tight_layout()
fig.savefig('./Figures/Pareto/Adding_Mass_Pareto/Pareto_Noload_Bi_NonidealVsIdeal.pdf',orientation='landscape',bbox_inches='tight')
plt.show()

# average pareto curve: monoarticular ideal vs mass added

plot_dic = {'x1_data':mean_mono_noload_metabolics_addedmass_percent,'x1err_data':std_mono_noload_metabolics_addedmass_percent,
          'x2_data':mean_mono_noload_metabolics_percent,'x2err_data':std_mono_noload_metabolics_percent,
          'y1_data':mean_mono_noload_energy,'y1err_data':std_mono_noload_energy,
          'y2_data':mean_mono_noload_energy,'y2err_data':std_mono_noload_energy,
          'color_1':mycolors['royal blue'],'color_2':mycolors['lavender purple'],
          'legend_1': 'mono non-ideal','legend_2': 'mono ideal'}
        
fig = plt.figure(num='Pareto Curve: noload mono non-ideal vs ideal',figsize=(10.4, 8.8))
utils.plot_pareto_avg_curve (plot_dic,loadcond='noload')
plt.xlabel('Metabolic Cost Reduction (%)')
plt.ylabel('Exoskeleton Energy Consumption (W/kg)')
ax = plt.gca()
utils.no_top_right(ax)
plt.legend(loc='best',frameon=False)
fig.tight_layout()
fig.savefig('./Figures/Pareto/Adding_Mass_Pareto/Pareto_Noload_Mono_NonidealVsIdeal.pdf',orientation='landscape',bbox_inches='tight')
plt.show()

#####################################################################################
# plots with regenration and mass

# subjects pareto curve: noload mono vs biarticular mass added

plot_dic = {'x1_data':bi_noload_metabolics_addedmass_percent,'x2_data':mono_noload_metabolics_addedmass_percent,
          'y1_data':bi_noload_regen_energy,'y2_data':mono_noload_regen_energy,
          'color_1':mycolors['burgundy red'],'color_2':mycolors['royal blue'],
          'ylabel':'Energy Consumption (W/kg)','xlabel':'Metabolic Reduction (%)',
          'legend_1':'bi-regenerated','legend_2':'mono-regenerated'
          }
fig = plt.figure(num='Pareto Curve: noload mono vs bi',figsize=(10.4, 18.8))
utils.plot_pareto_curve_subjects (nrows=7,ncols=3,nplot=21,plot_dic=plot_dic,loadcond='noload')
fig.tight_layout()
fig.savefig('./Figures/Pareto/Adding_Mass_Pareto/Pareto_Noload_Subjects_BiVsMono_MassAdded_Regenerated.pdf',orientation='landscape',bbox_inches='tight')
plt.show()

# subjects pareto curve: biarticular mass added vs biarticular noload

plot_dic = {'x1_data':bi_noload_metabolics_addedmass_percent,'x2_data':bi_noload_metabolics_addedmass_percent,
          'y1_data':bi_noload_regen_energy,'y2_data':bi_noload_energy,
          'color_1':mycolors['burgundy red'],'color_2':mycolors['gold'],
          'ylabel':'Energy Consumption (W/kg)','xlabel':'Metabolic Reduction (%)',
          'legend_1': 'bi regenerated','legend_2': 'bi non-regenerated'}
fig = plt.figure(num='Pareto Curve: noload bi regenerated vs no regeneration',figsize=(10.4, 18.8))
utils.plot_pareto_curve_subjects (nrows=7,ncols=3,nplot=21,plot_dic=plot_dic,loadcond='noload')
fig.tight_layout()
fig.savefig('./Figures/Pareto/Adding_Mass_Pareto/Pareto_Noload_Subjects_Bi_GenVSNoGen.pdf',orientation='landscape',bbox_inches='tight')
plt.show()

# subjects pareto curve: monoarticular mass added vs monoarticular noload

plot_dic = {'x1_data':mono_noload_metabolics_addedmass_percent,'x2_data':mono_noload_metabolics_addedmass_percent,
          'y1_data':mono_noload_regen_energy,'y2_data':mono_noload_energy,
          'color_1':mycolors['royal blue'],'color_2':mycolors['lavender purple'],
          'ylabel':'Energy Consumption (W/kg)','xlabel':'Metabolic Reduction (%)',
          'legend_1': 'mono non-regenerated','legend_2': 'mono regenerated'}
fig = plt.figure(num='Pareto Curve: noload mono  non-ideal vs ideal',figsize=(10.4, 18.8))
utils.plot_pareto_curve_subjects (nrows=7,ncols=3,nplot=21,plot_dic=plot_dic,loadcond='noload')
fig.tight_layout()
fig.savefig('./Figures/Pareto/Adding_Mass_Pareto/Pareto_Noload_Subjects_Mono_GenVsNonGen.pdf',orientation='landscape',bbox_inches='tight')
plt.show()


# average pareto curve: biarticular ideal vs mass added

plot_dic = {'x1_data':mean_bi_noload_metabolics_addedmass_percent,'x1err_data':std_bi_noload_metabolics_addedmass_percent,
          'x2_data':mean_bi_noload_metabolics_addedmass_percent,'x2err_data':std_bi_noload_metabolics_addedmass_percent,
          'y1_data':mean_bi_noload_energy,'y1err_data':std_bi_noload_energy,
          'y2_data':mean_bi_noload_regen_energy,'y2err_data':std_bi_noload_regen_energy,
          'color_1':mycolors['burgundy red'],'color_2':mycolors['gold'],
          'legend_1': 'bi no-regenrated','legend_2': 'bi regenrated'}

fig = plt.figure(num='Pareto Curve: noload bi non-regenerated vs regenerated',figsize=(10.4, 8.8))
utils.plot_pareto_avg_curve (plot_dic,loadcond='noload')
plt.xlabel('Metabolic Cost Reduction (%)')
plt.ylabel('Exoskeleton Energy Consumption (W/kg)')
ax = plt.gca()
utils.no_top_right(ax)
plt.legend(loc='best',frameon=False)
fig.tight_layout()
fig.savefig('./Figures/Pareto/Adding_Mass_Pareto/Pareto_Noload_Bi_NoGenVsGen.pdf',orientation='landscape',bbox_inches='tight')
plt.show()

# average pareto curve: monoarticular ideal vs mass added

plot_dic = {'x1_data':mean_mono_noload_metabolics_addedmass_percent,'x1err_data':std_mono_noload_metabolics_addedmass_percent,
          'x2_data':mean_mono_noload_metabolics_addedmass_percent,'x2err_data':std_mono_noload_metabolics_addedmass_percent,
          'y1_data':mean_mono_noload_energy,'y1err_data':std_mono_noload_energy,
          'y2_data':mean_mono_noload_regen_energy,'y2err_data':std_mono_noload_regen_energy,
          'color_1':mycolors['royal blue'],'color_2':mycolors['lavender purple'],
          'legend_1': 'mono non-regenerated','legend_2': 'mono regenerated'}
        
fig = plt.figure(num='Pareto Curve: noload mono non-regenerated vs regenerated',figsize=(10.4, 8.8))
utils.plot_pareto_avg_curve (plot_dic,loadcond='noload')
plt.xlabel('Metabolic Cost Reduction (%)')
plt.ylabel('Exoskeleton Energy Consumption (W/kg)')
ax = plt.gca()
utils.no_top_right(ax)
plt.legend(loc='best',frameon=False)
fig.tight_layout()
fig.savefig('./Figures/Pareto/Adding_Mass_Pareto/Pareto_Noload_Mono_NonGenVsGen.pdf',orientation='landscape',bbox_inches='tight')
plt.show()

#####################################################################################
# All average paretos


x1_data = np.reshape(np.concatenate((mean_bi_noload_metabolics_percent,mean_bi_noload_metabolics_addedmass_percent,mean_bi_noload_metabolics_addedmass_percent),axis=0),(25,3),order='F')
x2_data = np.reshape(np.concatenate((mean_mono_noload_metabolics_percent,mean_mono_noload_metabolics_addedmass_percent,mean_mono_noload_metabolics_addedmass_percent),axis=0),(25,3),order='F')
y1_data = np.reshape(np.concatenate((mean_bi_noload_energy,mean_bi_noload_energy,mean_bi_noload_regen_energy),axis=0),(25,3),order='F')
y2_data = np.reshape(np.concatenate((mean_mono_noload_energy,mean_mono_noload_energy,mean_mono_noload_regen_energy),axis=0),(25,3),order='F')
color_1 = mycolors['magenta pink']
color_2 = mycolors['lavender purple']
plot_titles = ['noload, ideal','noload, exo mass','noload, exo mass + renegeration']

fig = plt.figure(num='Pareto Curve: noload mono vs bi',figsize=(14.4, 10.8))
nplot=3
ncols=3
nrows=1
labels = np.arange(1,26,1)
for i in range(nplot):
    plt.subplot(nrows,ncols,i+1)
    plt.scatter(x1_data[:,i],y1_data[:,i],marker="o",color=color_1,label='biarticular')
    plt.scatter(x2_data[:,i],y2_data[:,i],marker="v",color=color_2,label='monoarticular')
    utils.label_datapoints(x1_data[:,i],y1_data[:,i],labels)
    utils.label_datapoints(x2_data[:,i],y2_data[:,i],labels,ha='left')
    plt.title(plot_titles[i])
    ax = plt.gca()
    plt.tight_layout()
    utils.no_top_right(ax)
    plt.legend(loc='best',frameon=False)
    plt.xlabel('Metabolic Reduction (%)')
    plt.ylabel('Energy Consumption (W/kg)')
    plt.xlim((8,18))
    plt.ylim((0.5,3.5))
fig.tight_layout(h_pad=-1.5, w_pad=-1.5)
fig.subplots_adjust(top=0.99, bottom=0.075, left=0.100, right=0.975,hspace=0.2,wspace=0.15)
fig.savefig('./Figures/Pareto/Adding_Mass_Pareto/Avg_Paretos.pdf',orientation='landscape',bbox_inches='tight')
plt.show()
