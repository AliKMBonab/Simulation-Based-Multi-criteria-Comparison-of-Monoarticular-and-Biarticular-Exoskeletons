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
jrf_dataset = utils.csv2numpy('./Data/RRA/jrf_final_data.csv') 
ideal_jrf_dataset = utils.csv2numpy('./Data/Ideal/jrf_ideal_exo_data.csv') 
# pareto exo torque dataset
directory = './Data/Pareto/*_reaction_moments.csv'
files = enumerate(glob.iglob(directory), 1)
pareto_jrf_dataset = {pathlib.PurePath(f[1]).stem: np.loadtxt(f[1], delimiter=',') for f in files}
bi_noload_RM_dictionary = utils.clasify_data(pareto_jrf_dataset['biarticular_paretofront_noload_reaction_moments'],loadcondition='noload',pareto=True,device='biarticular')
bi_loaded_RM_dictionary = utils.clasify_data(pareto_jrf_dataset['biarticular_paretofront_loaded_reaction_moments'],loadcondition='loaded',pareto=True,device='biarticular')
mono_noload_RM_dictionary = utils.clasify_data(pareto_jrf_dataset['monoarticular_paretofront_noload_reaction_moments'],loadcondition='noload',pareto=True,device='monoarticular')
mono_loaded_RM_dictionary = utils.clasify_data(pareto_jrf_dataset['monoarticular_paretofront_loaded_reaction_moments'],loadcondition='loaded',pareto=True,device='monoarticular')
# pareto exo force dataset
directory = './Data/Pareto/*_reaction_forces.csv'
files = enumerate(glob.iglob(directory), 1)
pareto_jrf_dataset = {pathlib.PurePath(f[1]).stem: np.loadtxt(f[1], delimiter=',') for f in files}
bi_noload_RF_dictionary = utils.clasify_data(pareto_jrf_dataset['biarticular_paretofront_noload_reaction_forces'],loadcondition='noload',pareto=True,device='biarticular',forces_name=['Fx','Fy','Fz'])
bi_loaded_RF_dictionary = utils.clasify_data(pareto_jrf_dataset['biarticular_paretofront_loaded_reaction_forces'],loadcondition='loaded',pareto=True,device='biarticular',forces_name=['Fx','Fy','Fz'])
mono_noload_RF_dictionary = utils.clasify_data(pareto_jrf_dataset['monoarticular_paretofront_noload_reaction_forces'],loadcondition='noload',pareto=True,device='monoarticular',forces_name=['Fx','Fy','Fz'])
mono_loaded_RF_dictionary = utils.clasify_data(pareto_jrf_dataset['monoarticular_paretofront_loaded_reaction_forces'],loadcondition='loaded',pareto=True,device='monoarticular',forces_name=['Fx','Fy','Fz'])

# gls
gl_noload = {'noload_subject{}_trial{}'.format(i,j): utils.construct_gl_mass_side(subjectno=i,trialno=j,loadcond='noload') for i in subjects for j in trials_num}
gl_loaded = {'loaded_subject{}_trial{}'.format(i,j): utils.construct_gl_mass_side(subjectno=i,trialno=j,loadcond='loaded') for i in subjects for j in trials_num}
#####################################################################################
# Processing Data
# toe-off
noload_mean_toe_off,_,loaded_mean_toe_off,_= utils.toe_off_avg_std(gl_noload,gl_loaded)
# indices
bi_loaded_indices = np.array([25,24,23,22,21,17,16,13,12,11,6,1])
mono_loaded_indices = np.array([25,20,15,10,5,4,3,2,1])
bi_noload_indices = np.array([25,24,23,22,21,19,18,17,13,12,11,1])
mono_noload_indices = np.array([25,20,15,14,13,8,7,6,2,1])
#******************************
# knee joint MX
# bi
mean_bi_loaded_knee_joint_jrf_Mx, std_bi_loaded_knee_joint_jrf_Mx = utils.pareto_profiles_avg_std(bi_loaded_RM_dictionary['knee_joint_Mx'],gl_noload,simulation_num=len(bi_loaded_indices),change_direction=False)
mean_bi_noload_knee_joint_jrf_Mx, std_bi_noload_knee_joint_jrf_Mx = utils.pareto_profiles_avg_std(bi_noload_RM_dictionary['knee_joint_Mx'],gl_noload,simulation_num=len(bi_noload_indices),change_direction=False)
# mono
mean_mono_loaded_knee_joint_jrf_Mx, std_mono_loaded_knee_joint_jrf_Mx = utils.pareto_profiles_avg_std(mono_loaded_RM_dictionary['knee_joint_Mx'],gl_noload,simulation_num=len(mono_loaded_indices),change_direction=False)
mean_mono_noload_knee_joint_jrf_Mx, std_mono_noload_knee_joint_jrf_Mx = utils.pareto_profiles_avg_std(mono_noload_RM_dictionary['knee_joint_Mx'],gl_noload,simulation_num=len(mono_noload_indices),change_direction=False)
# knee joint MY
# bi
mean_bi_loaded_knee_joint_jrf_My, std_bi_loaded_knee_joint_jrf_My = utils.pareto_profiles_avg_std(bi_loaded_RM_dictionary['knee_joint_My'],gl_noload,simulation_num=len(bi_loaded_indices),change_direction=False)
mean_bi_noload_knee_joint_jrf_My, std_bi_noload_knee_joint_jrf_My = utils.pareto_profiles_avg_std(bi_noload_RM_dictionary['knee_joint_My'],gl_noload,simulation_num=len(bi_noload_indices),change_direction=False)
# mono
mean_mono_loaded_knee_joint_jrf_My, std_mono_loaded_knee_joint_jrf_My = utils.pareto_profiles_avg_std(mono_loaded_RM_dictionary['knee_joint_My'],gl_noload,simulation_num=len(mono_loaded_indices),change_direction=False)
mean_mono_noload_knee_joint_jrf_My, std_mono_noload_knee_joint_jrf_My = utils.pareto_profiles_avg_std(mono_noload_RM_dictionary['knee_joint_My'],gl_noload,simulation_num=len(mono_noload_indices),change_direction=False)
# knee joint MZ
# bi
mean_bi_loaded_knee_joint_jrf_Mz, std_bi_loaded_knee_joint_jrf_Mz = utils.pareto_profiles_avg_std(bi_loaded_RM_dictionary['knee_joint_Mz'],gl_noload,simulation_num=len(bi_loaded_indices),change_direction=False)
mean_bi_noload_knee_joint_jrf_Mz, std_bi_noload_knee_joint_jrf_Mz = utils.pareto_profiles_avg_std(bi_noload_RM_dictionary['knee_joint_Mz'],gl_noload,simulation_num=len(bi_noload_indices),change_direction=False)
# mono
mean_mono_loaded_knee_joint_jrf_Mz, std_mono_loaded_knee_joint_jrf_Mz = utils.pareto_profiles_avg_std(mono_loaded_RM_dictionary['knee_joint_Mz'],gl_noload,simulation_num=len(mono_loaded_indices),change_direction=False)
mean_mono_noload_knee_joint_jrf_Mz, std_mono_noload_knee_joint_jrf_Mz = utils.pareto_profiles_avg_std(mono_noload_RM_dictionary['knee_joint_Mz'],gl_noload,simulation_num=len(mono_noload_indices),change_direction=False)
#####################################################################################
# profile plots
#************************************************************************************
# torque profile
plt.rcParams.update({'font.size': 14})
fig, axes = plt.subplots(nrows=4,ncols=3,num='Pareto Curve: loaded mono vs bi',figsize=(12.6, 14.8))
# biarticular loaded Mx
plot_dic = {'data':utils.smooth(mean_bi_loaded_knee_joint_jrf_Mx,5,multidim=True),
            'joint_data':utils.smooth(jrf_dataset['mean_loaded_kneejoint_RMx'],5),'joint_color':'k',
            'avg_toeoff':loaded_mean_toe_off,'indices':bi_loaded_indices,'title':'knee Mx,\n loaded biarticular',
            'ideal_data':ideal_jrf_dataset['mean_bi_loaded_knee_RMx'],'ideal_color':mycolors['crimson red']}
ax = plt.subplot(4,3,1)
utils.plot_paretofront_profile_changes(plot_dic,colormap='tab20',include_colorbar=False,
                                       toeoff_color='k',add_ideal_profile=True,ylabel='moment (N.m/kg)')
ax.set_yticks([-0.6,-0.4,-0.2,0,0.2,0.4,0.6])
ax.set_ylim((-0.6,0.6))
# biarticular loaded My
plot_dic = {'data':utils.smooth(mean_bi_loaded_knee_joint_jrf_My,5,multidim=True),
            'joint_data':utils.smooth(jrf_dataset['mean_loaded_kneejoint_RMy'],5),'joint_color':'k',
            'avg_toeoff':loaded_mean_toe_off,'indices':bi_loaded_indices,'title':'knee My,\n loaded biarticular',
            'ideal_data':ideal_jrf_dataset['mean_bi_loaded_knee_RMy'],'ideal_color':mycolors['crimson red']}
ax = plt.subplot(4,3,2)
utils.plot_paretofront_profile_changes(plot_dic,colormap='tab20',include_colorbar=False,
                                       toeoff_color='k',add_ideal_profile=True)
ax.set_yticks([-0.2,0,0.2,0.4])
ax.set_ylim((-0.2,0.4))
# biarticular loaded Mz
plot_dic = {'data':utils.smooth(mean_bi_loaded_knee_joint_jrf_Mz,5,multidim=True),
            'joint_data':utils.smooth(jrf_dataset['mean_loaded_kneejoint_RMz'],5),'joint_color':'k',
            'avg_toeoff':loaded_mean_toe_off,'indices':bi_loaded_indices,'title':'knee Mz,\n loaded biarticular',
            'ideal_data':ideal_jrf_dataset['mean_bi_loaded_knee_RMz'],'ideal_color':mycolors['crimson red']}
ax = plt.subplot(4,3,3)
utils.plot_paretofront_profile_changes(plot_dic,colormap='tab20',include_colorbar=True,adjust_axes=True,
                                       toeoff_color='k',add_ideal_profile=True)
ax.set_yticks([-0.8,-0.6,-0.4,-0.2,0])
ax.set_ylim((-0.85,0))

#****************************************
#****************************************

# monoarticular loaded Mx
plot_dic = {'data':utils.smooth(mean_mono_loaded_knee_joint_jrf_Mx,5,multidim=True),
            'joint_data':utils.smooth(jrf_dataset['mean_loaded_kneejoint_RMx'],5),'joint_color':'k',
            'avg_toeoff':loaded_mean_toe_off,'indices':mono_loaded_indices,'title':'knee Mx,\n loaded monoarticular',
            'ideal_data':ideal_jrf_dataset['mean_mono_loaded_knee_RMx'],'ideal_color':mycolors['crimson red']}
ax = plt.subplot(4,3,4)
utils.plot_paretofront_profile_changes(plot_dic,colormap='tab20',include_colorbar=False,
                                       toeoff_color='k',add_ideal_profile=True,ylabel='moment (N.m/kg)')
ax.set_yticks([-0.6,-0.4,-0.2,0,0.2,0.4,0.6])
ax.set_ylim((-0.6,0.6))
# monoarticular loaded My
plot_dic = {'data':utils.smooth(mean_mono_loaded_knee_joint_jrf_My,5,multidim=True),
            'joint_data':utils.smooth(jrf_dataset['mean_loaded_kneejoint_RMy'],5),'joint_color':'k',
            'avg_toeoff':loaded_mean_toe_off,'indices':mono_loaded_indices,'title':'knee My,\n loaded monoarticular',
            'ideal_data':ideal_jrf_dataset['mean_mono_loaded_knee_RMy'],'ideal_color':mycolors['crimson red']}
ax = plt.subplot(4,3,5)
utils.plot_paretofront_profile_changes(plot_dic,colormap='tab20',include_colorbar=False,
                                       toeoff_color='k',add_ideal_profile=True)
ax.set_yticks([-0.2,0,0.2,0.4])
ax.set_ylim((-0.2,0.4))
# monoarticular loaded Mz
plot_dic = {'data':utils.smooth(mean_mono_loaded_knee_joint_jrf_Mz,5,multidim=True),
            'joint_data':utils.smooth(jrf_dataset['mean_loaded_kneejoint_RMz'],5),'joint_color':'k',
            'avg_toeoff':loaded_mean_toe_off,'indices':mono_loaded_indices,'title':'knee Mz,\n loaded monoarticular',
            'ideal_data':ideal_jrf_dataset['mean_mono_loaded_knee_RMz'],'ideal_color':mycolors['crimson red']}
ax = plt.subplot(4,3,6)
utils.plot_paretofront_profile_changes(plot_dic,colormap='tab20',include_colorbar=True,adjust_axes=True,
                                       toeoff_color='k',add_ideal_profile=True)
ax.set_yticks([-0.8,-0.6,-0.4,-0.2,0])
ax.set_ylim((-0.85,0))

#****************************************
#****************************************
# biarticular noload Mx
plot_dic = {'data':utils.smooth(mean_bi_noload_knee_joint_jrf_Mx,5,multidim=True),
            'joint_data':utils.smooth(jrf_dataset['mean_noload_kneejoint_RMx'],5),'joint_color':'xkcd:shamrock green',
            'avg_toeoff':noload_mean_toe_off,'indices':bi_noload_indices,'title':'knee Mx,\n noload biarticular',
            'ideal_data':ideal_jrf_dataset['mean_bi_noload_knee_RMx'],'ideal_color':mycolors['french rose']}
ax = plt.subplot(4,3,7)
utils.plot_paretofront_profile_changes(plot_dic,colormap='tab20',include_colorbar=False,
                                       toeoff_color='k',add_ideal_profile=True,ylabel='moment (N.m/kg)')
ax.set_yticks([-0.6,-0.4,-0.2,0,0.2,0.4,0.6])
ax.set_ylim((-0.6,0.6))
# biarticular noload My
plot_dic = {'data':utils.smooth(mean_bi_noload_knee_joint_jrf_My,5,multidim=True),
            'joint_data':utils.smooth(jrf_dataset['mean_noload_kneejoint_RMy'],5),'joint_color':'xkcd:shamrock green',
            'avg_toeoff':noload_mean_toe_off,'indices':bi_noload_indices,'title':'knee My,\n noload biarticular',
            'ideal_data':ideal_jrf_dataset['mean_bi_noload_knee_RMy'],'ideal_color':mycolors['french rose']}
ax = plt.subplot(4,3,8)
utils.plot_paretofront_profile_changes(plot_dic,colormap='tab20',include_colorbar=False,
                                       toeoff_color='k',add_ideal_profile=True)
ax.set_yticks([-0.2,0,0.2,0.4])
ax.set_ylim((-0.2,0.4))
# biarticular noload Mz
plot_dic = {'data':utils.smooth(mean_bi_noload_knee_joint_jrf_Mz,5,multidim=True),
            'joint_data':utils.smooth(jrf_dataset['mean_noload_kneejoint_RMz'],5),'joint_color':'xkcd:shamrock green',
            'avg_toeoff':noload_mean_toe_off,'indices':bi_noload_indices,'title':'knee Mz,\n noload biarticular',
            'ideal_data':ideal_jrf_dataset['mean_bi_noload_knee_RMz'],'ideal_color':mycolors['french rose']}
ax = plt.subplot(4,3,9)
utils.plot_paretofront_profile_changes(plot_dic,colormap='tab20',include_colorbar=True,adjust_axes=True,
                                       toeoff_color='k',add_ideal_profile=True)
ax.set_yticks([-0.8,-0.6,-0.4,-0.2,0])
ax.set_ylim((-0.85,0))

#****************************************
#****************************************

# monoarticular noload Mx
plot_dic = {'data':utils.smooth(mean_mono_noload_knee_joint_jrf_Mx,5,multidim=True),
            'joint_data':utils.smooth(jrf_dataset['mean_noload_kneejoint_RMx'],5),'joint_color':'xkcd:shamrock green',
            'avg_toeoff':noload_mean_toe_off,'indices':mono_noload_indices,'title':'knee Mx,\n noload monoarticular',
            'ideal_data':ideal_jrf_dataset['mean_mono_noload_knee_RMx'],'ideal_color':mycolors['french rose']}
ax = plt.subplot(4,3,10)
utils.plot_paretofront_profile_changes(plot_dic,colormap='tab20',include_colorbar=False,
                                       toeoff_color='k',add_ideal_profile=True,ylabel='moment (N.m/kg)')
ax.set_yticks([-0.6,-0.4,-0.2,0,0.2,0.4,0.6])
ax.set_ylim((-0.6,0.6))
# monoarticular noload My
plot_dic = {'data':utils.smooth(mean_mono_noload_knee_joint_jrf_My,5,multidim=True),
            'joint_data':utils.smooth(jrf_dataset['mean_noload_kneejoint_RMy'],5),'joint_color':'xkcd:shamrock green',
            'avg_toeoff':noload_mean_toe_off,'indices':mono_noload_indices,'title':'knee My,\n noload monoarticular',
            'ideal_data':ideal_jrf_dataset['mean_mono_noload_knee_RMy'],'ideal_color':mycolors['french rose']}
ax = plt.subplot(4,3,11)
utils.plot_paretofront_profile_changes(plot_dic,colormap='tab20',include_colorbar=False,
                                       toeoff_color='k',add_ideal_profile=True)
ax.set_yticks([-0.2,0,0.2,0.4])
ax.set_ylim((-0.2,0.4))
# monoarticular noload Mz
plot_dic = {'data':utils.smooth(mean_mono_noload_knee_joint_jrf_Mz,5,multidim=True),
            'joint_data':utils.smooth(jrf_dataset['mean_noload_kneejoint_RMz'],5),'joint_color':'xkcd:shamrock green',
            'avg_toeoff':noload_mean_toe_off,'indices':mono_noload_indices,'title':'knee Mz,\n noload monoarticular',
            'ideal_data':ideal_jrf_dataset['mean_mono_noload_knee_RMz'],'ideal_color':mycolors['french rose']}
ax = plt.subplot(4,3,12)
utils.plot_paretofront_profile_changes(plot_dic,colormap='tab20',include_colorbar=True,adjust_axes=True,
                                       toeoff_color='k',add_ideal_profile=True)
ax.set_yticks([-0.8,-0.6,-0.4,-0.2,0])
ax.set_ylim((-0.85,0))
fig.tight_layout(h_pad=-1.5, w_pad=-1.5)
fig.subplots_adjust(top=0.98, bottom=0.075, left=0.05, right=0.95,hspace=0.45,wspace=0.05)
fig.savefig('./Figures/Paretofront/Analyses_Pareto/JRFs/Paretofront_JRF_Knee_Moment.pdf',orientation='landscape',bbox_inches='tight')
plt.show()

