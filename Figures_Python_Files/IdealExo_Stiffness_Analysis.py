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
plt.rcParams.update({'font.size': 12})
#####################################################################################
# Reading CSV files into a dictionary and constructing gls
rra_dataset = utils.csv2numpy('./Data/RRA/rra_final_data.csv') 
unassist_dataset = utils.csv2numpy('./Data/Unassist/unassist_final_data.csv') 
stiffness_dataset = utils.csv2numpy('./Data/Unassist/unassist_stiffness_data.csv') 
unnormalized_moment_dataset = utils.csv2numpy('./Data/Unassist/unassist_unnormalized_moment_data.csv') 
# ideal exo torque dataset
directory = './Data/Ideal/*_torque.csv'
files = enumerate(glob.iglob(directory), 1)
exo_torque_dataset = {pathlib.PurePath(f[1]).stem: np.loadtxt(f[1], delimiter=',') for f in files}
# ideal exo torque dataset
directory = './Data/Ideal/*_kinematics.csv'
files = enumerate(glob.iglob(directory), 1)
jointkinematics_dataset = {pathlib.PurePath(f[1]).stem: np.loadtxt(f[1], delimiter=',') for f in files}
# muscles moment dataset
directory = './Data/Ideal/*_musclesmoment.csv'
files = enumerate(glob.iglob(directory), 1)
musclesmoment_dataset = {pathlib.PurePath(f[1]).stem: np.loadtxt(f[1], delimiter=',') for f in files}
# gls
gl_noload = {'noload_subject{}_trial{}'.format(i,j): utils.construct_gl_mass_side(subjectno=i,trialno=j,loadcond='noload') for i in subjects for j in trials_num}
gl_loaded = {'loaded_subject{}_trial{}'.format(i,j): utils.construct_gl_mass_side(subjectno=i,trialno=j,loadcond='loaded') for i in subjects for j in trials_num}
#####################################################################################
# Processing Data
# toe-off
noload_mean_toe_off,_,loaded_mean_toe_off,_ = utils.toe_off_avg_std(gl_noload,gl_loaded)
# exoskeleton torque profiles
# biarticular
# hip
bi_loaded_hip_torque = utils.normalize_direction_data(exo_torque_dataset['biarticular_ideal_loaded_hipactuator_torque'],gl_noload, normalize=False,direction=False)
bi_noload_hip_torque = utils.normalize_direction_data(exo_torque_dataset['biarticular_ideal_noload_hipactuator_torque'],gl_noload, normalize=False,direction=False)
mean_bi_loaded_hip_torque,std_bi_loaded_hip_torque = utils.mean_std_over_subjects(bi_loaded_hip_torque,avg_trials=False)
mean_bi_noload_hip_torque,std_bi_noload_hip_torque = utils.mean_std_over_subjects(bi_noload_hip_torque,avg_trials=False)
# knee
bi_loaded_knee_torque = utils.normalize_direction_data(exo_torque_dataset['biarticular_ideal_loaded_kneeactuator_torque'],gl_noload, normalize=False,direction=True)
bi_noload_knee_torque = utils.normalize_direction_data(exo_torque_dataset['biarticular_ideal_noload_kneeactuator_torque'],gl_noload, normalize=False,direction=True)
mean_bi_loaded_knee_torque,std_bi_loaded_knee_torque = utils.mean_std_over_subjects(bi_loaded_knee_torque,avg_trials=False)
mean_bi_noload_knee_torque,std_bi_noload_knee_torque = utils.mean_std_over_subjects(bi_noload_knee_torque,avg_trials=False)
# monoarticular
# hip
mono_loaded_hip_torque = utils.normalize_direction_data(exo_torque_dataset['monoarticular_ideal_loaded_hipactuator_torque'],gl_noload, normalize=False,direction=False)
mono_noload_hip_torque = utils.normalize_direction_data(exo_torque_dataset['monoarticular_ideal_noload_hipactuator_torque'],gl_noload, normalize=False,direction=False)
mean_mono_loaded_hip_torque,std_mono_loaded_hip_torque = utils.mean_std_over_subjects(mono_loaded_hip_torque,avg_trials=False)
mean_mono_noload_hip_torque,std_mono_noload_hip_torque = utils.mean_std_over_subjects(mono_noload_hip_torque,avg_trials=False)
# knee
mono_loaded_knee_torque = utils.normalize_direction_data(exo_torque_dataset['monoarticular_ideal_loaded_kneeactuator_torque'],gl_noload, normalize=False,direction=True)
mono_noload_knee_torque = utils.normalize_direction_data(exo_torque_dataset['monoarticular_ideal_noload_kneeactuator_torque'],gl_noload, normalize=False,direction=True)
mean_mono_loaded_knee_torque,std_mono_loaded_knee_torque = utils.mean_std_over_subjects(mono_loaded_knee_torque,avg_trials=False)
mean_mono_noload_knee_torque,std_mono_noload_knee_torque = utils.mean_std_over_subjects(mono_noload_knee_torque,avg_trials=False)
#******************************
# hip muscles moment
# biarticular
bi_loaded_hipmuscles_moment = utils.normalize_direction_data(musclesmoment_dataset['biarticular_ideal_loaded_hip_musclesmoment'],gl_noload, normalize=False,direction=True)
bi_noload_hipmuscles_moment = utils.normalize_direction_data(musclesmoment_dataset['biarticular_ideal_noload_hip_musclesmoment'],gl_noload, normalize=False,direction=True)
mean_bi_loaded_hipmuscles_moment,std_bi_loaded_hipmuscles_moment = utils.mean_std_over_subjects(bi_loaded_hipmuscles_moment,avg_trials=False)
mean_bi_noload_hipmuscles_moment,std_bi_noload_hipmuscles_moment = utils.mean_std_over_subjects(bi_noload_hipmuscles_moment,avg_trials=False)
# monoarticular
mono_loaded_hipmuscles_moment = utils.normalize_direction_data(musclesmoment_dataset['monoarticular_ideal_loaded_hip_musclesmoment'],gl_noload, normalize=False,direction=True)
mono_noload_hipmuscles_moment = utils.normalize_direction_data(musclesmoment_dataset['monoarticular_ideal_noload_hip_musclesmoment'],gl_noload, normalize=False,direction=True)
mean_mono_loaded_hipmuscles_moment,std_mono_loaded_hipmuscles_moment = utils.mean_std_over_subjects(mono_loaded_hipmuscles_moment,avg_trials=False)
mean_mono_noload_hipmuscles_moment,std_mono_noload_hipmuscles_moment = utils.mean_std_over_subjects(mono_noload_hipmuscles_moment,avg_trials=False)
# knee muscles moment
# biarticular
bi_loaded_kneemuscles_moment = utils.normalize_direction_data(musclesmoment_dataset['biarticular_ideal_loaded_knee_musclesmoment'],gl_noload, normalize=False,direction=True)
bi_noload_kneemuscles_moment = utils.normalize_direction_data(musclesmoment_dataset['biarticular_ideal_noload_knee_musclesmoment'],gl_noload, normalize=False,direction=True)
mean_bi_loaded_kneemuscles_moment,std_bi_loaded_kneemuscles_moment = utils.mean_std_over_subjects(bi_loaded_kneemuscles_moment,avg_trials=False)
mean_bi_noload_kneemuscles_moment,std_bi_noload_kneemuscles_moment = utils.mean_std_over_subjects(bi_noload_kneemuscles_moment,avg_trials=False)
# monoarticular
mono_loaded_kneemuscles_moment = utils.normalize_direction_data(musclesmoment_dataset['monoarticular_ideal_loaded_knee_musclesmoment'],gl_noload, normalize=False,direction=True)
mono_noload_kneemuscles_moment = utils.normalize_direction_data(musclesmoment_dataset['monoarticular_ideal_noload_knee_musclesmoment'],gl_noload, normalize=False,direction=True)
mean_mono_loaded_kneemuscles_moment,std_mono_loaded_kneemuscles_moment = utils.mean_std_over_subjects(mono_loaded_kneemuscles_moment,avg_trials=False)
mean_mono_noload_kneemuscles_moment,std_mono_noload_kneemuscles_moment = utils.mean_std_over_subjects(mono_noload_kneemuscles_moment,avg_trials=False)
#******************************
# hip kinematics
# biarticular
bi_loaded_hip_kinematics = utils.normalize_direction_data(jointkinematics_dataset['biarticular_ideal_loaded_hip_kinematics'],gl_noload, normalize=False,direction=False)
bi_noload_hip_kinematics = utils.normalize_direction_data(jointkinematics_dataset['biarticular_ideal_noload_hip_kinematics'],gl_noload, normalize=False,direction=False)
mean_bi_loaded_hip_kinematics,std_bi_loaded_hip_kinematics = utils.mean_std_over_subjects(bi_loaded_hip_kinematics,avg_trials=False)
mean_bi_noload_hip_kinematics,std_bi_noload_hip_kinematics = utils.mean_std_over_subjects(bi_noload_hip_kinematics,avg_trials=False)
# monoarticular
mono_loaded_hip_kinematics = utils.normalize_direction_data(jointkinematics_dataset['monoarticular_ideal_loaded_hip_kinematics'],gl_noload, normalize=False,direction=False)
mono_noload_hip_kinematics = utils.normalize_direction_data(jointkinematics_dataset['monoarticular_ideal_noload_hip_kinematics'],gl_noload, normalize=False,direction=False)
mean_mono_loaded_hip_kinematics,std_mono_loaded_hip_kinematics = utils.mean_std_over_subjects(mono_loaded_hip_kinematics,avg_trials=False)
mean_mono_noload_hip_kinematics,std_mono_noload_hip_kinematics = utils.mean_std_over_subjects(mono_noload_hip_kinematics,avg_trials=False)
# knee kinematics
# biarticular
bi_loaded_knee_kinematics = utils.normalize_direction_data(jointkinematics_dataset['biarticular_ideal_loaded_knee_kinematics']-jointkinematics_dataset['biarticular_ideal_loaded_hip_kinematics'],gl_noload, normalize=False,direction=False)
bi_noload_knee_kinematics = utils.normalize_direction_data(jointkinematics_dataset['biarticular_ideal_noload_knee_kinematics']-jointkinematics_dataset['biarticular_ideal_noload_hip_kinematics'],gl_noload, normalize=False,direction=False)
mean_bi_loaded_knee_kinematics,std_bi_loaded_knee_kinematics = utils.mean_std_over_subjects(bi_loaded_knee_kinematics,avg_trials=False)
mean_bi_noload_knee_kinematics,std_bi_noload_knee_kinematics = utils.mean_std_over_subjects(bi_noload_knee_kinematics,avg_trials=False)
# monoarticular
mono_loaded_knee_kinematics = utils.normalize_direction_data(jointkinematics_dataset['monoarticular_ideal_loaded_knee_kinematics'],gl_noload, normalize=False,direction=False)
mono_noload_knee_kinematics = utils.normalize_direction_data(jointkinematics_dataset['monoarticular_ideal_noload_knee_kinematics'],gl_noload, normalize=False,direction=False)
mean_mono_loaded_knee_kinematics,std_mono_loaded_knee_kinematics = utils.mean_std_over_subjects(mono_loaded_knee_kinematics,avg_trials=False)
mean_mono_noload_knee_kinematics,std_mono_noload_knee_kinematics = utils.mean_std_over_subjects(mono_noload_knee_kinematics,avg_trials=False)

#*********************************
# hip muscles and actuators stiffness
#bi_loaded_hip_stiffness_dict, bi_loaded_hip_Rsquare_dict, bi_loaded_hip_bias_dict = utils.calculate_quasi_stiffness(angle=loaded_hipjoint_kinematics,moment=loaded_hipmuscles_moment,toe_off=subjects_loaded_toe_off,joint='hip')

##########################################################################################################################################################
# biarticular loaded vs noload hip joint stiffness
fig = plt.figure(num='Hip Joint Stiffness',figsize=(15, 12))
gridsize = (3, 4)
ax1 = plt.subplot2grid(gridsize, (0, 0), colspan=2, rowspan=2) # stiffness plot
ax2 = plt.subplot2grid(gridsize, (2, 0)) # kinematics plot
ax3 = plt.subplot2grid(gridsize, (2, 1)) # moment plot
ax4 = plt.subplot2grid(gridsize, (0, 2), colspan=2, rowspan=2) # stiffness plot
ax5 = plt.subplot2grid(gridsize, (2, 2)) # kinematics plot
ax6 = plt.subplot2grid(gridsize, (2, 3)) # moment plot
# hip loaded case
hip_stiffness_plot_dic = {'loaded_toe_off':loaded_mean_toe_off,'noload_toe_off':noload_mean_toe_off,'kinematics':utils.smooth(rra_dataset['mean_loaded_hipjoint_kinematics'],5),
                          'kinematics_std':utils.smooth(rra_dataset['std_loaded_hipjoint_kinematics'],5),'moment':utils.smooth(unnormalized_moment_dataset['mean_loaded_hipmuscles_moment'],5),
                          'moment_std':utils.smooth(unnormalized_moment_dataset['std_loaded_hipmuscles_moment'],5),'color':'k','toe_off_color':'grey','label':'hip joint'}
utils.plot_stiffness(plot_dic = hip_stiffness_plot_dic, load_condition='loaded',\
                     kinematics_ticks=[-20,-10,0,10,20,30,40,50],moment_ticks=[-120,-80,-40,0,40,80,120],ax1=ax1,ax2=ax2,ax3=ax3)
# hip noload case
hip_stiffness_plot_dic = {'loaded_toe_off':loaded_mean_toe_off,'noload_toe_off':noload_mean_toe_off,'kinematics':utils.smooth(rra_dataset['mean_noload_hipjoint_kinematics'],5),
                          'kinematics_std':utils.smooth(rra_dataset['std_noload_hipjoint_kinematics'],5),'moment':utils.smooth(unnormalized_moment_dataset['mean_noload_hipmuscles_moment'],5),
                          'moment_std':utils.smooth(unnormalized_moment_dataset['std_noload_hipmuscles_moment'],5),'color':'xkcd:irish green','toe_off_color':'xkcd:shamrock green','label':'hip joint'}
utils.plot_stiffness(plot_dic = hip_stiffness_plot_dic, load_condition='noload',
kinematics_ticks=[-20,-10,0,10,20,30,40,50],moment_ticks=[-120,-80,-40,0,40,80,120],ax1=ax4,ax2=ax5,ax3=ax6)
# biarticular hip loaded case
hip_stiffness_plot_dic = {'loaded_toe_off':loaded_mean_toe_off,'noload_toe_off':noload_mean_toe_off,'kinematics':utils.smooth(mean_bi_loaded_hip_kinematics,5),
                          'kinematics_std':utils.smooth(std_bi_loaded_hip_kinematics,5),'moment':utils.smooth(mean_bi_loaded_hip_torque,5),
                          'moment_std':utils.smooth(std_bi_loaded_hip_torque,5),'color':mycolors['cyan blue'],'toe_off_color':mycolors['cyan blue'],'label':'bi loaded hip actuator'}
utils.plot_stiffness(plot_dic = hip_stiffness_plot_dic, load_condition='loaded',\
                     kinematics_ticks=[-20,-10,0,10,20,30,40,50],moment_ticks=[-120,-80,-40,0,40,80,120],ax1=ax1,ax2=ax2,ax3=ax3)
# biarticular hip noload case
hip_stiffness_plot_dic = {'loaded_toe_off':loaded_mean_toe_off,'noload_toe_off':noload_mean_toe_off,'kinematics':utils.smooth(mean_bi_noload_hip_kinematics,5),
                          'kinematics_std':utils.smooth(std_bi_noload_hip_kinematics,5),'moment':utils.smooth(mean_bi_noload_hip_torque,5),
                          'moment_std':utils.smooth(std_bi_noload_hip_torque,5),'color':mycolors['olympic blue'],'toe_off_color':mycolors['olympic blue'],'label':'bi noload hip actuator'}
utils.plot_stiffness(plot_dic = hip_stiffness_plot_dic, load_condition='noload',
kinematics_ticks=[-20,-10,0,10,20,30,40,50],moment_ticks=[-120,-80,-40,0,40,80,120],ax1=ax4,ax2=ax5,ax3=ax6)
# biarticular hip muscles loaded case
hip_stiffness_plot_dic = {'loaded_toe_off':loaded_mean_toe_off,'noload_toe_off':noload_mean_toe_off,'kinematics':utils.smooth(mean_bi_loaded_hip_kinematics,5),
                          'kinematics_std':utils.smooth(std_bi_loaded_hip_kinematics,5),'moment':utils.smooth(mean_bi_loaded_hipmuscles_moment,5),
                          'moment_std':utils.smooth(std_bi_loaded_hipmuscles_moment,5),'color':mycolors['crimson red'],'toe_off_color':mycolors['crimson red'],'label':'bi loaded hip muscles'}
utils.plot_stiffness(plot_dic = hip_stiffness_plot_dic, load_condition='loaded',\
                     kinematics_ticks=[-20,-10,0,10,20,30,40,50],moment_ticks=[-120,-80,-40,0,40,80,120],ax1=ax1,ax2=ax2,ax3=ax3)
# biarticular hip muscles noload case
hip_stiffness_plot_dic = {'loaded_toe_off':loaded_mean_toe_off,'noload_toe_off':noload_mean_toe_off,'kinematics':utils.smooth(mean_bi_noload_hip_kinematics,5),
                          'kinematics_std':utils.smooth(std_bi_noload_hip_kinematics,5),'moment':utils.smooth(mean_bi_noload_hipmuscles_moment,5),
                          'moment_std':utils.smooth(std_bi_noload_hipmuscles_moment,5),'color':mycolors['french rose'],'toe_off_color':mycolors['french rose'],'label':'bi noload hip muscles'}
utils.plot_stiffness(plot_dic = hip_stiffness_plot_dic, load_condition='noload',
kinematics_ticks=[-20,-10,0,10,20,30,40,50],moment_ticks=[-120,-80,-40,0,40,80,120],ax1=ax4,ax2=ax5,ax3=ax6)

fig.tight_layout(h_pad=-1, w_pad=-1.5)
fig.subplots_adjust(top=0.98, bottom=0.075, left=0.100, right=0.975,hspace=0.45,wspace=0.35)
plt.show()
fig.savefig('./Figures/Ideal/Biarticular_LoadedVsNoload_HipActuator_Stiffness.pdf',orientation='landscape',bbox_inches='tight')
#************************************************************
# monoarticular loaded vs noload hip joint stiffness
fig = plt.figure(num='Hip Joint Stiffness',figsize=(15, 12))
gridsize = (3, 4)
ax1 = plt.subplot2grid(gridsize, (0, 0), colspan=2, rowspan=2) # stiffness plot
ax2 = plt.subplot2grid(gridsize, (2, 0)) # kinematics plot
ax3 = plt.subplot2grid(gridsize, (2, 1)) # moment plot
ax4 = plt.subplot2grid(gridsize, (0, 2), colspan=2, rowspan=2) # stiffness plot
ax5 = plt.subplot2grid(gridsize, (2, 2)) # kinematics plot
ax6 = plt.subplot2grid(gridsize, (2, 3)) # moment plot
# hip loaded case
hip_stiffness_plot_dic = {'loaded_toe_off':loaded_mean_toe_off,'noload_toe_off':noload_mean_toe_off,'kinematics':utils.smooth(rra_dataset['mean_loaded_hipjoint_kinematics'],5),
                          'kinematics_std':utils.smooth(rra_dataset['std_loaded_hipjoint_kinematics'],5),'moment':utils.smooth(unnormalized_moment_dataset['mean_loaded_hipmuscles_moment'],5),
                          'moment_std':utils.smooth(unnormalized_moment_dataset['std_loaded_hipmuscles_moment'],5),'color':'k','toe_off_color':'grey','label':'hip joint'}
utils.plot_stiffness(plot_dic = hip_stiffness_plot_dic, load_condition='loaded',\
                     kinematics_ticks=[-20,-10,0,10,20,30,40,50],moment_ticks=[-120,-80,-40,0,40,80,120],ax1=ax1,ax2=ax2,ax3=ax3)
# hip noload case
hip_stiffness_plot_dic = {'loaded_toe_off':loaded_mean_toe_off,'noload_toe_off':noload_mean_toe_off,'kinematics':utils.smooth(rra_dataset['mean_noload_hipjoint_kinematics'],5),
                          'kinematics_std':utils.smooth(rra_dataset['std_noload_hipjoint_kinematics'],5),'moment':utils.smooth(unnormalized_moment_dataset['mean_noload_hipmuscles_moment'],5),
                          'moment_std':utils.smooth(unnormalized_moment_dataset['std_noload_hipmuscles_moment'],5),'color':'xkcd:irish green','toe_off_color':'xkcd:shamrock green','label':'hip joint'}
utils.plot_stiffness(plot_dic = hip_stiffness_plot_dic, load_condition='noload',
kinematics_ticks=[-20,-10,0,10,20,30,40,50],moment_ticks=[-120,-80,-40,0,40,80,120],ax1=ax4,ax2=ax5,ax3=ax6)
# monoarticular hip loaded case
hip_stiffness_plot_dic = {'loaded_toe_off':loaded_mean_toe_off,'noload_toe_off':noload_mean_toe_off,'kinematics':utils.smooth(mean_mono_loaded_hip_kinematics,5),
                          'kinematics_std':utils.smooth(std_mono_loaded_hip_kinematics,5),'moment':utils.smooth(mean_mono_loaded_hip_torque,5),
                          'moment_std':utils.smooth(std_mono_loaded_hip_torque,5),'color':mycolors['cyan blue'],'toe_off_color':mycolors['cyan blue'],'label':'bi loaded hip actuator'}
utils.plot_stiffness(plot_dic = hip_stiffness_plot_dic, load_condition='loaded',\
                     kinematics_ticks=[-20,-10,0,10,20,30,40,50],moment_ticks=[-120,-80,-40,0,40,80,120],ax1=ax1,ax2=ax2,ax3=ax3)
# monoarticular hip noload case
hip_stiffness_plot_dic = {'loaded_toe_off':loaded_mean_toe_off,'noload_toe_off':noload_mean_toe_off,'kinematics':utils.smooth(mean_mono_noload_hip_kinematics,5),
                          'kinematics_std':utils.smooth(std_mono_noload_hip_kinematics,5),'moment':utils.smooth(mean_mono_noload_hip_torque,5),
                          'moment_std':utils.smooth(std_mono_noload_hip_torque,5),'color':mycolors['olympic blue'],'toe_off_color':mycolors['olympic blue'],'label':'bi noload hip actuator'}
utils.plot_stiffness(plot_dic = hip_stiffness_plot_dic, load_condition='noload',
kinematics_ticks=[-20,-10,0,10,20,30,40,50],moment_ticks=[-120,-80,-40,0,40,80,120],ax1=ax4,ax2=ax5,ax3=ax6)
# monoarticular hip muscles loaded case
hip_stiffness_plot_dic = {'loaded_toe_off':loaded_mean_toe_off,'noload_toe_off':noload_mean_toe_off,'kinematics':utils.smooth(mean_mono_loaded_hip_kinematics,5),
                          'kinematics_std':utils.smooth(std_mono_loaded_hip_kinematics,5),'moment':utils.smooth(mean_mono_loaded_hipmuscles_moment,5),
                          'moment_std':utils.smooth(std_mono_loaded_hipmuscles_moment,5),'color':mycolors['crimson red'],'toe_off_color':mycolors['crimson red'],'label':'bi loaded hip muscles'}
utils.plot_stiffness(plot_dic = hip_stiffness_plot_dic, load_condition='loaded',\
                     kinematics_ticks=[-20,-10,0,10,20,30,40,50],moment_ticks=[-120,-80,-40,0,40,80,120],ax1=ax1,ax2=ax2,ax3=ax3)
# monoarticular hip muscles noload case
hip_stiffness_plot_dic = {'loaded_toe_off':loaded_mean_toe_off,'noload_toe_off':noload_mean_toe_off,'kinematics':utils.smooth(mean_mono_noload_hip_kinematics,5),
                          'kinematics_std':utils.smooth(std_mono_noload_hip_kinematics,5),'moment':utils.smooth(mean_mono_noload_hipmuscles_moment,5),
                          'moment_std':utils.smooth(std_mono_noload_hipmuscles_moment,5),'color':mycolors['french rose'],'toe_off_color':mycolors['french rose'],'label':'bi noload hip muscles'}
utils.plot_stiffness(plot_dic = hip_stiffness_plot_dic, load_condition='noload',
kinematics_ticks=[-20,-10,0,10,20,30,40,50],moment_ticks=[-120,-80,-40,0,40,80,120],ax1=ax4,ax2=ax5,ax3=ax6)

fig.tight_layout(h_pad=-1, w_pad=-1.5)
fig.subplots_adjust(top=0.98, bottom=0.075, left=0.100, right=0.975,hspace=0.45,wspace=0.35)
plt.show()
fig.savefig('./Figures/Ideal/Monoarticular_LoadedVsNoload_HipActuator_Stiffness.pdf',orientation='landscape',bbox_inches='tight')
#************************************************************
#************************************************************
# biarticular loaded vs noload knee joint stiffness
fig = plt.figure(num='Knee Joint Stiffness',figsize=(15, 12))
gridsize = (3, 4)
ax1 = plt.subplot2grid(gridsize, (0, 0), colspan=2, rowspan=2) # stiffness plot
ax2 = plt.subplot2grid(gridsize, (2, 0)) # kinematics plot
ax3 = plt.subplot2grid(gridsize, (2, 1)) # moment plot
ax4 = plt.subplot2grid(gridsize, (0, 2), colspan=2, rowspan=2) # stiffness plot
ax5 = plt.subplot2grid(gridsize, (2, 2)) # kinematics plot
ax6 = plt.subplot2grid(gridsize, (2, 3)) # moment plot
# knee loaded case
knee_stiffness_plot_dic = {'loaded_toe_off':loaded_mean_toe_off,'noload_toe_off':noload_mean_toe_off,'kinematics':utils.smooth(rra_dataset['mean_loaded_kneejoint_kinematics'],5),
                          'kinematics_std':utils.smooth(rra_dataset['std_loaded_kneejoint_kinematics'],5),'moment':utils.smooth(unnormalized_moment_dataset['mean_loaded_kneemuscles_moment'],5),
                          'moment_std':utils.smooth(unnormalized_moment_dataset['std_loaded_kneemuscles_moment'],5),'color':'k','toe_off_color':'grey','label':'knee joint'}
utils.plot_stiffness(plot_dic = knee_stiffness_plot_dic, load_condition='loaded',\
                     kinematics_ticks=[-40,-20,0,20,40,60,80],moment_ticks=[-160,-120,-80,-40,0,40,80,120],ax1=ax1,ax2=ax2,ax3=ax3)
# knee noload case
knee_stiffness_plot_dic = {'loaded_toe_off':loaded_mean_toe_off,'noload_toe_off':noload_mean_toe_off,'kinematics':utils.smooth(rra_dataset['mean_noload_kneejoint_kinematics'],5),
                          'kinematics_std':utils.smooth(rra_dataset['std_noload_kneejoint_kinematics'],5),'moment':utils.smooth(unnormalized_moment_dataset['mean_noload_kneemuscles_moment'],5),
                          'moment_std':utils.smooth(unnormalized_moment_dataset['std_noload_kneemuscles_moment'],5),'color':'xkcd:irish green','toe_off_color':'xkcd:shamrock green','label':'knee joint'}
utils.plot_stiffness(plot_dic = knee_stiffness_plot_dic, load_condition='noload',
kinematics_ticks=[-40,-20,0,20,40,60,80],moment_ticks=[-160,-120,-80,-40,0,40,80,120],ax1=ax4,ax2=ax5,ax3=ax6)
# biarticular knee loaded case
knee_stiffness_plot_dic = {'loaded_toe_off':loaded_mean_toe_off,'noload_toe_off':noload_mean_toe_off,'kinematics':utils.smooth(mean_bi_loaded_knee_kinematics,5),
                          'kinematics_std':utils.smooth(std_bi_loaded_knee_kinematics,5),'moment':utils.smooth(mean_bi_loaded_knee_torque,5),
                          'moment_std':utils.smooth(std_bi_loaded_knee_torque,5),'color':mycolors['cyan blue'],'toe_off_color':mycolors['cyan blue'],'label':'bi loaded knee actuator'}
utils.plot_stiffness(plot_dic = knee_stiffness_plot_dic, load_condition='loaded',\
                     kinematics_ticks=[-40,-20,0,20,40,60,80],moment_ticks=[-160,-120,-80,-40,0,40,80,120],ax1=ax1,ax2=ax2,ax3=ax3)
# biarticular knee noload case
knee_stiffness_plot_dic = {'loaded_toe_off':loaded_mean_toe_off,'noload_toe_off':noload_mean_toe_off,'kinematics':utils.smooth(mean_bi_noload_knee_kinematics,5),
                          'kinematics_std':utils.smooth(std_bi_noload_knee_kinematics,5),'moment':utils.smooth(mean_bi_noload_knee_torque,5),
                          'moment_std':utils.smooth(std_bi_noload_knee_torque,5),'color':mycolors['olympic blue'],'toe_off_color':mycolors['olympic blue'],'label':'bi noload knee actuator'}
utils.plot_stiffness(plot_dic = knee_stiffness_plot_dic, load_condition='noload',
kinematics_ticks=[-40,-20,0,20,40,60,80],moment_ticks=[-160,-120,-80,-40,0,40,80,120],ax1=ax4,ax2=ax5,ax3=ax6)
# biarticular knee muscles loaded case
knee_stiffness_plot_dic = {'loaded_toe_off':loaded_mean_toe_off,'noload_toe_off':noload_mean_toe_off,'kinematics':utils.smooth(mean_mono_loaded_knee_kinematics,5),
                          'kinematics_std':utils.smooth(std_mono_loaded_knee_kinematics,5),'moment':utils.smooth(mean_bi_loaded_kneemuscles_moment,5),
                          'moment_std':utils.smooth(std_bi_loaded_kneemuscles_moment,5),'color':mycolors['crimson red'],'toe_off_color':mycolors['crimson red'],'label':'bi loaded knee muscles'}
utils.plot_stiffness(plot_dic = knee_stiffness_plot_dic, load_condition='loaded',\
                     kinematics_ticks=[-40,-20,0,20,40,60,80],moment_ticks=[-160,-120,-80,-40,0,40,80,120],ax1=ax1,ax2=ax2,ax3=ax3)
# biarticular knee muscles noload case
knee_stiffness_plot_dic = {'loaded_toe_off':loaded_mean_toe_off,'noload_toe_off':noload_mean_toe_off,'kinematics':utils.smooth(mean_mono_noload_knee_kinematics,5),
                          'kinematics_std':utils.smooth(std_mono_noload_knee_kinematics,5),'moment':utils.smooth(mean_bi_noload_kneemuscles_moment,5),
                          'moment_std':utils.smooth(std_bi_noload_kneemuscles_moment,5),'color':mycolors['french rose'],'toe_off_color':mycolors['french rose'],'label':'bi noload knee muscles'}
utils.plot_stiffness(plot_dic = knee_stiffness_plot_dic, load_condition='noload',
kinematics_ticks=[-40,-20,0,20,40,60,80],moment_ticks=[-160,-120,-80,-40,0,40,80,120],ax1=ax4,ax2=ax5,ax3=ax6)

fig.tight_layout(h_pad=-1, w_pad=-1.5)
fig.subplots_adjust(top=0.98, bottom=0.075, left=0.100, right=0.975,hspace=0.45,wspace=0.35)
plt.show()
fig.savefig('./Figures/Ideal/Biarticular_LoadedVsNoload_KneeActuator_Stiffness.pdf',orientation='landscape',bbox_inches='tight')
#************************************************************
# monoarticular loaded vs noload knee joint stiffness
fig = plt.figure(num='Knee Joint Stiffness',figsize=(15, 12))
gridsize = (3, 4)
ax1 = plt.subplot2grid(gridsize, (0, 0), colspan=2, rowspan=2) # stiffness plot
ax2 = plt.subplot2grid(gridsize, (2, 0)) # kinematics plot
ax3 = plt.subplot2grid(gridsize, (2, 1)) # moment plot
ax4 = plt.subplot2grid(gridsize, (0, 2), colspan=2, rowspan=2) # stiffness plot
ax5 = plt.subplot2grid(gridsize, (2, 2)) # kinematics plot
ax6 = plt.subplot2grid(gridsize, (2, 3)) # moment plot
# knee loaded case
knee_stiffness_plot_dic = {'loaded_toe_off':loaded_mean_toe_off,'noload_toe_off':noload_mean_toe_off,'kinematics':utils.smooth(rra_dataset['mean_loaded_kneejoint_kinematics'],5),
                          'kinematics_std':utils.smooth(rra_dataset['std_loaded_kneejoint_kinematics'],5),'moment':utils.smooth(unnormalized_moment_dataset['mean_loaded_kneemuscles_moment'],5),
                          'moment_std':utils.smooth(unnormalized_moment_dataset['std_loaded_kneemuscles_moment'],5),'color':'k','toe_off_color':'grey','label':'knee joint'}
utils.plot_stiffness(plot_dic = knee_stiffness_plot_dic, load_condition='loaded',\
                     kinematics_ticks=[0,10,20,30,40,50,60,70,80],moment_ticks=[-160,-120,-80,-40,0,40,80,120],ax1=ax1,ax2=ax2,ax3=ax3)
# knee noload case
knee_stiffness_plot_dic = {'loaded_toe_off':loaded_mean_toe_off,'noload_toe_off':noload_mean_toe_off,'kinematics':utils.smooth(rra_dataset['mean_noload_kneejoint_kinematics'],5),
                          'kinematics_std':utils.smooth(rra_dataset['std_noload_kneejoint_kinematics'],5),'moment':utils.smooth(unnormalized_moment_dataset['mean_noload_kneemuscles_moment'],5),
                          'moment_std':utils.smooth(unnormalized_moment_dataset['std_noload_kneemuscles_moment'],5),'color':'xkcd:irish green','toe_off_color':'xkcd:shamrock green','label':'knee joint'}
utils.plot_stiffness(plot_dic = knee_stiffness_plot_dic, load_condition='noload',
kinematics_ticks=[0,10,20,30,40,50,60,70,80],moment_ticks=[-160,-120,-80,-40,0,40,80,120],ax1=ax4,ax2=ax5,ax3=ax6)
# monoarticular knee loaded case
knee_stiffness_plot_dic = {'loaded_toe_off':loaded_mean_toe_off,'noload_toe_off':noload_mean_toe_off,'kinematics':utils.smooth(mean_mono_loaded_knee_kinematics,5),
                          'kinematics_std':utils.smooth(std_mono_loaded_knee_kinematics,5),'moment':utils.smooth(mean_mono_loaded_knee_torque,5),
                          'moment_std':utils.smooth(std_mono_loaded_knee_torque,5),'color':mycolors['cyan blue'],'toe_off_color':mycolors['cyan blue'],'label':'bi loaded knee actuator'}
utils.plot_stiffness(plot_dic = knee_stiffness_plot_dic, load_condition='loaded',\
                     kinematics_ticks=[0,10,20,30,40,50,60,70,80],moment_ticks=[-160,-120,-80,-40,0,40,80,120],ax1=ax1,ax2=ax2,ax3=ax3)
# monoarticular knee noload case
knee_stiffness_plot_dic = {'loaded_toe_off':loaded_mean_toe_off,'noload_toe_off':noload_mean_toe_off,'kinematics':utils.smooth(mean_mono_noload_knee_kinematics,5),
                          'kinematics_std':utils.smooth(std_mono_noload_knee_kinematics,5),'moment':utils.smooth(mean_mono_noload_knee_torque,5),
                          'moment_std':utils.smooth(std_mono_noload_knee_torque,5),'color':mycolors['olympic blue'],'toe_off_color':mycolors['olympic blue'],'label':'bi noload knee actuator'}
utils.plot_stiffness(plot_dic = knee_stiffness_plot_dic, load_condition='noload',
kinematics_ticks=[0,10,20,30,40,50,60,70,80],moment_ticks=[-160,-120,-80,-40,0,40,80,120],ax1=ax4,ax2=ax5,ax3=ax6)
# monoarticular knee muscles loaded case
knee_stiffness_plot_dic = {'loaded_toe_off':loaded_mean_toe_off,'noload_toe_off':noload_mean_toe_off,'kinematics':utils.smooth(mean_mono_loaded_knee_kinematics,5),
                          'kinematics_std':utils.smooth(std_mono_loaded_knee_kinematics,5),'moment':utils.smooth(mean_mono_loaded_kneemuscles_moment,5),
                          'moment_std':utils.smooth(std_mono_loaded_kneemuscles_moment,5),'color':mycolors['crimson red'],'toe_off_color':mycolors['crimson red'],'label':'bi loaded knee muscles'}
utils.plot_stiffness(plot_dic = knee_stiffness_plot_dic, load_condition='loaded',\
                     kinematics_ticks=[0,10,20,30,40,50,60,70,80],moment_ticks=[-160,-120,-80,-40,0,40,80,120],ax1=ax1,ax2=ax2,ax3=ax3)
# monoarticular knee muscles noload case
knee_stiffness_plot_dic = {'loaded_toe_off':loaded_mean_toe_off,'noload_toe_off':noload_mean_toe_off,'kinematics':utils.smooth(mean_mono_noload_knee_kinematics,5),
                          'kinematics_std':utils.smooth(std_mono_noload_knee_kinematics,5),'moment':utils.smooth(mean_mono_noload_kneemuscles_moment,5),
                          'moment_std':utils.smooth(std_mono_noload_kneemuscles_moment,5),'color':mycolors['french rose'],'toe_off_color':mycolors['french rose'],'label':'bi noload knee muscles'}
utils.plot_stiffness(plot_dic = knee_stiffness_plot_dic, load_condition='noload',
kinematics_ticks=[0,10,20,30,40,50,60,70,80],moment_ticks=[-160,-120,-80,-40,0,40,80,120],ax1=ax4,ax2=ax5,ax3=ax6)

fig.tight_layout(h_pad=-1, w_pad=-1.5)
fig.subplots_adjust(top=0.98, bottom=0.075, left=0.100, right=0.975,hspace=0.45,wspace=0.35)
plt.show()
fig.savefig('./Figures/Ideal/Monoarticular_LoadedVsNoload_KneeActuator_Stiffness.pdf',orientation='landscape',bbox_inches='tight')
#************************************************************
