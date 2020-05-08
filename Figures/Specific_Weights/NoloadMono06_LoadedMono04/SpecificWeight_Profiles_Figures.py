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
# ideal exo torque dataset
directory = 'D:/Ali.K.M.Bonab/Walking_Mass_Inertia_Effect/Data/Data/Specific_Weights/*_torque.csv'
files = enumerate(glob.iglob(directory), 1)
exo_torque_dataset = {pathlib.PurePath(f[1]).stem: np.loadtxt(f[1], delimiter=',') for f in files}
# ideal exo power dataset
directory = 'D:/Ali.K.M.Bonab/Walking_Mass_Inertia_Effect/Data/Data/Specific_Weights/*_power.csv'
files = enumerate(glob.iglob(directory), 1)
exo_power_dataset = {pathlib.PurePath(f[1]).stem: np.loadtxt(f[1], delimiter=',') for f in files}
# ideal exo speed dataset
directory = 'D:/Ali.K.M.Bonab/Walking_Mass_Inertia_Effect/Data/Data/Specific_Weights/*_speed.csv'
files = enumerate(glob.iglob(directory), 1)
exo_speed_dataset = {pathlib.PurePath(f[1]).stem: np.loadtxt(f[1], delimiter=',') for f in files}
# muscles moment dataset
directory = 'D:/Ali.K.M.Bonab/Walking_Mass_Inertia_Effect/Data/Data/Specific_Weights/*_moment.csv'
files = enumerate(glob.iglob(directory), 1)
musclesmoment_dataset = {pathlib.PurePath(f[1]).stem: np.loadtxt(f[1], delimiter=',') for f in files}
# gls
gl_noload = {'noload_subject{}_trial{}'.format(i,j): utils.construct_gl_mass_side(subjectno=i,trialno=j,loadcond='noload') for i in subjects for j in trials_num}
gl_loaded = {'loaded_subject{}_trial{}'.format(i,j): utils.construct_gl_mass_side(subjectno=i,trialno=j,loadcond='loaded') for i in subjects for j in trials_num}
#####################################################################################
# Processing Data
# toe-off
noload_mean_toe_off,_,loaded_mean_toe_off,_,noload_toe_off,loaded_toe_off = utils.toe_off_avg_std(gl_noload,gl_loaded,subjects=True)
# exoskeleton torque profiles
# monoarticular
# hip
mono_noload_hip_torque = utils.normalize_direction_data(exo_torque_dataset['monoarticular_hip60knee70_noload_hipactuator_torque'],gl_noload,direction=False)
mean_mono_noload_hip_torque,std_mono_noload_hip_torque = utils.mean_std_over_subjects(mono_noload_hip_torque,avg_trials=False)
# knee
mono_noload_knee_torque = utils.normalize_direction_data(exo_torque_dataset['monoarticular_hip60knee70_noload_kneeactuator_torque'],gl_noload,direction=True)
mean_mono_noload_knee_torque,std_mono_noload_knee_torque = utils.mean_std_over_subjects(mono_noload_knee_torque,avg_trials=False)
# monoarticular
# hip
mono_loaded_hip_torque = utils.normalize_direction_data(exo_torque_dataset['monoarticular_hip70knee40_load_hipactuator_torque'],gl_noload,direction=False)
mean_mono_loaded_hip_torque,std_mono_loaded_hip_torque = utils.mean_std_over_subjects(mono_loaded_hip_torque,avg_trials=False)
# knee
mono_loaded_knee_torque = utils.normalize_direction_data(exo_torque_dataset['monoarticular_hip70knee40_load_kneeactuator_torque'],gl_noload,direction=True)
mean_mono_loaded_knee_torque,std_mono_loaded_knee_torque = utils.mean_std_over_subjects(mono_loaded_knee_torque,avg_trials=False)
#******************************
# exoskeleton power profiles
# monoarticular
# hip
mono_noload_hip_power = utils.normalize_direction_data(exo_power_dataset['monoarticular_hip60knee70_noload_hipactuator_power'],gl_noload,direction=False)
mean_mono_noload_hip_power,std_mono_noload_hip_power = utils.mean_std_over_subjects(mono_noload_hip_power,avg_trials=False)
# knee
mono_noload_knee_power = utils.normalize_direction_data(exo_power_dataset['monoarticular_hip60knee70_noload_kneeactuator_power'],gl_noload,direction=False)
mean_mono_noload_knee_power,std_mono_noload_knee_power = utils.mean_std_over_subjects(mono_noload_knee_power,avg_trials=False)
# monoarticular
# hip
mono_loaded_hip_power = utils.normalize_direction_data(exo_power_dataset['monoarticular_hip70knee40_load_hipactuator_power'],gl_noload,direction=False)
mean_mono_loaded_hip_power,std_mono_loaded_hip_power = utils.mean_std_over_subjects(mono_loaded_hip_power,avg_trials=False)
# knee
mono_loaded_knee_power = utils.normalize_direction_data(exo_power_dataset['monoarticular_hip70knee40_load_kneeactuator_power'],gl_noload,direction=False)
mean_mono_loaded_knee_power,std_mono_loaded_knee_power = utils.mean_std_over_subjects(mono_loaded_knee_power,avg_trials=False)
#******************************
# exoskeleton speed profiles
# monoarticular
# hip
mono_noload_hip_speed = utils.normalize_direction_data(exo_speed_dataset['monoarticular_hip60knee70_noload_hipactuator_speed'],gl_noload,direction=False,normalize=False)
mean_mono_noload_hip_speed,std_mono_noload_hip_speed = utils.mean_std_over_subjects(mono_noload_hip_speed,avg_trials=False)
# knee
mono_noload_knee_speed = utils.normalize_direction_data(exo_speed_dataset['monoarticular_hip60knee70_noload_kneeactuator_speed'],gl_noload,direction=True,normalize=False)
mean_mono_noload_knee_speed,std_mono_noload_knee_speed = utils.mean_std_over_subjects(mono_noload_knee_speed,avg_trials=False)
# monoarticular
# hip
mono_loaded_hip_speed = utils.normalize_direction_data(exo_speed_dataset['monoarticular_hip70knee40_load_hipactuator_speed'],gl_noload,direction=False,normalize=False)
mean_mono_loaded_hip_speed,std_mono_loaded_hip_speed = utils.mean_std_over_subjects(mono_loaded_hip_speed,avg_trials=False)
# knee
mono_loaded_knee_speed = utils.normalize_direction_data(exo_speed_dataset['monoarticular_hip70knee40_load_kneeactuator_speed'],gl_noload,direction=True,normalize=False)
mean_mono_loaded_knee_speed,std_mono_loaded_knee_speed = utils.mean_std_over_subjects(mono_loaded_knee_speed,avg_trials=False)
#******************************
# hip muscles moment
# monoarticular
mono_noload_hipmuscles_moment = utils.normalize_direction_data(musclesmoment_dataset['monoarticular_hip60knee70_noload_hipmuscles_moment'],gl_noload,direction=True)
mean_mono_noload_hipmuscles_moment,std_mono_noload_hipmuscles_moment = utils.mean_std_over_subjects(mono_noload_hipmuscles_moment,avg_trials=False)
# monoarticular
mono_loaded_hipmuscles_moment = utils.normalize_direction_data(musclesmoment_dataset['monoarticular_hip70knee40_load_hipmuscles_moment'],gl_noload,direction=True)
mean_mono_loaded_hipmuscles_moment,std_mono_loaded_hipmuscles_moment = utils.mean_std_over_subjects(mono_loaded_hipmuscles_moment,avg_trials=False)
# knee muscles moment
# monoarticular
mono_noload_kneemuscles_moment = utils.normalize_direction_data(musclesmoment_dataset['monoarticular_hip60knee70_noload_kneemuscles_moment'],gl_noload,direction=True)
mean_mono_noload_kneemuscles_moment,std_mono_noload_kneemuscles_moment = utils.mean_std_over_subjects(mono_noload_kneemuscles_moment,avg_trials=False)
# monoarticular
mono_loaded_kneemuscles_moment = utils.normalize_direction_data(musclesmoment_dataset['monoarticular_hip70knee40_load_kneemuscles_moment'],gl_noload,direction=True)
mean_mono_loaded_kneemuscles_moment,std_mono_loaded_kneemuscles_moment = utils.mean_std_over_subjects(mono_loaded_kneemuscles_moment,avg_trials=False)
#####################################################################################
# rmse and tables
# torque difference
mean_rmse_hip_actuator_torque,std_rmse_hip_actuator_torque = utils.profiles_all_phases_rmse(mono_noload_hip_torque,mono_loaded_hip_torque,noload_toe_off,loaded_toe_off,which_comparison='ideal vs ideal')
mean_rmse_knee_actuator_torque,std_rmse_knee_actuator_torque = utils.profiles_all_phases_rmse(mono_noload_knee_torque,mono_loaded_knee_torque,noload_toe_off,loaded_toe_off,which_comparison='ideal vs ideal')
# power difference
mean_rmse_hip_actuator_power,std_rmse_hip_actuator_power = utils.profiles_all_phases_rmse(mono_noload_hip_power,mono_loaded_hip_power,noload_toe_off,loaded_toe_off,which_comparison='ideal vs ideal')
mean_rmse_knee_actuator_power,std_rmse_knee_actuator_power = utils.profiles_all_phases_rmse(mono_noload_knee_power,mono_loaded_knee_power,noload_toe_off,loaded_toe_off,which_comparison='ideal vs ideal')
# muscles moment difference
mean_rmse_hip_musclesmoment,std_rmse_hip_musclesmoment = utils.profiles_all_phases_rmse(mono_noload_hipmuscles_moment,mono_loaded_hipmuscles_moment,noload_toe_off,loaded_toe_off,which_comparison='ideal vs ideal')
mean_rmse_knee_musclesmoment,std_rmse_knee_musclesmoment = utils.profiles_all_phases_rmse(mono_noload_kneemuscles_moment,mono_loaded_kneemuscles_moment,noload_toe_off,loaded_toe_off,which_comparison='ideal vs ideal')
# RMSE plot       
plot_dic = {'mean_11':mean_rmse_hip_actuator_torque,'mean_12':mean_rmse_hip_actuator_power,
            'mean_13':mean_rmse_hip_musclesmoment,'mean_21':mean_rmse_knee_actuator_torque,
            'mean_22':mean_rmse_knee_actuator_power,'mean_23':mean_rmse_knee_musclesmoment,
            'std_11':std_rmse_hip_actuator_torque,'std_12':std_rmse_hip_actuator_power,
            'std_13':std_rmse_hip_musclesmoment,'std_21':std_rmse_knee_actuator_torque,
            'std_22':std_rmse_knee_actuator_power,'std_23':std_rmse_knee_musclesmoment,
            'color_1':mycolors['pastel blue'],'color_2':mycolors['deep space sparkle'],'title_1':'assistive actuators\n torque error',
            'title_2':'assistive actuators\n power error','title_3':'assisted muscles\n moment error',
            'y_ticks':[0,0.2,0.4,0.6,0.8,1,1.2,1.4,1.6]}
fig = plt.figure(num='RMSE',figsize=(20.8, 6.4))
utils.rmse_barplots(plot_dic=plot_dic)
fig.tight_layout(h_pad=-1, w_pad=-1.5)
fig.subplots_adjust(top=0.98, bottom=0.075, left=0.100, right=0.975,hspace=0.35,wspace=0.15)
fig.savefig('./Figures/Specific_Weights/NoloadMono06_LoadedMono04/RMSE.pdf',orientation='landscape',bbox_inches='tight')
plt.show()

#####################################################################################
# modified Augmentation Factor
thigh_length = 0.52 # m
motor_max_torque = 2  # N-m/Kg
motor_inertia = 0.000506  # kg.m^2
mono_thigh_com=0.30 # m
mono_thigh_com=0.30 # m
shank_com=0.18 # m

# mass of segments
mono_noload_m_waist = 2*3
mono_noload_m_thigh = 2*2.5
mono_noload_m_shank = 2*0.9
mono_loaded_m_waist = 2*3
mono_loaded_m_thigh = 2*2.5
mono_loaded_m_shank = 2*0.9

# Gear Ratio max_needed_torque/motor_max_torque
mono_noload_hip_weight = 60
mono_noload_knee_weight = 70
mono_loaded_hip_weight = 70
mono_loaded_knee_weight = 40
mono_noload_hip_ratio = mono_noload_hip_weight/motor_max_torque
mono_noload_knee_ratio = mono_noload_knee_weight/motor_max_torque
mono_loaded_hip_ratio = mono_loaded_hip_weight/motor_max_torque
mono_loaded_knee_ratio = mono_loaded_knee_weight/motor_max_torque

# I = motor_inertia*(ratio^2) + segment_mass*(segment_com^2)
mono_loaded_I_thigh = motor_inertia*(mono_loaded_hip_ratio**2)+ ((mono_thigh_com**2)*mono_loaded_m_thigh)
mono_loaded_I_shank =motor_inertia*(mono_loaded_knee_ratio**2) + (((thigh_length+shank_com)**2)*mono_loaded_m_shank)
mono_noload_I_thigh = motor_inertia*(mono_noload_hip_ratio**2)+ ((mono_thigh_com**2)*mono_noload_m_thigh)
mono_noload_I_shank =motor_inertia*(mono_noload_knee_ratio**2) + (((thigh_length+shank_com)**2)*mono_noload_m_shank)

# modified augmentation factor dictionary
mono_loaded_modified_AF_dic = {'positive_power' : exo_power_dataset['monoarticular_hip70knee40_load_hipactuator_avg_positive_power']+exo_power_dataset['monoarticular_hip70knee40_load_kneeactuator_avg_positive_power'],
                               'negative_power' : exo_power_dataset['monoarticular_hip70knee40_load_hipactuator_avg_negative_power']+exo_power_dataset['monoarticular_hip70knee40_load_kneeactuator_avg_negative_power'],
                                     'exo_mass' : [mono_loaded_m_waist,mono_loaded_m_thigh,mono_loaded_m_shank,0],
                                  'exo_inertia' : [mono_loaded_I_thigh,mono_loaded_I_shank,0],
                                           'gl' : gl_noload}
mono_noload_modified_AF_dic = {'positive_power' : exo_power_dataset['monoarticular_hip60knee70_noload_hipactuator_avg_positive_power']+exo_power_dataset['monoarticular_hip60knee70_noload_kneeactuator_avg_positive_power'],
                             'negative_power' : exo_power_dataset['monoarticular_hip60knee70_noload_hipactuator_avg_negative_power']+exo_power_dataset['monoarticular_hip60knee70_noload_kneeactuator_avg_negative_power'],
                                   'exo_mass' : [mono_noload_m_waist,mono_noload_m_thigh,mono_noload_m_shank,0],
                                'exo_inertia' : [mono_noload_I_thigh,mono_noload_I_shank,0],
                                         'gl' : gl_noload}
# modified augmentation factor
mean_mono_loaded_modified_AF,std_mono_loaded_modified_AF = utils.specific_weights_modified_AF(mono_loaded_modified_AF_dic,normalize_AF=True)
mean_mono_noload_modified_AF,std_mono_noload_modified_AF = utils.specific_weights_modified_AF(mono_noload_modified_AF_dic,normalize_AF=True)
mean_mono_loaded_regen_modified_AF,std_mono_loaded_regen_modified_AF = utils.specific_weights_modified_AF(mono_loaded_modified_AF_dic,normalize_AF=True,regen_effect=True,regeneration_efficiency=0.65)
mean_mono_noload_regen_modified_AF,std_mono_noload_regen_modified_AF = utils.specific_weights_modified_AF(mono_noload_modified_AF_dic,normalize_AF=True,regen_effect=True,regeneration_efficiency=0.65)
# Table
cellText = [[mono_noload_hip_weight,mono_noload_knee_weight,mono_noload_hip_ratio,mono_noload_knee_ratio,mono_noload_I_thigh,mono_noload_I_shank,\
            mean_mono_noload_modified_AF,std_mono_noload_modified_AF,mean_mono_noload_regen_modified_AF,std_mono_noload_regen_modified_AF],\
            [mono_loaded_hip_weight,mono_loaded_knee_weight,mono_loaded_hip_ratio,mono_loaded_knee_ratio,mono_loaded_I_thigh,mono_loaded_I_shank,\
            mean_mono_loaded_modified_AF,std_mono_loaded_modified_AF,mean_mono_loaded_regen_modified_AF,std_mono_loaded_regen_modified_AF]]
rows = ['monoarticular','monoarticular']
columns = ['hip actuator\npeak torque','knee actuator\n peak torque',
           'hip actuator\n gear ratio','knee actuator\n gear ratio',
           'thigh inertia','shank inertia','Mean Modified AF','Std Modified AF','Mean Modified AF\n+regeneration','Std Modified AF\n+regeneration']
fig, ax = plt.subplots(figsize=(6.4, 6.8))
table = ax.table(cellText=cellText,rowLabels=rows,colLabels=columns,loc='center')
table.scale(3,6)
table.set_fontsize(25)
ax.axis('off')
fig.tight_layout()
fig.savefig('./Figures/Specific_Weights/NoloadMono06_LoadedMono04/Exos_Inertial_Properties.pdf',orientation='landscape',bbox_inches='tight')
plt.show()

#####################################################################################
# Plots
# hip joint moment plot dictionaries
mono_noload_hip_musclesmoment_dic = {'pgc':gait_cycle,'avg':utils.smooth(mean_mono_noload_hipmuscles_moment,3),'label':'noload muscles',
                        'std':utils.smooth(std_mono_noload_hipmuscles_moment,3),'avg_toeoff':noload_mean_toe_off}
mono_loaded_hip_musclesmoment_dic = {'pgc':gait_cycle,'avg':utils.smooth(mean_mono_loaded_hipmuscles_moment,3),'label':'loaded muscles',
                        'std':utils.smooth(std_mono_loaded_hipmuscles_moment,3),'avg_toeoff':loaded_mean_toe_off}
unassist_noload_hip_musclesmoment_dic = {'pgc':gait_cycle,'avg':utils.smooth(unassist_dataset['mean_norm_noload_hipmuscles_moment'],3),'label':'noload joint',
                        'std':utils.smooth(unassist_dataset['std_norm_noload_hipmuscles_moment'],3),'avg_toeoff':noload_mean_toe_off}
unassist_loaded_hip_musclesmoment_dic = {'pgc':gait_cycle,'avg':utils.smooth(unassist_dataset['mean_norm_loaded_hipmuscles_moment'],3),'label':'noload joint',
                        'std':utils.smooth(unassist_dataset['std_norm_loaded_hipmuscles_moment'],3),'avg_toeoff':loaded_mean_toe_off}

# knee joint moment plot dictionaries
mono_noload_knee_musclesmoment_dic = {'pgc':gait_cycle,'avg':utils.smooth(mean_mono_noload_kneemuscles_moment,3),'label':'noload muscles',
                        'std':utils.smooth(std_mono_noload_kneemuscles_moment,3),'avg_toeoff':noload_mean_toe_off}
mono_loaded_knee_musclesmoment_dic = {'pgc':gait_cycle,'avg':utils.smooth(mean_mono_loaded_kneemuscles_moment,3),'label':'loaded muscles',
                        'std':utils.smooth(std_mono_loaded_kneemuscles_moment,3),'avg_toeoff':loaded_mean_toe_off}
unassist_noload_knee_musclesmoment_dic = {'pgc':gait_cycle,'avg':utils.smooth(unassist_dataset['mean_norm_noload_kneemuscles_moment'],3),'label':'noload joint',
                        'std':utils.smooth(unassist_dataset['std_norm_noload_kneemuscles_moment'],3),'avg_toeoff':noload_mean_toe_off}
unassist_loaded_knee_musclesmoment_dic = {'pgc':gait_cycle,'avg':utils.smooth(unassist_dataset['mean_norm_loaded_kneemuscles_moment'],3),'label':'noload joint',
                        'std':utils.smooth(unassist_dataset['std_norm_loaded_kneemuscles_moment'],3),'avg_toeoff':loaded_mean_toe_off}

# hip actuator torque plot dictionaries
mono_noload_hip_torque_dic= {'pgc':gait_cycle,'avg':utils.smooth(mean_mono_noload_hip_torque,3),'label':'noload actuator',
                        'std':utils.smooth(std_mono_noload_hip_torque,3),'avg_toeoff':noload_mean_toe_off}
mono_loaded_hip_torque_dic = {'pgc':gait_cycle,'avg':utils.smooth(mean_mono_loaded_hip_torque,3),'label':'loaded actuator',
                        'std':utils.smooth(std_mono_loaded_hip_torque,3),'avg_toeoff':loaded_mean_toe_off}

# knee actuator torque plot dictionaries
mono_noload_knee_torque_dic= {'pgc':gait_cycle,'avg':utils.smooth(mean_mono_noload_knee_torque,3),'label':'noload actuator',
                        'std':utils.smooth(std_mono_noload_knee_torque,3),'avg_toeoff':noload_mean_toe_off}
mono_loaded_knee_torque_dic = {'pgc':gait_cycle,'avg':utils.smooth(mean_mono_loaded_knee_torque,3),'label':'loaded actuator',
                        'std':utils.smooth(std_mono_loaded_knee_torque,3),'avg_toeoff':loaded_mean_toe_off}

# joint power plot dictionaries
noload_hip_power_dic = {'pgc':gait_cycle,'avg':utils.smooth(rra_dataset['mean_norm_noload_hipjoint_power'],3),'label':'noload joint',
                        'std':utils.smooth(rra_dataset['std_norm_noload_hipjoint_power'],3),'avg_toeoff':noload_mean_toe_off}
noload_knee_power_dic = {'pgc':gait_cycle,'avg':utils.smooth(rra_dataset['mean_norm_noload_kneejoint_power'],3),'label':'noload joint',
                        'std':utils.smooth(rra_dataset['std_norm_noload_kneejoint_power'],3),'avg_toeoff':noload_mean_toe_off}
loaded_hip_power_dic = {'pgc':gait_cycle,'avg':utils.smooth(rra_dataset['mean_norm_loaded_hipjoint_power'],3),'label':'noload joint',
                        'std':utils.smooth(rra_dataset['std_norm_loaded_hipjoint_power'],3),'avg_toeoff':loaded_mean_toe_off}
loaded_knee_power_dic = {'pgc':gait_cycle,'avg':utils.smooth(rra_dataset['mean_norm_loaded_kneejoint_power'],3),'label':'noload joint',
                        'std':utils.smooth(rra_dataset['std_norm_loaded_kneejoint_power'],3),'avg_toeoff':loaded_mean_toe_off}

# hip actuator power plot dictionaries
mono_noload_hip_power_dic= {'pgc':gait_cycle,'avg':utils.smooth(mean_mono_noload_hip_power,3),'label':'noload actuator',
                        'std':utils.smooth(std_mono_noload_hip_power,3),'avg_toeoff':noload_mean_toe_off}
mono_loaded_hip_power_dic = {'pgc':gait_cycle,'avg':utils.smooth(mean_mono_loaded_hip_power,3),'label':'loaded actuator',
                        'std':utils.smooth(std_mono_loaded_hip_power,3),'avg_toeoff':loaded_mean_toe_off}

# knee actuator power plot dictionaries
mono_noload_knee_power_dic= {'pgc':gait_cycle,'avg':utils.smooth(mean_mono_noload_knee_power,3),'label':'noload actuator',
                        'std':utils.smooth(std_mono_noload_knee_power,3),'avg_toeoff':noload_mean_toe_off}
mono_loaded_knee_power_dic = {'pgc':gait_cycle,'avg':utils.smooth(mean_mono_loaded_knee_power,3),'label':'loaded actuator',
                        'std':utils.smooth(std_mono_loaded_knee_power,3),'avg_toeoff':loaded_mean_toe_off}

# joint speed plot dictionaries
noload_hip_speed_dic = {'pgc':gait_cycle,'avg':utils.smooth(rra_dataset['mean_noload_hipjoint_speed'],3),'label':'noload joint',
                        'std':utils.smooth(rra_dataset['std_noload_hipjoint_speed'],3),'avg_toeoff':noload_mean_toe_off}
noload_knee_speed_dic = {'pgc':gait_cycle,'avg':utils.smooth(rra_dataset['mean_noload_kneejoint_speed'],3),'label':'noload joint',
                        'std':utils.smooth(rra_dataset['std_noload_kneejoint_speed'],3),'avg_toeoff':noload_mean_toe_off}
loaded_hip_speed_dic = {'pgc':gait_cycle,'avg':utils.smooth(rra_dataset['mean_loaded_hipjoint_speed'],3),'label':'noload joint',
                        'std':utils.smooth(rra_dataset['std_loaded_hipjoint_speed'],3),'avg_toeoff':loaded_mean_toe_off}
loaded_knee_speed_dic = {'pgc':gait_cycle,'avg':utils.smooth(rra_dataset['mean_loaded_kneejoint_speed'],3),'label':'noload joint',
                        'std':utils.smooth(rra_dataset['std_loaded_kneejoint_speed'],3),'avg_toeoff':loaded_mean_toe_off}

# hip actuator speed plot dictionaries
mono_noload_hip_speed_dic= {'pgc':gait_cycle,'avg':utils.smooth(mean_mono_noload_hip_speed,3),'label':'noload actuator',
                        'std':utils.smooth(std_mono_noload_hip_speed,3),'avg_toeoff':noload_mean_toe_off}
mono_loaded_hip_speed_dic = {'pgc':gait_cycle,'avg':utils.smooth(mean_mono_loaded_hip_speed,3),'label':'loaded actuator',
                        'std':utils.smooth(std_mono_loaded_hip_speed,3),'avg_toeoff':loaded_mean_toe_off}

# knee actuator speed plot dictionaries
mono_noload_knee_speed_dic= {'pgc':gait_cycle,'avg':utils.smooth(mean_mono_noload_knee_speed,3),'label':'noload actuator',
                        'std':utils.smooth(std_mono_noload_knee_speed,3),'avg_toeoff':noload_mean_toe_off}
mono_loaded_knee_speed_dic = {'pgc':gait_cycle,'avg':utils.smooth(mean_mono_loaded_knee_speed,3),'label':'loaded actuator',
                        'std':utils.smooth(std_mono_loaded_knee_speed,3),'avg_toeoff':loaded_mean_toe_off}

#####################################################################################
# Paper figure
monovsmono_color_dic = {
'color_1_list' : ['xkcd:irish green','k','xkcd:irish green','k'],
'color_2_list' : [mycolors['olympic blue'],mycolors['dark purple'],mycolors['olympic blue'],
                  mycolors['dark purple']],
'color_3_list' : [mycolors['french rose'],mycolors['crimson red'],mycolors['french rose'],
                  mycolors['crimson red']]
}

plot_dic={
'plot_1_list' : [unassist_noload_hip_musclesmoment_dic,unassist_loaded_hip_musclesmoment_dic,unassist_noload_knee_musclesmoment_dic,unassist_loaded_knee_musclesmoment_dic],
'plot_2_list' : [mono_noload_hip_torque_dic,mono_loaded_hip_torque_dic,mono_noload_knee_torque_dic,mono_loaded_knee_torque_dic],
'plot_3_list' : [mono_noload_hip_musclesmoment_dic, mono_loaded_hip_musclesmoment_dic,mono_noload_knee_musclesmoment_dic, mono_loaded_knee_musclesmoment_dic],
'plot_titles' : ['noload monoarticular hip joint','loaded monoarticular hip joint','noload monoarticular knee joint','loaded monoarticular knee joint']
}
# plot
fig = plt.figure(num='noload Knee Speed',figsize=(9.4, 6.8))
utils.plot_joint_muscle_exo(nrows=2,ncols=2,plot_dic=plot_dic,color_dic=monovsmono_color_dic,\
                            thirdplot=True,ylabel=' flexion/extension\n(N-m/kg)',y_ticks=np.arange(-2,3,1))
fig.tight_layout()
plt.show()
fig.savefig('./Figures/Specific_Weights/NoloadMono06_LoadedMono04/PaperFigure_TorqueProfiles.pdf',orientation='landscape',bbox_inches='tight')

plot_dic={
'plot_1_list' : [noload_hip_power_dic,loaded_hip_power_dic,noload_knee_power_dic,loaded_knee_power_dic],
'plot_2_list' : [mono_noload_hip_power_dic,mono_loaded_hip_power_dic,mono_noload_knee_power_dic,mono_loaded_knee_power_dic],
'plot_titles' : ['noload monoarticular hip joint','loaded monoarticular hip joint','noload monoarticular knee joint','loaded monoarticular knee joint']
}
# plot
fig = plt.figure(num='noload Knee Speed',figsize=(9.4, 6.8))
utils.plot_joint_muscle_exo(nrows=2,ncols=2,plot_dic=plot_dic,color_dic=monovsmono_color_dic,\
                            thirdplot=False,ylabel=' flexion/extension\n(W/kg)',y_ticks=np.arange(-2,4,1))
fig.tight_layout(h_pad=-1, w_pad=-1.5)
fig.subplots_adjust(top=0.98, bottom=0.075, left=0.100, right=0.975,hspace=0.25,wspace=0.15)
plt.show()
fig.savefig('./Figures/Specific_Weights/NoloadMono06_LoadedMono04/PaperFigure_PowerProfiles.pdf',orientation='landscape',bbox_inches='tight')

# nested subplot for merging two paper figures
plot_dic_1={
'plot_1_list' : [unassist_noload_hip_musclesmoment_dic,unassist_loaded_hip_musclesmoment_dic,unassist_noload_knee_musclesmoment_dic,unassist_loaded_knee_musclesmoment_dic],
'plot_2_list' : [mono_noload_hip_torque_dic,mono_loaded_hip_torque_dic,mono_noload_knee_torque_dic,mono_loaded_knee_torque_dic],
'plot_3_list' : [mono_noload_hip_musclesmoment_dic, mono_loaded_hip_musclesmoment_dic,mono_noload_knee_musclesmoment_dic, mono_loaded_knee_musclesmoment_dic],
'plot_titles' : ['noload monoarticular hip joint','loaded monoarticular hip joint','noload monoarticular knee joint','loaded monoarticular knee joint'],
'y_ticks': [-2,-1,0,1,2], 'y_label':'flexion/extension\nmoment (N-m/kg)','general_title':'Devices torque profiles','thirdplot':True
}
plot_dic_2={
'plot_1_list' : [noload_hip_power_dic,loaded_hip_power_dic,noload_knee_power_dic,loaded_knee_power_dic],
'plot_2_list' : [mono_noload_hip_power_dic,mono_loaded_hip_power_dic,mono_noload_knee_power_dic,mono_loaded_knee_power_dic],
'plot_titles' : ['noload monoarticular hip joint','loaded monoarticular hip joint','noload monoarticular knee joint','loaded monoarticular knee joint'],
'y_ticks': [-3,-2,-1,0,1,2,3], 'y_label':'flexion/extension\npower (W/kg)','general_title':'Devices power profiles','thirdplot':False
}
fig = plt.figure(num='Main Paper figure',figsize=(12.8, 16.8))
utils.nested_plots(fig,plot_dic_1,plot_dic_2,monovsmono_color_dic,monovsmono_color_dic)
fig.tight_layout(h_pad=-1, w_pad=-1.5)
fig.subplots_adjust(top=0.98, bottom=0.075, left=0.100, right=0.975,hspace=0.25,wspace=0.30)
plt.show()
fig.savefig('./Figures/Specific_Weights/NoloadMono06_LoadedMono04/PaperFigure_Profiles.pdf',orientation='landscape',bbox_inches='tight')

#******************************************************************************************************************************
#******************************************************************************************************************************
# defualt color dictionary
default_color_dic = {
'color_1_list' : ['k','xkcd:irish green','k','xkcd:irish green'],
'color_2_list' : [mycolors['cyan blue'],mycolors['olympic blue'],mycolors['cyan blue'],mycolors['olympic blue']],
'color_3_list' : [mycolors['crimson red'],mycolors['french rose'],mycolors['crimson red'],mycolors['french rose']]
}
monovsmono_color_dic = {
'color_1_list' : ['xkcd:irish green','k','xkcd:irish green','k'],
'color_2_list' : [mycolors['olympic blue'],mycolors['dark purple'],mycolors['olympic blue'],
                  mycolors['dark purple']],
'color_3_list' : [mycolors['french rose'],mycolors['crimson red'],mycolors['french rose'],
                  mycolors['crimson red']]
}

# ***************************
# hip joint moment figure
# required dictionary
plot_dic={
'plot_1_list' : [unassist_noload_hip_musclesmoment_dic,unassist_loaded_hip_musclesmoment_dic],
'plot_3_list' : [mono_noload_hip_musclesmoment_dic, mono_loaded_hip_musclesmoment_dic],
'plot_2_list' : [mono_noload_hip_torque_dic,mono_loaded_hip_torque_dic],
'plot_titles' : ['noload monoarticular hip joint','loaded monoarticular hip joint']
}
# plot
fig = plt.figure(num='Noload Hip Torque',figsize=(9.4, 6.8))
utils.plot_joint_muscle_exo(nrows=2,ncols=1,plot_dic=plot_dic,color_dic=monovsmono_color_dic,ylabel='hip flexion/extension (N-m/kg)')
fig.tight_layout()
fig.savefig('./Figures/Specific_Weights/NoloadMono06_LoadedMono04/Exoskeletons_Hip_Torque.pdf',orientation='landscape',bbox_inches='tight')
plt.show()

# ***************************
# knee joint moment figure
# required dictionary
plot_dic={
'plot_1_list' : [unassist_noload_knee_musclesmoment_dic,unassist_loaded_knee_musclesmoment_dic],
'plot_3_list' : [mono_noload_knee_musclesmoment_dic,mono_loaded_knee_musclesmoment_dic],
'plot_2_list' : [mono_noload_knee_torque_dic,mono_loaded_knee_torque_dic],
'plot_titles' : ['noload monoarticular knee joint','loaded monoarticular knee joint']
}
# plot
fig = plt.figure(num='Noload Knee Torque',figsize=(9.4, 6.8))
utils.plot_joint_muscle_exo(nrows=2,ncols=1,plot_dic=plot_dic,color_dic=monovsmono_color_dic,ylabel='knee flexion/extension (N-m/kg)')
fig.tight_layout()
plt.show()
fig.savefig('./Figures/Specific_Weights/NoloadMono06_LoadedMono04/Exoskeletons_Knee_Torque.pdf',orientation='landscape',bbox_inches='tight')

#******************************************************************************************************************************
# hip joint power figure
# required dictionary
plot_dic={
'plot_1_list' : [noload_hip_power_dic,loaded_hip_power_dic],
'plot_2_list' : [mono_noload_hip_power_dic,mono_loaded_hip_power_dic],
'plot_titles' : ['noload monoarticular hip joint','loaded monoarticular hip joint']
}
# plot
fig = plt.figure(num='Noload Hip Power',figsize=(9.4, 6.8))
utils.plot_joint_muscle_exo(nrows=2,ncols=1,plot_dic=plot_dic,color_dic=monovsmono_color_dic,\
                            thirdplot=False,ylabel='hip flexion/extension (W/kg)',y_ticks=np.arange(-4,5,2))
fig.tight_layout()
fig.savefig('./Figures/Specific_Weights/NoloadMono06_LoadedMono04/Exoskeletons_Hip_Power.pdf',orientation='landscape',bbox_inches='tight')
plt.show()

# ***************************
# knee joint power figure
# required dictionary
plot_dic={
'plot_1_list' : [noload_knee_power_dic,loaded_knee_power_dic],
'plot_2_list' : [mono_noload_knee_power_dic,mono_loaded_knee_power_dic],
'plot_titles' : ['noload monoarticular knee joint','loaded monoarticular knee joint']
}
# 
fig = plt.figure(num='Noload Knee Power',figsize=(9.4, 6.8))
utils.plot_joint_muscle_exo(nrows=2,ncols=1,plot_dic=plot_dic,color_dic=monovsmono_color_dic,\
                            thirdplot=False,ylabel='knee flexion/extension (W/kg)',y_ticks=np.arange(-4,6,2))
fig.tight_layout()
plt.show()
fig.savefig('./Figures/Specific_Weights/NoloadMono06_LoadedMono04/Exoskeletons_Knee_Power.pdf',orientation='landscape',bbox_inches='tight')

#******************************************************************************************************************************
# hip joint speed figure
# required dictionary
plot_dic={
'plot_1_list' : [noload_hip_speed_dic,loaded_hip_speed_dic],
'plot_2_list' : [mono_noload_hip_speed_dic,mono_loaded_hip_speed_dic],
'plot_titles' : ['noload monoarticular hip joint','loaded monoarticular hip joint']
}
# plot
fig = plt.figure(num='Noload Hip Speed',figsize=(9.4, 6.8))
utils.plot_joint_muscle_exo(nrows=2,ncols=1,plot_dic=plot_dic,color_dic=monovsmono_color_dic,\
                            thirdplot=False,ylabel='hip flexion/extension (rad/s)',y_ticks=np.arange(-6,4,2))
fig.tight_layout()
fig.savefig('./Figures/Specific_Weights/NoloadMono06_LoadedMono04/Exoskeletons_Hip_Speed.pdf',orientation='landscape',bbox_inches='tight')
plt.show()

# ***************************
# knee joint speed figure
# required dictionary
plot_dic={
'plot_1_list' : [noload_knee_speed_dic,loaded_knee_speed_dic],
'plot_2_list' : [mono_noload_knee_speed_dic,mono_loaded_knee_speed_dic],
'plot_titles' : ['noload monoarticular knee joint','loaded monoarticular knee joint']
}
# plot
fig = plt.figure(num='Noload Knee Speed',figsize=(9.4, 6.8))
utils.plot_joint_muscle_exo(nrows=2,ncols=1,plot_dic=plot_dic,color_dic=monovsmono_color_dic,\
                            thirdplot=False,ylabel='knee flexion/extension (rad/s)',y_ticks=np.arange(-6,11,3))
fig.tight_layout()
plt.show()
fig.savefig('./Figures/Specific_Weights/NoloadMono06_LoadedMono04/Exoskeletons_Knee_Speed.pdf',orientation='landscape',bbox_inches='tight')
