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
#####################################################################################
subjects = ['05','07','09','10','11','12','14']
trials_num = ['01','02','03']
gait_cycle = np.linspace(0,100,1000)
#####################################################################################
# Reading CSV files into a dictionary and constructing gls
# joint reaction moment dataset
directory = r'.\Data\Unassist\*_reaction_moments.csv'
files = enumerate(glob.iglob(directory), 1)
jointreaction_dataset = {pathlib.PurePath(f[1]).stem: np.loadtxt(f[1], delimiter=',') for f in files}
noload_reaction_moment_dictionary = utils.clasify_data(jointreaction_dataset['unassist_noload_reaction_moments'],loadcondition='noload')
loaded_reaction_moment_dictionary = utils.clasify_data(jointreaction_dataset['unassist_loaded_reaction_moments'],loadcondition='loaded')
# joint reaction force dataset
directory = r'.\Data\Unassist\*_reaction_forces.csv'
files = enumerate(glob.iglob(directory), 1)
jointreaction_dataset = {pathlib.PurePath(f[1]).stem: np.loadtxt(f[1], delimiter=',') for f in files}
noload_reaction_force_dictionary = utils.clasify_data(jointreaction_dataset['unassist_noload_reaction_forces'],loadcondition='noload',forces_name=['Fx','Fy','Fz'])
loaded_reaction_force_dictionary = utils.clasify_data(jointreaction_dataset['unassist_loaded_reaction_forces'],loadcondition='loaded',forces_name=['Fx','Fy','Fz'])
# gls
gl_noload = {'noload_subject{}_trial{}'.format(i,j): utils.construct_gl_mass_side(subjectno=i,trialno=j,loadcond='noload') for i in subjects for j in trials_num}
gl_loaded = {'loaded_subject{}_trial{}'.format(i,j): utils.construct_gl_mass_side(subjectno=i,trialno=j,loadcond='loaded') for i in subjects for j in trials_num}
#####################################################################################
# Processing Data
# toe-off
noload_mean_toe_off,_,loaded_mean_toe_off,_ = utils.toe_off_avg_std(gl_noload,gl_loaded)
# reaction forces and moments
# "RMz: z component of reaction moment"
# back joint reaction moment
# moments
loaded_backjoint_RMz = utils.normalize_direction_data(loaded_reaction_moment_dictionary['back_joint_Mz'],gl_noload,direction=False)
noload_backjoint_RMz = utils.normalize_direction_data(noload_reaction_moment_dictionary['back_joint_Mz'],gl_noload,direction=False)
mean_loaded_backjoint_RMz,std_loaded_backjoint_RMz = utils.mean_std_over_subjects(loaded_backjoint_RMz)
mean_noload_backjoint_RMz,std_noload_backjoint_RMz = utils.mean_std_over_subjects(noload_backjoint_RMz)
# forces
loaded_backjoint_RFz = utils.normalize_direction_data(loaded_reaction_force_dictionary['back_joint_Fz'],gl_noload,direction=False)
noload_backjoint_RFz = utils.normalize_direction_data(noload_reaction_force_dictionary['back_joint_Fz'],gl_noload,direction=False)
mean_loaded_backjoint_RFz,std_loaded_backjoint_RFz = utils.mean_std_over_subjects(loaded_backjoint_RFz)
mean_noload_backjoint_RFz,std_noload_backjoint_RFz = utils.mean_std_over_subjects(noload_backjoint_RFz)
# duct tape joint reaction moment
# moments
loaded_duct_tape_joint_RMz = utils.normalize_direction_data(loaded_reaction_moment_dictionary['duct_tape_joint_Mz'],gl_noload,direction=False)
mean_loaded_duct_tape_joint_RMz,std_loaded_duct_tape_joint_RMz = utils.mean_std_over_subjects(loaded_duct_tape_joint_RMz)
# forces
loaded_duct_tape_joint_RFz = utils.normalize_direction_data(loaded_reaction_force_dictionary['duct_tape_joint_Fz'],gl_noload,direction=False)
mean_loaded_duct_tape_joint_RFz,std_loaded_duct_tape_joint_RFz = utils.mean_std_over_subjects(loaded_duct_tape_joint_RFz)

# hip joint reaction moment
# moments
loaded_hipjoint_RMz = utils.normalize_direction_data(loaded_reaction_moment_dictionary['hip_joint_Mz'],gl_noload,direction=False)
noload_hipjoint_RMz = utils.normalize_direction_data(noload_reaction_moment_dictionary['hip_joint_Mz'],gl_noload,direction=False)
mean_loaded_hipjoint_RMz,std_loaded_hipjoint_RMz = utils.mean_std_over_subjects(loaded_hipjoint_RMz)
mean_noload_hipjoint_RMz,std_noload_hipjoint_RMz = utils.mean_std_over_subjects(noload_hipjoint_RMz)
# forces
# Fx
loaded_hipjoint_RFx = utils.normalize_direction_data(loaded_reaction_force_dictionary['hip_joint_Fx'],gl_noload,direction=False)
noload_hipjoint_RFx = utils.normalize_direction_data(noload_reaction_force_dictionary['hip_joint_Fx'],gl_noload,direction=False)
mean_loaded_hipjoint_RFx,std_loaded_hipjoint_RFx = utils.mean_std_over_subjects(loaded_hipjoint_RFx)
mean_noload_hipjoint_RFx,std_noload_hipjoint_RFx = utils.mean_std_over_subjects(noload_hipjoint_RFx)
# Fy
loaded_hipjoint_RFy = utils.normalize_direction_data(loaded_reaction_force_dictionary['hip_joint_Fy'],gl_noload,direction=False)
noload_hipjoint_RFy = utils.normalize_direction_data(noload_reaction_force_dictionary['hip_joint_Fy'],gl_noload,direction=False)
mean_loaded_hipjoint_RFy,std_loaded_hipjoint_RFy = utils.mean_std_over_subjects(loaded_hipjoint_RFy)
mean_noload_hipjoint_RFy,std_noload_hipjoint_RFy = utils.mean_std_over_subjects(noload_hipjoint_RFy)
# Fz
loaded_hipjoint_RFz = utils.normalize_direction_data(loaded_reaction_force_dictionary['hip_joint_Fz'],gl_noload,direction=False)
noload_hipjoint_RFz = utils.normalize_direction_data(noload_reaction_force_dictionary['hip_joint_Fz'],gl_noload,direction=False)
mean_loaded_hipjoint_RFz,std_loaded_hipjoint_RFz = utils.mean_std_over_subjects(loaded_hipjoint_RFz)
mean_noload_hipjoint_RFz,std_noload_hipjoint_RFz = utils.mean_std_over_subjects(noload_hipjoint_RFz)

# knee joint reaction moment
#Mx
loaded_kneejoint_RMx = utils.normalize_direction_data(loaded_reaction_moment_dictionary['knee_joint_Mx'],gl_noload,direction=False)
noload_kneejoint_RMx = utils.normalize_direction_data(noload_reaction_moment_dictionary['knee_joint_Mx'],gl_noload,direction=False)
mean_loaded_kneejoint_RMx,std_loaded_kneejoint_RMx = utils.mean_std_over_subjects(loaded_kneejoint_RMx)
mean_noload_kneejoint_RMx,std_noload_kneejoint_RMx = utils.mean_std_over_subjects(noload_kneejoint_RMx)
#My
loaded_kneejoint_RMy = utils.normalize_direction_data(loaded_reaction_moment_dictionary['knee_joint_My'],gl_noload,direction=False)
noload_kneejoint_RMy = utils.normalize_direction_data(noload_reaction_moment_dictionary['knee_joint_My'],gl_noload,direction=False)
mean_loaded_kneejoint_RMy,std_loaded_kneejoint_RMy = utils.mean_std_over_subjects(loaded_kneejoint_RMy)
mean_noload_kneejoint_RMy,std_noload_kneejoint_RMy = utils.mean_std_over_subjects(noload_kneejoint_RMy)
#Mz
loaded_kneejoint_RMz = utils.normalize_direction_data(loaded_reaction_moment_dictionary['knee_joint_Mz'],gl_noload,direction=False)
noload_kneejoint_RMz = utils.normalize_direction_data(noload_reaction_moment_dictionary['knee_joint_Mz'],gl_noload,direction=False)
mean_loaded_kneejoint_RMz,std_loaded_kneejoint_RMz = utils.mean_std_over_subjects(loaded_kneejoint_RMz)
mean_noload_kneejoint_RMz,std_noload_kneejoint_RMz = utils.mean_std_over_subjects(noload_kneejoint_RMz)
#Fx
loaded_kneejoint_RFx = utils.normalize_direction_data(loaded_reaction_force_dictionary['knee_joint_Fx'],gl_noload,direction=False)
noload_kneejoint_RFx = utils.normalize_direction_data(noload_reaction_force_dictionary['knee_joint_Fx'],gl_noload,direction=False)
mean_loaded_kneejoint_RFx,std_loaded_kneejoint_RFx = utils.mean_std_over_subjects(loaded_kneejoint_RFx)
mean_noload_kneejoint_RFx,std_noload_kneejoint_RFx = utils.mean_std_over_subjects(noload_kneejoint_RFx)
#Fy
loaded_kneejoint_RFy = utils.normalize_direction_data(loaded_reaction_force_dictionary['knee_joint_Fy'],gl_noload,direction=False)
noload_kneejoint_RFy = utils.normalize_direction_data(noload_reaction_force_dictionary['knee_joint_Fy'],gl_noload,direction=False)
mean_loaded_kneejoint_RFy,std_loaded_kneejoint_RFy = utils.mean_std_over_subjects(loaded_kneejoint_RFy)
mean_noload_kneejoint_RFy,std_noload_kneejoint_RFy = utils.mean_std_over_subjects(noload_kneejoint_RFy)
#Fz
loaded_kneejoint_RFz = utils.normalize_direction_data(loaded_reaction_force_dictionary['knee_joint_Fz'],gl_noload,direction=False)
noload_kneejoint_RFz = utils.normalize_direction_data(noload_reaction_force_dictionary['knee_joint_Fz'],gl_noload,direction=False)
mean_loaded_kneejoint_RFz,std_loaded_kneejoint_RFz = utils.mean_std_over_subjects(loaded_kneejoint_RFz)
mean_noload_kneejoint_RFz,std_noload_kneejoint_RFz = utils.mean_std_over_subjects(noload_kneejoint_RFz)

# patellofemoral joint reaction moment
#Mx
loaded_patellofemoraljoint_RMx = utils.normalize_direction_data(loaded_reaction_moment_dictionary['patellofemoral_joint_Mx'],gl_noload,direction=False)
noload_patellofemoraljoint_RMx = utils.normalize_direction_data(noload_reaction_moment_dictionary['patellofemoral_joint_Mx'],gl_noload,direction=False)
mean_loaded_patellofemoraljoint_RMx,std_loaded_patellofemoraljoint_RMx = utils.mean_std_over_subjects(loaded_patellofemoraljoint_RMx)
mean_noload_patellofemoraljoint_RMx,std_noload_patellofemoraljoint_RMx = utils.mean_std_over_subjects(noload_patellofemoraljoint_RMx)
#My
loaded_patellofemoraljoint_RMy = utils.normalize_direction_data(loaded_reaction_moment_dictionary['patellofemoral_joint_My'],gl_noload,direction=False)
noload_patellofemoraljoint_RMy = utils.normalize_direction_data(noload_reaction_moment_dictionary['patellofemoral_joint_My'],gl_noload,direction=False)
mean_loaded_patellofemoraljoint_RMy,std_loaded_patellofemoraljoint_RMy = utils.mean_std_over_subjects(loaded_patellofemoraljoint_RMy)
mean_noload_patellofemoraljoint_RMy,std_noload_patellofemoraljoint_RMy = utils.mean_std_over_subjects(noload_patellofemoraljoint_RMy)
#Mz
loaded_patellofemoraljoint_RMz = utils.normalize_direction_data(loaded_reaction_moment_dictionary['patellofemoral_joint_Mz'],gl_noload,direction=False)
noload_patellofemoraljoint_RMz = utils.normalize_direction_data(noload_reaction_moment_dictionary['patellofemoral_joint_Mz'],gl_noload,direction=False)
mean_loaded_patellofemoraljoint_RMz,std_loaded_patellofemoraljoint_RMz = utils.mean_std_over_subjects(loaded_patellofemoraljoint_RMz)
mean_noload_patellofemoraljoint_RMz,std_noload_patellofemoraljoint_RMz = utils.mean_std_over_subjects(noload_patellofemoraljoint_RMz)
#Fx
loaded_patellofemoraljoint_RFx = utils.normalize_direction_data(loaded_reaction_force_dictionary['patellofemoral_joint_Fx'],gl_noload,direction=False)
noload_patellofemoraljoint_RFx = utils.normalize_direction_data(noload_reaction_force_dictionary['patellofemoral_joint_Fx'],gl_noload,direction=False)
mean_loaded_patellofemoraljoint_RFx,std_loaded_patellofemoraljoint_RFx = utils.mean_std_over_subjects(loaded_patellofemoraljoint_RFx)
mean_noload_patellofemoraljoint_RFx,std_noload_patellofemoraljoint_RFx = utils.mean_std_over_subjects(noload_patellofemoraljoint_RFx)
#Fy
loaded_patellofemoraljoint_RFy = utils.normalize_direction_data(loaded_reaction_force_dictionary['patellofemoral_joint_Fy'],gl_noload,direction=False)
noload_patellofemoraljoint_RFy = utils.normalize_direction_data(noload_reaction_force_dictionary['patellofemoral_joint_Fy'],gl_noload,direction=False)
mean_loaded_patellofemoraljoint_RFy,std_loaded_patellofemoraljoint_RFy = utils.mean_std_over_subjects(loaded_patellofemoraljoint_RFy)
mean_noload_patellofemoraljoint_RFy,std_noload_patellofemoraljoint_RFy = utils.mean_std_over_subjects(noload_patellofemoraljoint_RFy)
#Fz
loaded_patellofemoraljoint_RFz = utils.normalize_direction_data(loaded_reaction_force_dictionary['patellofemoral_joint_Fz'],gl_noload,direction=False)
noload_patellofemoraljoint_RFz = utils.normalize_direction_data(noload_reaction_force_dictionary['patellofemoral_joint_Fz'],gl_noload,direction=False)
mean_loaded_patellofemoraljoint_RFz,std_loaded_patellofemoraljoint_RFz = utils.mean_std_over_subjects(loaded_patellofemoraljoint_RFz)
mean_noload_patellofemoraljoint_RFz,std_noload_patellofemoraljoint_RFz = utils.mean_std_over_subjects(noload_patellofemoraljoint_RFz)
# ankle joint reaction moment
loaded_anklejoint_RMz = utils.normalize_direction_data(loaded_reaction_moment_dictionary['ankle_joint_Mz'],gl_noload,direction=False)
noload_anklejoint_RMz = utils.normalize_direction_data(noload_reaction_moment_dictionary['ankle_joint_Mz'],gl_noload,direction=False)
mean_loaded_anklejoint_RMz,std_loaded_anklejoint_RMz = utils.mean_std_over_subjects(loaded_anklejoint_RMz)
mean_noload_anklejoint_RMz,std_noload_anklejoint_RMz = utils.mean_std_over_subjects(noload_anklejoint_RMz)
# Fx
loaded_anklejoint_RFx = utils.normalize_direction_data(loaded_reaction_force_dictionary['ankle_joint_Fx'],gl_noload,direction=False)
noload_anklejoint_RFx = utils.normalize_direction_data(noload_reaction_force_dictionary['ankle_joint_Fx'],gl_noload,direction=False)
mean_loaded_anklejoint_RFx,std_loaded_anklejoint_RFx = utils.mean_std_over_subjects(loaded_anklejoint_RFx)
mean_noload_anklejoint_RFx,std_noload_anklejoint_RFx = utils.mean_std_over_subjects(noload_anklejoint_RFx)
# Fy
loaded_anklejoint_RFy = utils.normalize_direction_data(loaded_reaction_force_dictionary['ankle_joint_Fy'],gl_noload,direction=False)
noload_anklejoint_RFy = utils.normalize_direction_data(noload_reaction_force_dictionary['ankle_joint_Fy'],gl_noload,direction=False)
mean_loaded_anklejoint_RFy,std_loaded_anklejoint_RFy = utils.mean_std_over_subjects(loaded_anklejoint_RFy)
mean_noload_anklejoint_RFy,std_noload_anklejoint_RFy = utils.mean_std_over_subjects(noload_anklejoint_RFy)
# Fz
loaded_anklejoint_RFz = utils.normalize_direction_data(loaded_reaction_force_dictionary['ankle_joint_Fz'],gl_noload,direction=False)
noload_anklejoint_RFz = utils.normalize_direction_data(noload_reaction_force_dictionary['ankle_joint_Fz'],gl_noload,direction=False)
mean_loaded_anklejoint_RFz,std_loaded_anklejoint_RFz = utils.mean_std_over_subjects(loaded_anklejoint_RFz)
mean_noload_anklejoint_RFz,std_noload_anklejoint_RFz = utils.mean_std_over_subjects(noload_anklejoint_RFz)

#####################################################################################
# Write final data to csv file.
# TODO: optimize data saving method.
# Reaction Moment
# Headers
Headers = [ 'mean_loaded_backjoint_RMz','std_loaded_backjoint_RMz','mean_noload_backjoint_RMz','std_noload_backjoint_RMz',\
            'mean_loaded_duct_tape_joint_RMz','std_loaded_duct_tape_joint_RMz',\
            'mean_loaded_hipjoint_RMz','std_loaded_hipjoint_RMz','mean_noload_hipjoint_RMz','std_noload_hipjoint_RMz',\
            'mean_loaded_kneejoint_RMx','std_loaded_kneejoint_RMx','mean_noload_kneejoint_RMx','std_noload_kneejoint_RMx',\
            'mean_loaded_kneejoint_RMy','std_loaded_kneejoint_RMy','mean_noload_kneejoint_RMy','std_noload_kneejoint_RMy',\
            'mean_loaded_kneejoint_RMz','std_loaded_kneejoint_RMz','mean_noload_kneejoint_RMz','std_noload_kneejoint_RMz',\
            'mean_loaded_patellofemoraljoint_RMx','std_loaded_patellofemoraljoint_RMx','mean_noload_patellofemoraljoint_RMx','std_noload_patellofemoraljoint_RMx',\
            'mean_loaded_patellofemoraljoint_RMy','std_loaded_patellofemoraljoint_RMy','mean_noload_patellofemoraljoint_RMy','std_noload_patellofemoraljoint_RMy',\
            'mean_loaded_patellofemoraljoint_RMz','std_loaded_patellofemoraljoint_RMz','mean_noload_patellofemoraljoint_RMz','std_noload_patellofemoraljoint_RMz',\
            'mean_loaded_anklejoint_RMz','std_loaded_anklejoint_RMz','mean_noload_anklejoint_RMz','std_noload_anklejoint_RMz']
# Dataset
Data =[mean_loaded_backjoint_RMz,std_loaded_backjoint_RMz,mean_noload_backjoint_RMz,std_noload_backjoint_RMz,\
      mean_loaded_duct_tape_joint_RMz,std_loaded_duct_tape_joint_RMz,\
      mean_loaded_hipjoint_RMz,std_loaded_hipjoint_RMz,mean_noload_hipjoint_RMz,std_noload_hipjoint_RMz,\
      mean_loaded_kneejoint_RMx,std_loaded_kneejoint_RMx,mean_noload_kneejoint_RMx,std_noload_kneejoint_RMx,\
      mean_loaded_kneejoint_RMy,std_loaded_kneejoint_RMy,mean_noload_kneejoint_RMy,std_noload_kneejoint_RMy,\
      mean_loaded_kneejoint_RMz,std_loaded_kneejoint_RMz,mean_noload_kneejoint_RMz,std_noload_kneejoint_RMz,\
      mean_loaded_patellofemoraljoint_RMx,std_loaded_patellofemoraljoint_RMx,mean_noload_patellofemoraljoint_RMx,std_noload_patellofemoraljoint_RMx,\
      mean_loaded_patellofemoraljoint_RMy,std_loaded_patellofemoraljoint_RMy,mean_noload_patellofemoraljoint_RMy,std_noload_patellofemoraljoint_RMy,\
      mean_loaded_patellofemoraljoint_RMz,std_loaded_patellofemoraljoint_RMz,mean_noload_patellofemoraljoint_RMz,std_noload_patellofemoraljoint_RMz,\
      mean_loaded_anklejoint_RMz,std_loaded_anklejoint_RMz,mean_noload_anklejoint_RMz,std_noload_anklejoint_RMz]
# List of numpy vectors to a numpy ndarray and save to csv file
Data = utils.vec2mat(Data)
with open(r'.\Data\RRA\jrm_final_data.csv', 'wb') as f:
  f.write(bytes(utils.listToString(Headers)+'\n','UTF-8'))
  np.savetxt(f, Data, fmt='%s', delimiter=",")

# Reaction Force
# Headers
Headers = [ 'mean_loaded_backjoint_RMz','std_loaded_backjoint_RMz','mean_noload_backjoint_RMz','std_noload_backjoint_RMz',\
            'mean_loaded_duct_tape_joint_RMz','std_loaded_duct_tape_joint_RMz',\
            'mean_loaded_hipjoint_RMz','std_loaded_hipjoint_RMz','mean_noload_hipjoint_RMz','std_noload_hipjoint_RMz',\
            'mean_loaded_kneejoint_RMx','std_loaded_kneejoint_RMx','mean_noload_kneejoint_RMx','std_noload_kneejoint_RMx',\
            'mean_loaded_kneejoint_RMy','std_loaded_kneejoint_RMy','mean_noload_kneejoint_RMy','std_noload_kneejoint_RMy',\
            'mean_loaded_kneejoint_RMz','std_loaded_kneejoint_RMz','mean_noload_kneejoint_RMz','std_noload_kneejoint_RMz',\
            'mean_loaded_patellofemoraljoint_RMx','std_loaded_patellofemoraljoint_RMx','mean_noload_patellofemoraljoint_RMx','std_noload_patellofemoraljoint_RMx',\
            'mean_loaded_patellofemoraljoint_RMy','std_loaded_patellofemoraljoint_RMy','mean_noload_patellofemoraljoint_RMy','std_noload_patellofemoraljoint_RMy',\
            'mean_loaded_patellofemoraljoint_RMz','std_loaded_patellofemoraljoint_RMz','mean_noload_patellofemoraljoint_RMz','std_noload_patellofemoraljoint_RMz',\
            'mean_loaded_anklejoint_RMz','std_loaded_anklejoint_RMz','mean_noload_anklejoint_RMz','std_noload_anklejoint_RMz']
# Dataset
Data =[mean_loaded_backjoint_RFz,std_loaded_backjoint_RFz,mean_noload_backjoint_RFz,std_noload_backjoint_RFz,\
      mean_loaded_hipjoint_RFx,std_loaded_hipjoint_RFx,mean_noload_hipjoint_RFx,std_noload_hipjoint_RFx,\
      mean_loaded_hipjoint_RFy,std_loaded_hipjoint_RFy,mean_noload_hipjoint_RFy,std_noload_hipjoint_RFy,\
      mean_loaded_hipjoint_RFz,std_loaded_hipjoint_RFz,mean_noload_hipjoint_RFz,std_noload_hipjoint_RFz,\
      mean_loaded_kneejoint_RFx,std_loaded_kneejoint_RFx,mean_noload_kneejoint_RFx,std_noload_kneejoint_RFx,\
      mean_loaded_kneejoint_RFy,std_loaded_kneejoint_RFy,mean_noload_kneejoint_RFy,std_noload_kneejoint_RFy,\
      mean_loaded_kneejoint_RFz,std_loaded_kneejoint_RFz,mean_noload_kneejoint_RFz,std_noload_kneejoint_RFz,\
      mean_loaded_patellofemoraljoint_RFx,std_loaded_patellofemoraljoint_RFx,mean_noload_patellofemoraljoint_RFx,std_noload_patellofemoraljoint_RFx,\
      mean_loaded_patellofemoraljoint_RFy,std_loaded_patellofemoraljoint_RFy,mean_noload_patellofemoraljoint_RFy,std_noload_patellofemoraljoint_RFy,\
      mean_loaded_patellofemoraljoint_RFz,std_loaded_patellofemoraljoint_RFz,mean_noload_patellofemoraljoint_RFz,std_noload_patellofemoraljoint_RFz,\
      mean_loaded_anklejoint_RFx,std_loaded_anklejoint_RFx,mean_noload_anklejoint_RFx,std_noload_anklejoint_RFx,\
      mean_loaded_anklejoint_RFy,std_loaded_anklejoint_RFy,mean_noload_anklejoint_RFy,std_noload_anklejoint_RFy,\
      mean_loaded_anklejoint_RFz,std_loaded_anklejoint_RFz,mean_noload_anklejoint_RFz,std_noload_anklejoint_RFz]
# List of numpy vectors to a numpy ndarray and save to csv file
Data = utils.vec2mat(Data)
with open(r'.\Data\RRA\jrf_final_data.csv', 'wb') as f:
  f.write(bytes(utils.listToString(Headers)+'\n','UTF-8'))
  np.savetxt(f, Data, fmt='%s', delimiter=",")

#####################################################################################
# Plots
# joint reaction moment dictionary in saggital plane
# back joint reaction moment plot dictionaries
back_joint_RMz_loaded_plot_dic = {'pgc':gait_cycle,'avg':utils.smooth(mean_loaded_backjoint_RMz,9),'label':'Loaded',
                        'std':utils.smooth(std_loaded_backjoint_RMz,9),'avg_toeoff':loaded_mean_toe_off}
back_joint_RMz_noload_plot_dic = {'pgc':gait_cycle,'avg':utils.smooth(mean_noload_backjoint_RMz,9),'label':'Noload',
                        'std':utils.smooth(std_noload_backjoint_RMz,9),'avg_toeoff':noload_mean_toe_off}
# RFz
back_joint_RFz_loaded_plot_dic = {'pgc':gait_cycle,'avg':utils.smooth(mean_loaded_backjoint_RFz,9),'label':'Loaded',
                        'std':utils.smooth(std_loaded_backjoint_RFz,9),'avg_toeoff':loaded_mean_toe_off}
back_joint_RFz_noload_plot_dic = {'pgc':gait_cycle,'avg':utils.smooth(mean_noload_backjoint_RFz,9),'label':'Noload',
                        'std':utils.smooth(std_noload_backjoint_RFz,9),'avg_toeoff':noload_mean_toe_off}

# duct tape reaction moment plot dictionaries
duct_tape_joint_RMz_loaded_plot_dic = {'pgc':gait_cycle,'avg':utils.smooth(mean_loaded_duct_tape_joint_RMz,9),'label':'Loaded',
                        'std':utils.smooth(std_loaded_duct_tape_joint_RMz,9),'avg_toeoff':loaded_mean_toe_off}
# hip joint reaction moment plot dictionaries
hip_joint_RMz_loaded_plot_dic = {'pgc':gait_cycle,'avg':utils.smooth(mean_loaded_hipjoint_RMz,9),'label':'Loaded',
                        'std':utils.smooth(std_loaded_hipjoint_RMz,9),'avg_toeoff':loaded_mean_toe_off}
hip_joint_RMz_noload_plot_dic = {'pgc':gait_cycle,'avg':utils.smooth(mean_noload_hipjoint_RMz,9),'label':'Noload',
                        'std':utils.smooth(std_noload_hipjoint_RMz,9),'avg_toeoff':noload_mean_toe_off}
#RFx
hip_joint_RFx_loaded_plot_dic = {'pgc':gait_cycle,'avg':utils.smooth(mean_loaded_hipjoint_RFx,9),'label':'Loaded',
                        'std':utils.smooth(std_loaded_hipjoint_RFx,9),'avg_toeoff':loaded_mean_toe_off}
hip_joint_RFx_noload_plot_dic = {'pgc':gait_cycle,'avg':utils.smooth(mean_noload_hipjoint_RFx,9),'label':'Noload',
                        'std':utils.smooth(std_noload_hipjoint_RFx,9),'avg_toeoff':noload_mean_toe_off}
#RFy
hip_joint_RFy_loaded_plot_dic = {'pgc':gait_cycle,'avg':utils.smooth(mean_loaded_hipjoint_RFy,9),'label':'Loaded',
                        'std':utils.smooth(std_loaded_hipjoint_RFy,9),'avg_toeoff':loaded_mean_toe_off}
hip_joint_RFy_noload_plot_dic = {'pgc':gait_cycle,'avg':utils.smooth(mean_noload_hipjoint_RFy,9),'label':'Noload',
                        'std':utils.smooth(std_noload_hipjoint_RFy,9),'avg_toeoff':noload_mean_toe_off}
#RFz
hip_joint_RFz_loaded_plot_dic = {'pgc':gait_cycle,'avg':utils.smooth(mean_loaded_hipjoint_RFz,9),'label':'Loaded',
                        'std':utils.smooth(std_loaded_hipjoint_RFz,9),'avg_toeoff':loaded_mean_toe_off}
hip_joint_RFz_noload_plot_dic = {'pgc':gait_cycle,'avg':utils.smooth(mean_noload_hipjoint_RFz,9),'label':'Noload',
                        'std':utils.smooth(std_noload_hipjoint_RFz,9),'avg_toeoff':noload_mean_toe_off}

# knee joint reaction moment plot dictionaries
#Mx
knee_joint_RMx_loaded_plot_dic = {'pgc':gait_cycle,'avg':utils.smooth(mean_loaded_kneejoint_RMx,9),'label':'Loaded',
                        'std':utils.smooth(std_loaded_kneejoint_RMx,9),'avg_toeoff':loaded_mean_toe_off}
knee_joint_RMx_noload_plot_dic = {'pgc':gait_cycle,'avg':utils.smooth(mean_noload_kneejoint_RMx,9),'label':'Noload',
                        'std':utils.smooth(std_noload_kneejoint_RMx,9),'avg_toeoff':noload_mean_toe_off}
#My
knee_joint_RMy_loaded_plot_dic = {'pgc':gait_cycle,'avg':utils.smooth(mean_loaded_kneejoint_RMy,9),'label':'Loaded',
                        'std':utils.smooth(std_loaded_kneejoint_RMy,9),'avg_toeoff':loaded_mean_toe_off}
knee_joint_RMy_noload_plot_dic = {'pgc':gait_cycle,'avg':utils.smooth(mean_noload_kneejoint_RMy,9),'label':'Noload',
                        'std':utils.smooth(std_noload_kneejoint_RMy,9),'avg_toeoff':noload_mean_toe_off}
#Mz
knee_joint_RMz_loaded_plot_dic = {'pgc':gait_cycle,'avg':utils.smooth(mean_loaded_kneejoint_RMz,9),'label':'Loaded',
                        'std':utils.smooth(std_loaded_kneejoint_RMz,9),'avg_toeoff':loaded_mean_toe_off}
knee_joint_RMz_noload_plot_dic = {'pgc':gait_cycle,'avg':utils.smooth(mean_noload_kneejoint_RMz,9),'label':'Noload',
                        'std':utils.smooth(std_noload_kneejoint_RMz,9),'avg_toeoff':noload_mean_toe_off}
#Fx
knee_joint_RFx_loaded_plot_dic = {'pgc':gait_cycle,'avg':utils.smooth(mean_loaded_kneejoint_RMx,9),'label':'Loaded',
                        'std':utils.smooth(std_loaded_kneejoint_RMx,9),'avg_toeoff':loaded_mean_toe_off}
knee_joint_RFx_noload_plot_dic = {'pgc':gait_cycle,'avg':utils.smooth(mean_noload_kneejoint_RMx,9),'label':'Noload',
                        'std':utils.smooth(std_noload_kneejoint_RMx,9),'avg_toeoff':noload_mean_toe_off}
#Fy
knee_joint_RFy_loaded_plot_dic = {'pgc':gait_cycle,'avg':utils.smooth(mean_loaded_kneejoint_RFy,9),'label':'Loaded',
                        'std':utils.smooth(std_loaded_kneejoint_RFy,9),'avg_toeoff':loaded_mean_toe_off}
knee_joint_RFy_noload_plot_dic = {'pgc':gait_cycle,'avg':utils.smooth(mean_noload_kneejoint_RFy,9),'label':'Noload',
                        'std':utils.smooth(std_noload_kneejoint_RFy,9),'avg_toeoff':noload_mean_toe_off}
#Fz
knee_joint_RFz_loaded_plot_dic = {'pgc':gait_cycle,'avg':utils.smooth(mean_loaded_kneejoint_RFz,9),'label':'Loaded',
                        'std':utils.smooth(std_loaded_kneejoint_RFz,9),'avg_toeoff':loaded_mean_toe_off}
knee_joint_RFz_noload_plot_dic = {'pgc':gait_cycle,'avg':utils.smooth(mean_noload_kneejoint_RFz,9),'label':'Noload',
                        'std':utils.smooth(std_noload_kneejoint_RFz,9),'avg_toeoff':noload_mean_toe_off}

# patellofemoral joint reaction moment plot dictionaries
#Mx
patellofemoral_joint_RMx_loaded_plot_dic = {'pgc':gait_cycle,'avg':utils.smooth(mean_loaded_patellofemoraljoint_RMx,9),'label':'Loaded',
                        'std':utils.smooth(std_loaded_patellofemoraljoint_RMx,9),'avg_toeoff':loaded_mean_toe_off}
patellofemoral_joint_RMx_noload_plot_dic = {'pgc':gait_cycle,'avg':utils.smooth(mean_noload_patellofemoraljoint_RMx,9),'label':'Noload',
                        'std':utils.smooth(std_noload_patellofemoraljoint_RMx,9),'avg_toeoff':noload_mean_toe_off}
#My
patellofemoral_joint_RMy_loaded_plot_dic = {'pgc':gait_cycle,'avg':utils.smooth(mean_loaded_patellofemoraljoint_RMy,9),'label':'Loaded',
                        'std':utils.smooth(std_loaded_patellofemoraljoint_RMy,9),'avg_toeoff':loaded_mean_toe_off}
patellofemoral_joint_RMy_noload_plot_dic = {'pgc':gait_cycle,'avg':utils.smooth(mean_noload_patellofemoraljoint_RMy,9),'label':'Noload',
                        'std':utils.smooth(std_noload_patellofemoraljoint_RMy,9),'avg_toeoff':noload_mean_toe_off}
#Mz
patellofemoral_joint_RMz_loaded_plot_dic = {'pgc':gait_cycle,'avg':utils.smooth(mean_loaded_patellofemoraljoint_RMz,9),'label':'Loaded',
                        'std':utils.smooth(std_loaded_patellofemoraljoint_RMz,9),'avg_toeoff':loaded_mean_toe_off}
patellofemoral_joint_RMz_noload_plot_dic = {'pgc':gait_cycle,'avg':utils.smooth(mean_noload_patellofemoraljoint_RMz,9),'label':'Noload',
                        'std':utils.smooth(std_noload_patellofemoraljoint_RMz,9),'avg_toeoff':noload_mean_toe_off}
#Fx
patellofemoral_joint_RFx_loaded_plot_dic = {'pgc':gait_cycle,'avg':utils.smooth(mean_loaded_patellofemoraljoint_RFx,9),'label':'Loaded',
                        'std':utils.smooth(std_loaded_patellofemoraljoint_RFx,9),'avg_toeoff':loaded_mean_toe_off}
patellofemoral_joint_RFx_noload_plot_dic = {'pgc':gait_cycle,'avg':utils.smooth(mean_noload_patellofemoraljoint_RFx,9),'label':'Noload',
                        'std':utils.smooth(std_noload_patellofemoraljoint_RFx,9),'avg_toeoff':noload_mean_toe_off}
#Fy
patellofemoral_joint_RFy_loaded_plot_dic = {'pgc':gait_cycle,'avg':utils.smooth(mean_loaded_patellofemoraljoint_RFy,9),'label':'Loaded',
                        'std':utils.smooth(std_loaded_patellofemoraljoint_RFy,9),'avg_toeoff':loaded_mean_toe_off}
patellofemoral_joint_RFy_noload_plot_dic = {'pgc':gait_cycle,'avg':utils.smooth(mean_noload_patellofemoraljoint_RFy,9),'label':'Noload',
                        'std':utils.smooth(std_noload_patellofemoraljoint_RFy,9),'avg_toeoff':noload_mean_toe_off}
#Fz
patellofemoral_joint_RFz_loaded_plot_dic = {'pgc':gait_cycle,'avg':utils.smooth(mean_loaded_patellofemoraljoint_RFz,9),'label':'Loaded',
                        'std':utils.smooth(std_loaded_patellofemoraljoint_RMz,9),'avg_toeoff':loaded_mean_toe_off}
patellofemoral_joint_RFz_noload_plot_dic = {'pgc':gait_cycle,'avg':utils.smooth(mean_noload_patellofemoraljoint_RFz,9),'label':'Noload',
                        'std':utils.smooth(std_noload_patellofemoraljoint_RFz,9),'avg_toeoff':noload_mean_toe_off}

# ankle joint reaction moment plot dictionaries
ankle_joint_RMz_loaded_plot_dic = {'pgc':gait_cycle,'avg':utils.smooth(mean_loaded_anklejoint_RMz,9),'label':'Loaded',
                        'std':utils.smooth(std_loaded_anklejoint_RMz,9),'avg_toeoff':loaded_mean_toe_off}
ankle_joint_RMz_noload_plot_dic = {'pgc':gait_cycle,'avg':utils.smooth(mean_noload_anklejoint_RMz,9),'label':'Noload',
                        'std':utils.smooth(std_noload_anklejoint_RMz,9),'avg_toeoff':noload_mean_toe_off}
#Fx
ankle_joint_RFx_loaded_plot_dic = {'pgc':gait_cycle,'avg':utils.smooth(mean_loaded_anklejoint_RFx,9),'label':'Loaded',
                        'std':utils.smooth(std_loaded_anklejoint_RFx,9),'avg_toeoff':loaded_mean_toe_off}
ankle_joint_RFx_noload_plot_dic = {'pgc':gait_cycle,'avg':utils.smooth(mean_noload_anklejoint_RFx,9),'label':'Noload',
                        'std':utils.smooth(std_noload_anklejoint_RFx,9),'avg_toeoff':noload_mean_toe_off}
#Fy
ankle_joint_RFy_loaded_plot_dic = {'pgc':gait_cycle,'avg':utils.smooth(mean_loaded_anklejoint_RFy,9),'label':'Loaded',
                        'std':utils.smooth(std_loaded_anklejoint_RFy,9),'avg_toeoff':loaded_mean_toe_off}
ankle_joint_RFy_noload_plot_dic = {'pgc':gait_cycle,'avg':utils.smooth(mean_noload_anklejoint_RFy,9),'label':'Noload',
                        'std':utils.smooth(std_noload_anklejoint_RFy,9),'avg_toeoff':noload_mean_toe_off}
#Fz
ankle_joint_RFz_loaded_plot_dic = {'pgc':gait_cycle,'avg':utils.smooth(mean_loaded_anklejoint_RFz,9),'label':'Loaded',
                        'std':utils.smooth(std_loaded_anklejoint_RFz,9),'avg_toeoff':loaded_mean_toe_off}
ankle_joint_RFz_noload_plot_dic = {'pgc':gait_cycle,'avg':utils.smooth(mean_noload_anklejoint_RFz,9),'label':'Noload',
                        'std':utils.smooth(std_noload_anklejoint_RFz,9),'avg_toeoff':noload_mean_toe_off}

#*****************************
# back joint reaction moment figure
fig, ax = plt.subplots(num='Back Joint Reaction Moment',figsize=(6.4, 4.8))
utils.plot_shaded_avg(plot_dic=back_joint_RMz_loaded_plot_dic,color='k')
utils.plot_shaded_avg(plot_dic=back_joint_RMz_noload_plot_dic,toeoff_color='xkcd:shamrock green',color='xkcd:irish green')
plt.legend(loc='best',frameon=False)
#plt.ylim((-1,2))
plt.xlabel('gait cycle (%)')
plt.ylabel('joint reaction moment (N-m/kg)')
utils.no_top_right(ax)
plt.show()
fig.tight_layout(h_pad=-4.0, w_pad=-4.0)
fig.savefig('./Figures/JRF/JRF_BackJoint_ReactionMoment.pdf',orientation='landscape',bbox_inches='tight')
# back joint reaction force figure
fig, ax = plt.subplots(num='Back Joint Reaction Force',figsize=(6.4, 4.8))
utils.plot_shaded_avg(plot_dic=back_joint_RFz_loaded_plot_dic,color='k')
utils.plot_shaded_avg(plot_dic=back_joint_RFz_noload_plot_dic,toeoff_color='xkcd:shamrock green',color='xkcd:irish green')
plt.legend(loc='best',frameon=False)
#plt.ylim((-1,2))
plt.xlabel('gait cycle (%)')
plt.ylabel('joint reaction Force (N/kg)')
utils.no_top_right(ax)
plt.show()
fig.tight_layout(h_pad=-4.0, w_pad=-4.0)
fig.savefig('./Figures/JRF/JRF_BackJoint_ReactionForce.pdf',orientation='landscape',bbox_inches='tight')

# duct tape joint reaction moment figure
fig, ax = plt.subplots(num='Duct Tape Joint Reaction Moment',figsize=(6.4, 4.8))
utils.plot_shaded_avg(plot_dic=duct_tape_joint_RMz_loaded_plot_dic,color='k')
plt.legend(loc='best',frameon=False)
#plt.ylim((-1,2))
plt.xlabel('gait cycle (%)')
plt.ylabel('joint reaction moment (N-m/kg)')
utils.no_top_right(ax)
plt.show()
fig.tight_layout(h_pad=-4.0, w_pad=-4.0)
fig.savefig('./Figures/JRF/JRF_Duct_Tape_Joint_ReactionMoment.pdf',orientation='landscape',bbox_inches='tight')

# hip joint reaction moment figure
fig, ax = plt.subplots(num='Hip Joint Reaction Moment',figsize=(6.4, 4.8))
utils.plot_shaded_avg(plot_dic=hip_joint_RMz_loaded_plot_dic,color='k')
utils.plot_shaded_avg(plot_dic=hip_joint_RMz_noload_plot_dic,toeoff_color='xkcd:shamrock green',color='xkcd:irish green')
plt.legend(loc='best',frameon=False)
#plt.ylim((-1,2))
plt.xlabel('gait cycle (%)')
plt.ylabel('joint reaction moment (N-m/kg)')
utils.no_top_right(ax)
plt.show()
fig.tight_layout(h_pad=-4.0, w_pad=-4.0)
fig.savefig('./Figures/JRF/JRF_HipJoint_ReactionMoment.pdf',orientation='landscape',bbox_inches='tight')
# hip joint reaction force figure
# Fx
fig, ax = plt.subplots(num='Hip Joint Reaction force (Fx)',figsize=(6.4, 4.8))
utils.plot_shaded_avg(plot_dic=hip_joint_RFx_loaded_plot_dic,color='k')
utils.plot_shaded_avg(plot_dic=hip_joint_RFx_noload_plot_dic,toeoff_color='xkcd:shamrock green',color='xkcd:irish green')
plt.legend(loc='best',frameon=False)
#plt.ylim((-1,2))
plt.xlabel('gait cycle (%)')
plt.ylabel('joint reaction force (N/kg)')
utils.no_top_right(ax)
plt.show()
fig.tight_layout(h_pad=-4.0, w_pad=-4.0)
fig.savefig('./Figures/JRF/JRF_HipJoint_ReactionForce_X.pdf',orientation='landscape',bbox_inches='tight')
# Fy
fig, ax = plt.subplots(num='Hip Joint Reaction force (Fy)',figsize=(6.4, 4.8))
utils.plot_shaded_avg(plot_dic=hip_joint_RFy_loaded_plot_dic,color='k')
utils.plot_shaded_avg(plot_dic=hip_joint_RFy_noload_plot_dic,toeoff_color='xkcd:shamrock green',color='xkcd:irish green')
plt.legend(loc='best',frameon=False)
#plt.ylim((-1,2))
plt.xlabel('gait cycle (%)')
plt.ylabel('joint reaction force (N/kg)')
utils.no_top_right(ax)
plt.show()
fig.tight_layout(h_pad=-4.0, w_pad=-4.0)
fig.savefig('./Figures/JRF/JRF_HipJoint_ReactionForce_Y.pdf',orientation='landscape',bbox_inches='tight')
# Fz
fig, ax = plt.subplots(num='Hip Joint Reaction force (Fz)',figsize=(6.4, 4.8))
utils.plot_shaded_avg(plot_dic=hip_joint_RFz_loaded_plot_dic,color='k')
utils.plot_shaded_avg(plot_dic=hip_joint_RFz_noload_plot_dic,toeoff_color='xkcd:shamrock green',color='xkcd:irish green')
plt.legend(loc='best',frameon=False)
#plt.ylim((-1,2))
plt.xlabel('gait cycle (%)')
plt.ylabel('joint reaction force (N/kg)')
utils.no_top_right(ax)
plt.show()
fig.tight_layout(h_pad=-4.0, w_pad=-4.0)
fig.savefig('./Figures/JRF/JRF_HipJoint_ReactionForce_Z.pdf',orientation='landscape',bbox_inches='tight')

# knee joint reaction moment figure
#Mx
fig, ax = plt.subplots(num='Knee Joint Reaction Moment Mx',figsize=(6.4, 4.8))
utils.plot_shaded_avg(plot_dic=knee_joint_RMx_loaded_plot_dic,color='k')
utils.plot_shaded_avg(plot_dic=knee_joint_RMx_noload_plot_dic,toeoff_color='xkcd:shamrock green',color='xkcd:irish green')
plt.legend(loc='best',frameon=False)
#plt.ylim((-1,2))
plt.xlabel('gait cycle (%)')
plt.ylabel('joint reaction moment (N-m/kg)')
utils.no_top_right(ax)
plt.show()
fig.tight_layout(h_pad=-4.0, w_pad=-4.0)
fig.savefig('./Figures/JRF/JRF_KneeJoint_ReactionMoment_X.pdf',orientation='landscape',bbox_inches='tight')

#My
fig, ax = plt.subplots(num='Knee Joint Reaction Moment My',figsize=(6.4, 4.8))
utils.plot_shaded_avg(plot_dic=knee_joint_RMy_loaded_plot_dic,color='k')
utils.plot_shaded_avg(plot_dic=knee_joint_RMy_noload_plot_dic,toeoff_color='xkcd:shamrock green',color='xkcd:irish green')
plt.legend(loc='best',frameon=False)
#plt.ylim((-1,2))
plt.xlabel('gait cycle (%)')
plt.ylabel('joint reaction moment (N-m/kg)')
utils.no_top_right(ax)
plt.show()
fig.tight_layout(h_pad=-4.0, w_pad=-4.0)
fig.savefig('./Figures/JRF/JRF_KneeJoint_ReactionMoment_Y.pdf',orientation='landscape',bbox_inches='tight')

#Mz
fig, ax = plt.subplots(num='Knee Joint Reaction Moment Mz',figsize=(6.4, 4.8))
utils.plot_shaded_avg(plot_dic=knee_joint_RMz_loaded_plot_dic,color='k')
utils.plot_shaded_avg(plot_dic=knee_joint_RMz_noload_plot_dic,toeoff_color='xkcd:shamrock green',color='xkcd:irish green')
plt.legend(loc='best',frameon=False)
#plt.ylim((-1,2))
plt.xlabel('gait cycle (%)')
plt.ylabel('joint reaction moment (N-m/kg)')
utils.no_top_right(ax)
plt.show()
fig.tight_layout(h_pad=-4.0, w_pad=-4.0)
fig.savefig('./Figures/JRF/JRF_KneeJoint_ReactionMoment_Z.pdf',orientation='landscape',bbox_inches='tight')
#Fx
fig, ax = plt.subplots(num='Knee Joint Reaction Force Fx',figsize=(6.4, 4.8))
utils.plot_shaded_avg(plot_dic=knee_joint_RFx_loaded_plot_dic,color='k')
utils.plot_shaded_avg(plot_dic=knee_joint_RFx_noload_plot_dic,toeoff_color='xkcd:shamrock green',color='xkcd:irish green')
plt.legend(loc='best',frameon=False)
#plt.ylim((-1,2))
plt.xlabel('gait cycle (%)')
plt.ylabel('joint reaction force (N/kg)')
utils.no_top_right(ax)
plt.show()
fig.tight_layout(h_pad=-4.0, w_pad=-4.0)
fig.savefig('./Figures/JRF/JRF_KneeJoint_ReactionForce_X.pdf',orientation='landscape',bbox_inches='tight')

#Fy
fig, ax = plt.subplots(num='Knee Joint Reaction Force Fy',figsize=(6.4, 4.8))
utils.plot_shaded_avg(plot_dic=knee_joint_RFy_loaded_plot_dic,color='k')
utils.plot_shaded_avg(plot_dic=knee_joint_RFy_noload_plot_dic,toeoff_color='xkcd:shamrock green',color='xkcd:irish green')
plt.legend(loc='best',frameon=False)
#plt.ylim((-1,2))
plt.xlabel('gait cycle (%)')
plt.ylabel('joint reaction force (N/kg)')
utils.no_top_right(ax)
plt.show()
fig.tight_layout(h_pad=-4.0, w_pad=-4.0)
fig.savefig('./Figures/JRF/JRF_KneeJoint_ReactionForce_Y.pdf',orientation='landscape',bbox_inches='tight')

#Fz
fig, ax = plt.subplots(num='Knee Joint Reaction Force Fz',figsize=(6.4, 4.8))
utils.plot_shaded_avg(plot_dic=knee_joint_RFz_loaded_plot_dic,color='k')
utils.plot_shaded_avg(plot_dic=knee_joint_RFz_noload_plot_dic,toeoff_color='xkcd:shamrock green',color='xkcd:irish green')
plt.legend(loc='best',frameon=False)
#plt.ylim((-1,2))
plt.xlabel('gait cycle (%)')
plt.ylabel('joint reaction force (N/kg)')
utils.no_top_right(ax)
plt.show()
fig.tight_layout(h_pad=-4.0, w_pad=-4.0)
fig.savefig('./Figures/JRF/JRF_KneeJoint_ReactionForce_Z.pdf',orientation='landscape',bbox_inches='tight')

# patellofemoral joint reaction moment figure
#Mx
fig, ax = plt.subplots(num='Patellofemoral Joint Reaction Moment Mx',figsize=(6.4, 4.8))
utils.plot_shaded_avg(plot_dic=patellofemoral_joint_RMx_loaded_plot_dic,color='k')
utils.plot_shaded_avg(plot_dic=patellofemoral_joint_RMx_noload_plot_dic,toeoff_color='xkcd:shamrock green',color='xkcd:irish green')
plt.legend(loc='best',frameon=False)
#plt.ylim((-1,2))
plt.xlabel('gait cycle (%)')
plt.ylabel('joint reaction moment (N-m/kg)')
utils.no_top_right(ax)
plt.show()
fig.tight_layout(h_pad=-4.0, w_pad=-4.0)
fig.savefig('./Figures/JRF/JRF_PatellofemoralJoint_ReactionMoment_X.pdf',orientation='landscape',bbox_inches='tight')
#My
fig, ax = plt.subplots(num='Patellofemoral Joint Reaction Moment My',figsize=(6.4, 4.8))
utils.plot_shaded_avg(plot_dic=patellofemoral_joint_RMy_loaded_plot_dic,color='k')
utils.plot_shaded_avg(plot_dic=patellofemoral_joint_RMy_noload_plot_dic,toeoff_color='xkcd:shamrock green',color='xkcd:irish green')
plt.legend(loc='best',frameon=False)
#plt.ylim((-1,2))
plt.xlabel('gait cycle (%)')
plt.ylabel('joint reaction moment (N-m/kg)')
utils.no_top_right(ax)
plt.show()
fig.tight_layout(h_pad=-4.0, w_pad=-4.0)
fig.savefig('./Figures/JRF/JRF_PatellofemoralJoint_ReactionMoment_Y.pdf',orientation='landscape',bbox_inches='tight')
#Mz
fig, ax = plt.subplots(num='Patellofemoral Joint Reaction Moment Mz',figsize=(6.4, 4.8))
utils.plot_shaded_avg(plot_dic=patellofemoral_joint_RMz_loaded_plot_dic,color='k')
utils.plot_shaded_avg(plot_dic=patellofemoral_joint_RMz_noload_plot_dic,toeoff_color='xkcd:shamrock green',color='xkcd:irish green')
plt.legend(loc='best',frameon=False)
#plt.ylim((-1,2))
plt.xlabel('gait cycle (%)')
plt.ylabel('joint reaction moment (N-m/kg)')
utils.no_top_right(ax)
plt.show()
fig.tight_layout(h_pad=-4.0, w_pad=-4.0)
fig.savefig('./Figures/JRF/JRF_PatellofemoralJoint_ReactionMoment_Z.pdf',orientation='landscape',bbox_inches='tight')
#Fx
fig, ax = plt.subplots(num='Patellofemoral Joint Reaction Force Fx',figsize=(6.4, 4.8))
utils.plot_shaded_avg(plot_dic=patellofemoral_joint_RFx_loaded_plot_dic,color='k')
utils.plot_shaded_avg(plot_dic=patellofemoral_joint_RFx_noload_plot_dic,toeoff_color='xkcd:shamrock green',color='xkcd:irish green')
plt.legend(loc='best',frameon=False)
#plt.ylim((-1,2))
plt.xlabel('gait cycle (%)')
plt.ylabel('joint reaction force (N/kg)')
utils.no_top_right(ax)
plt.show()
fig.tight_layout(h_pad=-4.0, w_pad=-4.0)
fig.savefig('./Figures/JRF/JRF_PatellofemoralJoint_ReactionForce_X.pdf',orientation='landscape',bbox_inches='tight')
#Fy
fig, ax = plt.subplots(num='Patellofemoral Joint Reaction Force Fy',figsize=(6.4, 4.8))
utils.plot_shaded_avg(plot_dic=patellofemoral_joint_RFy_loaded_plot_dic,color='k')
utils.plot_shaded_avg(plot_dic=patellofemoral_joint_RFy_noload_plot_dic,toeoff_color='xkcd:shamrock green',color='xkcd:irish green')
plt.legend(loc='best',frameon=False)
#plt.ylim((-1,2))
plt.xlabel('gait cycle (%)')
plt.ylabel('joint reaction force (N/kg)')
utils.no_top_right(ax)
plt.show()
fig.tight_layout(h_pad=-4.0, w_pad=-4.0)
fig.savefig('./Figures/JRF/JRF_PatellofemoralJoint_ReactionForce_Y.pdf',orientation='landscape',bbox_inches='tight')
#Fz
fig, ax = plt.subplots(num='Patellofemoral Joint Reaction Force Fz',figsize=(6.4, 4.8))
utils.plot_shaded_avg(plot_dic=patellofemoral_joint_RFz_loaded_plot_dic,color='k')
utils.plot_shaded_avg(plot_dic=patellofemoral_joint_RFz_noload_plot_dic,toeoff_color='xkcd:shamrock green',color='xkcd:irish green')
plt.legend(loc='best',frameon=False)
#plt.ylim((-1,2))
plt.xlabel('gait cycle (%)')
plt.ylabel('joint reaction force (N/kg)')
utils.no_top_right(ax)
plt.show()
fig.tight_layout(h_pad=-4.0, w_pad=-4.0)
fig.savefig('./Figures/JRF/JRF_PatellofemoralJoint_ReactionForce_Z.pdf',orientation='landscape',bbox_inches='tight')

# ankle joint reaction moment figure
fig, ax = plt.subplots(num='Ankle Joint Reaction Moment',figsize=(6.4, 4.8))
utils.plot_shaded_avg(plot_dic=ankle_joint_RMz_loaded_plot_dic,color='k')
utils.plot_shaded_avg(plot_dic=ankle_joint_RMz_noload_plot_dic,toeoff_color='xkcd:shamrock green',color='xkcd:irish green')
plt.legend(loc='best',frameon=False)
#plt.ylim((-1,2))
plt.xlabel('gait cycle (%)')
plt.ylabel('joint reaction moment (N-m/kg)')
utils.no_top_right(ax)
plt.show()
fig.tight_layout(h_pad=-4.0, w_pad=-4.0)
fig.savefig('./Figures/JRF/JRF_AnkleJoint_ReactionMoment.pdf',orientation='landscape',bbox_inches='tight')
# Fx
fig, ax = plt.subplots(num='Ankle Joint Reaction Force Fx',figsize=(6.4, 4.8))
utils.plot_shaded_avg(plot_dic=ankle_joint_RFx_loaded_plot_dic,color='k')
utils.plot_shaded_avg(plot_dic=ankle_joint_RFx_noload_plot_dic,toeoff_color='xkcd:shamrock green',color='xkcd:irish green')
plt.legend(loc='best',frameon=False)
#plt.ylim((-1,2))
plt.xlabel('gait cycle (%)')
plt.ylabel('joint reaction force (N/kg)')
utils.no_top_right(ax)
plt.show()
fig.tight_layout(h_pad=-4.0, w_pad=-4.0)
fig.savefig('./Figures/JRF/JRF_AnkleJoint_ReactionForce_X.pdf',orientation='landscape',bbox_inches='tight')
# Fy
fig, ax = plt.subplots(num='Ankle Joint Reaction Force Fy',figsize=(6.4, 4.8))
utils.plot_shaded_avg(plot_dic=ankle_joint_RFy_loaded_plot_dic,color='k')
utils.plot_shaded_avg(plot_dic=ankle_joint_RFy_noload_plot_dic,toeoff_color='xkcd:shamrock green',color='xkcd:irish green')
plt.legend(loc='best',frameon=False)
#plt.ylim((-1,2))
plt.xlabel('gait cycle (%)')
plt.ylabel('joint reaction force (N/kg)')
utils.no_top_right(ax)
plt.show()
fig.tight_layout(h_pad=-4.0, w_pad=-4.0)
fig.savefig('./Figures/JRF/JRF_AnkleJoint_ReactionForce_Y.pdf',orientation='landscape',bbox_inches='tight')
# Fz
fig, ax = plt.subplots(num='Ankle Joint Reaction Force Fz',figsize=(6.4, 4.8))
utils.plot_shaded_avg(plot_dic=ankle_joint_RFz_loaded_plot_dic,color='k')
utils.plot_shaded_avg(plot_dic=ankle_joint_RFz_noload_plot_dic,toeoff_color='xkcd:shamrock green',color='xkcd:irish green')
plt.legend(loc='best',frameon=False)
#plt.ylim((-1,2))
plt.xlabel('gait cycle (%)')
plt.ylabel('joint reaction force (N/kg)')
utils.no_top_right(ax)
plt.show()
fig.tight_layout(h_pad=-4.0, w_pad=-4.0)
fig.savefig('./Figures/JRF/JRF_AnkleJoint_ReactionForce_Z.pdf',orientation='landscape',bbox_inches='tight')
