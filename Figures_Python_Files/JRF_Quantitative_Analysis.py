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
import itertools
import Utils as utils
from Colors import colors as mycolors
#####################################################################################
subjects = ['05','07','09','10','11','12','14']
trials_num = ['01','02','03']
gait_cycle = np.linspace(0,100,1000)
plt.rcParams.update({'font.size': 12})
#####################################################################################
# assisted reaction moments
directory = r'.\Data\Ideal\*_reaction_moments.csv'
files = enumerate(glob.iglob(directory), 1)
jointreaction_dataset = {pathlib.PurePath(f[1]).stem: np.loadtxt(f[1], delimiter=',') for f in files}
bi_noload_RM_dictionary = utils.clasify_data(jointreaction_dataset['biarticular_ideal_noload_reaction_moments'],loadcondition='noload')
bi_loaded_RM_dictionary = utils.clasify_data(jointreaction_dataset['biarticular_ideal_loaded_reaction_moments'],loadcondition='loaded')
mono_noload_RM_dictionary = utils.clasify_data(jointreaction_dataset['monoarticular_ideal_noload_reaction_moments'],loadcondition='noload')
mono_loaded_RM_dictionary = utils.clasify_data(jointreaction_dataset['monoarticular_ideal_loaded_reaction_moments'],loadcondition='loaded')
# assisted reaction forces
directory = r'.\Data\Ideal\*_reaction_forces.csv'
files = enumerate(glob.iglob(directory), 1)
jointreaction_dataset = {pathlib.PurePath(f[1]).stem: np.loadtxt(f[1], delimiter=',') for f in files}
bi_noload_RF_dictionary = utils.clasify_data(jointreaction_dataset['biarticular_ideal_noload_reaction_forces'],loadcondition='noload',forces_name=['Fx','Fy','Fz'])
bi_loaded_RF_dictionary = utils.clasify_data(jointreaction_dataset['biarticular_ideal_loaded_reaction_forces'],loadcondition='loaded',forces_name=['Fx','Fy','Fz'])
mono_noload_RF_dictionary = utils.clasify_data(jointreaction_dataset['monoarticular_ideal_noload_reaction_forces'],loadcondition='noload',forces_name=['Fx','Fy','Fz'])
mono_loaded_RF_dictionary = utils.clasify_data(jointreaction_dataset['monoarticular_ideal_loaded_reaction_forces'],loadcondition='loaded',forces_name=['Fx','Fy','Fz'])
#####################################################################################
# Reading CSV files into a dictionary and constructing gls
# joint reaction moment dataset
directory = r'.\Data\Unassist\*_reaction_moments.csv'
files = enumerate(glob.iglob(directory), 1)
jointreaction_dataset = {pathlib.PurePath(f[1]).stem: np.loadtxt(f[1], delimiter=',') for f in files}
noload_RM_dictionary = utils.clasify_data(jointreaction_dataset['unassist_noload_reaction_moments'],loadcondition='noload')
loaded_RM_dictionary = utils.clasify_data(jointreaction_dataset['unassist_loaded_reaction_moments'],loadcondition='loaded')
# joint reaction force dataset
directory = r'.\Data\Unassist\*_reaction_forces.csv'
files = enumerate(glob.iglob(directory), 1)
jointreaction_dataset = {pathlib.PurePath(f[1]).stem: np.loadtxt(f[1], delimiter=',') for f in files}
noload_RF_dictionary = utils.clasify_data(jointreaction_dataset['unassist_noload_reaction_forces'],loadcondition='noload',forces_name=['Fx','Fy','Fz'])
loaded_RF_dictionary = utils.clasify_data(jointreaction_dataset['unassist_loaded_reaction_forces'],loadcondition='loaded',forces_name=['Fx','Fy','Fz'])
#####################################################################################
# pareto exo force dataset
directory = r'./Data/Pareto/*_reaction_forces.csv'
files = enumerate(glob.iglob(directory), 1)
pareto_jrf_dataset = {pathlib.PurePath(f[1]).stem: np.loadtxt(f[1], delimiter=',') for f in files}
pareto_bi_noload_RF_dictionary = utils.clasify_data(pareto_jrf_dataset['biarticular_paretofront_noload_reaction_forces'],loadcondition='noload',pareto=True,device='biarticular',forces_name=['Fx','Fy','Fz'])
pareto_bi_loaded_RF_dictionary = utils.clasify_data(pareto_jrf_dataset['biarticular_paretofront_loaded_reaction_forces'],loadcondition='loaded',pareto=True,device='biarticular',forces_name=['Fx','Fy','Fz'])
pareto_mono_noload_RF_dictionary = utils.clasify_data(pareto_jrf_dataset['monoarticular_paretofront_noload_reaction_forces'],loadcondition='noload',pareto=True,device='monoarticular',forces_name=['Fx','Fy','Fz'])
pareto_mono_loaded_RF_dictionary = utils.clasify_data(pareto_jrf_dataset['monoarticular_paretofront_loaded_reaction_forces'],loadcondition='loaded',pareto=True,device='monoarticular',forces_name=['Fx','Fy','Fz'])
# indices
bi_loaded_indices = np.array([25,24,23,22,21,17,16,13,12,11,6,1])
mono_loaded_indices = np.array([25,20,15,10,5,4,3,2,1])
bi_noload_indices = np.array([25,24,23,22,21,19,18,17,13,12,11,1])
mono_noload_indices = np.array([25,20,15,14,13,8,7,6,2,1])
#####################################################################################
# gls
gl_noload = {'noload_subject{}_trial{}'.format(i,j): utils.construct_gl_mass_side(subjectno=i,trialno=j,loadcond='noload') for i in subjects for j in trials_num}
gl_loaded = {'loaded_subject{}_trial{}'.format(i,j): utils.construct_gl_mass_side(subjectno=i,trialno=j,loadcond='loaded') for i in subjects for j in trials_num}
#####################################################################################
# Processing Data for Ideal Exoskeleton wrt Unassisted Case
# toe-off
_,_,_,_,noload_subjects_toe_off,loaded_subjects_toe_off = utils.toe_off_avg_std(gl_noload,gl_loaded,subjects=True)

# biarticular noload walking reaction forces analysis
mean_bi_noload_max_jrf_change_dict,std_bi_noload_max_jrf_change_dict = utils.quantify_max_jrf_change(noload_subjects_toe_off,noload_RF_dictionary,bi_noload_RF_dictionary)

# biarticular loaded walking reaction forces analysis
mean_bi_loaded_max_jrf_change_dict,std_bi_loaded_max_jrf_change_dict = utils.quantify_max_jrf_change(loaded_subjects_toe_off,loaded_RF_dictionary,bi_loaded_RF_dictionary)

# monoarticular noload walking reaction forces analysis
mean_mono_noload_max_jrf_change_dict,std_mono_noload_max_jrf_change_dict = utils.quantify_max_jrf_change(noload_subjects_toe_off,noload_RF_dictionary,bi_noload_RF_dictionary)

# monoarticular loaded walking reaction forces analysis
mean_mono_loaded_max_jrf_change_dict,std_mono_loaded_max_jrf_change_dict = utils.quantify_max_jrf_change(loaded_subjects_toe_off,loaded_RF_dictionary,bi_loaded_RF_dictionary)

#####################################################################################
# Processing Data for Torque-limited Exoskeleton wrt Unassisted Case

# biarticular noload walking reaction forces analysis
mean_pareto_bi_noload_max_jrf_change_dict,std_pareto_bi_noload_max_jrf_change_dict =utils.paterofront_quantify_max_jrf_change(noload_subjects_toe_off,bi_noload_indices,pareto_bi_noload_RF_dictionary,noload_RF_dictionary)

# biarticular loaded walking reaction forces analysis
mean_pareto_bi_loaded_max_jrf_change_dict,std_pareto_bi_loaded_max_jrf_change_dict =utils.paterofront_quantify_max_jrf_change(loaded_subjects_toe_off,bi_loaded_indices,pareto_bi_loaded_RF_dictionary,loaded_RF_dictionary)

# monoarticular noload walking reaction forces analysis
mean_pareto_mono_noload_max_jrf_change_dict,std_pareto_mono_noload_max_jrf_change_dict =utils.paterofront_quantify_max_jrf_change(noload_subjects_toe_off,mono_noload_indices,pareto_mono_noload_RF_dictionary,noload_RF_dictionary)

# monoarticular loaded walking reaction forces analysis
mean_pareto_mono_loaded_max_jrf_change_dict,std_pareto_mono_loaded_max_jrf_change_dict =utils.paterofront_quantify_max_jrf_change(loaded_subjects_toe_off,mono_loaded_indices,pareto_mono_loaded_RF_dictionary,loaded_RF_dictionary)

#####################################################################################
# Write to csv files: Ideal exoskeletons
exo_col = [item for item in ['biarticular','monoarticular'] for i in range(2*3)] # loadcondition*joints*forces
load_col = [item for item in ['noload','loaded','noload','loaded'] for i in range(3)] 
joint_col = ['hip','knee','patellofemoral','hip','knee','patellofemoral','hip','knee','patellofemoral','hip','knee','patellofemoral']
string_cols = np.column_stack((load_col,joint_col))
string_cols = np.column_stack((exo_col,string_cols))
#--------------------------------------------------------------------------------------------------------------
bi_noload_max_jrf_change = np.array(list(mean_bi_noload_max_jrf_change_dict.items()))[:,1]
mono_noload_max_jrf_change = np.array(list(mean_mono_noload_max_jrf_change_dict.items()))[:,1]
bi_loaded_max_jrf_change = np.array(list(mean_bi_loaded_max_jrf_change_dict.items()))[:,1]
mono_loaded_max_jrf_change = np.array(list(mean_mono_loaded_max_jrf_change_dict.items()))[:,1]
biarticular_dataset = np.row_stack((bi_noload_max_jrf_change.reshape((3,7*3)),bi_loaded_max_jrf_change.reshape((3,7*3))))
monoarticular_dataset = np.row_stack((mono_noload_max_jrf_change.reshape((3,7*3)),mono_loaded_max_jrf_change.reshape((3,7*3))))
dataset = np.row_stack((biarticular_dataset,monoarticular_dataset))
mean_dataset_string_col_included = np.column_stack((string_cols,dataset))
#--------------------------------------------------------------------------------------------------------------
bi_noload_max_jrf_change = np.array(list(std_bi_noload_max_jrf_change_dict.items()))[:,1]
mono_noload_max_jrf_change = np.array(list(std_mono_noload_max_jrf_change_dict.items()))[:,1]
bi_loaded_max_jrf_change = np.array(list(std_bi_loaded_max_jrf_change_dict.items()))[:,1]
mono_loaded_max_jrf_change = np.array(list(std_mono_loaded_max_jrf_change_dict.items()))[:,1]
biarticular_dataset = np.row_stack((bi_noload_max_jrf_change.reshape((3,7*3)),bi_loaded_max_jrf_change.reshape((3,7*3))))
monoarticular_dataset = np.row_stack((mono_noload_max_jrf_change.reshape((3,7*3)),mono_loaded_max_jrf_change.reshape((3,7*3))))
dataset = np.row_stack((biarticular_dataset,monoarticular_dataset))
std_dataset_string_col_included = np.column_stack((string_cols,dataset))
#--------------------------------------------------------------------------------------------------------------
force_row = ['Fx','Fy','Fz']*7
gait_phase_row = [item for item in['loading response','mid stance','terminal stance','pre swing',\
                  'initial swing','mid swing','terminal swing'] for i in range(3)]
skip_cols = np.row_stack((['DEVICE','LOAD COND','JOINT'],['-','-','-']))
headers = np.row_stack((gait_phase_row,force_row))
headers_skipcols = np.column_stack((skip_cols,headers))
#--------------------------------------------------------------------------------------------------------------
final_dataset = np.row_stack((headers_skipcols,mean_dataset_string_col_included))
directory = r'.\Data\Ideal\JRF_Quantitatives_Mean_Values.csv'
np.savetxt(directory, final_dataset, fmt="%s", delimiter=",")
#--------------------------------------------------------------------------------------------------------------
final_dataset = np.row_stack((headers_skipcols,std_dataset_string_col_included))
directory = r'.\Data\Ideal\JRF_Quantitatives_Std_Values.csv'
np.savetxt(directory, final_dataset, fmt="%s", delimiter=",")

#####################################################################################
# Write to csv files: Torque-limited exoskeletons
#***************************************************************************************************************
# biarticular no load 
skip_col = ['-','-','-']
#--------------------------------------------------------------------------------------------------------------
# columns
mean_joint_col = ['mean hip','mean knee','mean patellofemoral']
std_joint_col = ['std hip','std knee','std patellofemoral']
joint_col = np.concatenate((mean_joint_col,std_joint_col),axis=0)
#--------------------------------------------------------------------------------------------------------------
# rows
gait_phase_row = [item for item in['loading response','mid stance','terminal stance','pre swing',\
                  'initial swing','mid swing','terminal swing'] for i in range(len(bi_noload_indices)*3)]
force_row = ['Fx','Fy','Fz']*len(bi_noload_indices)*7
config_row =  [item for item in bi_noload_indices.tolist() for i in range(3)]*7
rows = np.row_stack((gait_phase_row,config_row))
rows = np.row_stack((rows,force_row))
rows = np.column_stack((skip_col,rows))
#--------------------------------------------------------------------------------------------------------------
# data
mean_bi_noload_max_jrf_change = np.array(list(mean_pareto_bi_noload_max_jrf_change_dict.items()))[:,1]
mean_bi_noload_max_jrf_change = mean_bi_noload_max_jrf_change.reshape((3,len(bi_noload_indices)*7*3),order='C')
std_bi_noload_max_jrf_change = np.array(list(std_pareto_bi_noload_max_jrf_change_dict.items()))[:,1]
std_bi_noload_max_jrf_change = std_bi_noload_max_jrf_change.reshape((3,len(bi_noload_indices)*7*3),order='C')
#--------------------------------------------------------------------------------------------------------------
# save
bi_noload_max_jrf_change = np.row_stack((mean_bi_noload_max_jrf_change,std_bi_noload_max_jrf_change))
bi_noload_dataset = np.column_stack((joint_col,bi_noload_max_jrf_change))
final_dataset = np.row_stack((rows,bi_noload_dataset))
directory = r'.\Data\Pareto\Biarticular_Noload_JRF_Quantitatives.csv'
np.savetxt(directory, final_dataset, fmt="%s", delimiter=",")

#***************************************************************************************************************
# monoarticular no load 
skip_col = ['-','-','-']
#--------------------------------------------------------------------------------------------------------------
# columns
mean_joint_col = ['mean hip','mean knee','mean patellofemoral']
std_joint_col = ['std hip','std knee','std patellofemoral']
joint_col = np.concatenate((mean_joint_col,std_joint_col),axis=0)
#--------------------------------------------------------------------------------------------------------------
# rows
gait_phase_row = [item for item in['loading response','mid stance','terminal stance','pre swing',\
                  'initial swing','mid swing','terminal swing'] for i in range(len(mono_noload_indices)*3)]
force_row = ['Fx','Fy','Fz']*len(mono_noload_indices)*7
config_row =  [item for item in mono_noload_indices.tolist() for i in range(3)]*7
rows = np.row_stack((gait_phase_row,config_row))
rows = np.row_stack((rows,force_row))
rows = np.column_stack((skip_col,rows))
#--------------------------------------------------------------------------------------------------------------
# data
mean_mono_noload_max_jrf_change = np.array(list(mean_pareto_mono_noload_max_jrf_change_dict.items()))[:,1]
mean_mono_noload_max_jrf_change = mean_mono_noload_max_jrf_change.reshape((3,len(mono_noload_indices)*7*3),order='C')
std_mono_noload_max_jrf_change = np.array(list(std_pareto_mono_noload_max_jrf_change_dict.items()))[:,1]
std_mono_noload_max_jrf_change = std_mono_noload_max_jrf_change.reshape((3,len(mono_noload_indices)*7*3),order='C')
#--------------------------------------------------------------------------------------------------------------
# save
mono_noload_max_jrf_change = np.row_stack((mean_mono_noload_max_jrf_change,std_mono_noload_max_jrf_change))
mono_noload_dataset = np.column_stack((joint_col,mono_noload_max_jrf_change))
final_dataset = np.row_stack((rows,mono_noload_dataset))
directory = r'.\Data\Pareto\Monoarticular_Noload_JRF_Quantitatives.csv'
np.savetxt(directory, final_dataset, fmt="%s", delimiter=",")
#***************************************************************************************************************
# biarticular loaded
skip_col = ['-','-','-']
#--------------------------------------------------------------------------------------------------------------
# columns
mean_joint_col = ['mean hip','mean knee','mean patellofemoral']
std_joint_col = ['std hip','std knee','std patellofemoral']
joint_col = np.concatenate((mean_joint_col,std_joint_col),axis=0)
#--------------------------------------------------------------------------------------------------------------
# rows
gait_phase_row = [item for item in['loading response','mid stance','terminal stance','pre swing',\
                  'initial swing','mid swing','terminal swing'] for i in range(len(bi_loaded_indices)*3)]
force_row = ['Fx','Fy','Fz']*len(bi_loaded_indices)*7
config_row =  [item for item in bi_loaded_indices.tolist() for i in range(3)]*7
rows = np.row_stack((gait_phase_row,config_row))
rows = np.row_stack((rows,force_row))
rows = np.column_stack((skip_col,rows))
#--------------------------------------------------------------------------------------------------------------
# data
mean_bi_loaded_max_jrf_change = np.array(list(mean_pareto_bi_loaded_max_jrf_change_dict.items()))[:,1]
mean_bi_loaded_max_jrf_change = mean_bi_loaded_max_jrf_change.reshape((3,len(bi_loaded_indices)*7*3),order='C')
std_bi_loaded_max_jrf_change = np.array(list(std_pareto_bi_loaded_max_jrf_change_dict.items()))[:,1]
std_bi_loaded_max_jrf_change = std_bi_loaded_max_jrf_change.reshape((3,len(bi_loaded_indices)*7*3),order='C')
#--------------------------------------------------------------------------------------------------------------
# save
bi_loaded_max_jrf_change = np.row_stack((mean_bi_loaded_max_jrf_change,std_bi_loaded_max_jrf_change))
bi_loaded_dataset = np.column_stack((joint_col,bi_loaded_max_jrf_change))
final_dataset = np.row_stack((rows,bi_loaded_dataset))
directory = r'.\Data\Pareto\Biarticular_Loaded_JRF_Quantitatives.csv'
np.savetxt(directory, final_dataset, fmt="%s", delimiter=",")

#***************************************************************************************************************
# monoarticular loaded
skip_col = ['-','-','-']
#--------------------------------------------------------------------------------------------------------------
# columns
mean_joint_col = ['mean hip','mean knee','mean patellofemoral']
std_joint_col = ['std hip','std knee','std patellofemoral']
joint_col = np.concatenate((mean_joint_col,std_joint_col),axis=0)
#--------------------------------------------------------------------------------------------------------------
# rows
gait_phase_row = [item for item in['loading response','mid stance','terminal stance','pre swing',\
                  'initial swing','mid swing','terminal swing'] for i in range(len(mono_loaded_indices)*3)]
force_row = ['Fx','Fy','Fz']*len(mono_loaded_indices)*7
config_row =  [item for item in mono_loaded_indices.tolist() for i in range(3)]*7
rows = np.row_stack((gait_phase_row,config_row))
rows = np.row_stack((rows,force_row))
rows = np.column_stack((skip_col,rows))
#--------------------------------------------------------------------------------------------------------------
# data
mean_mono_loaded_max_jrf_change = np.array(list(mean_pareto_mono_loaded_max_jrf_change_dict.items()))[:,1]
mean_mono_loaded_max_jrf_change = mean_mono_loaded_max_jrf_change.reshape((3,len(mono_loaded_indices)*7*3),order='C')
std_mono_loaded_max_jrf_change = np.array(list(std_pareto_mono_loaded_max_jrf_change_dict.items()))[:,1]
std_mono_loaded_max_jrf_change = std_mono_loaded_max_jrf_change.reshape((3,len(mono_loaded_indices)*7*3),order='C')
#--------------------------------------------------------------------------------------------------------------
# save
mono_loaded_max_jrf_change = np.row_stack((mean_mono_loaded_max_jrf_change,std_mono_loaded_max_jrf_change))
mono_loaded_dataset = np.column_stack((joint_col,mono_loaded_max_jrf_change))
final_dataset = np.row_stack((rows,mono_loaded_dataset))
directory = r'.\Data\Pareto\Monoarticular_Loaded_JRF_Quantitatives.csv'
np.savetxt(directory, final_dataset, fmt="%s", delimiter=",")
