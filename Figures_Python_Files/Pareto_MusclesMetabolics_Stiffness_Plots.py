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
rra_dataset = utils.csv2numpy('./Data/RRA/rra_final_data.csv') 
unassist_dataset = utils.csv2numpy('./Data/Unassist/unassist_final_data.csv') 
# pareto muscles metabolic rate dataset
directory = './Data/Pareto/*_metabolic_rate.csv'
files = enumerate(glob.iglob(directory), 1)
assisted_muscles_metabolic_dataset = {pathlib.PurePath(f[1]).stem: np.loadtxt(f[1], delimiter=',') for f in files}
# unassist muscles metabolic rate dataset
directory = './Data/Unassist/*_metabolic_rate.csv'
files = enumerate(glob.iglob(directory), 1)
unassisted_muscles_metabolic_dataset = {pathlib.PurePath(f[1]).stem: np.loadtxt(f[1], delimiter=',') for f in files}
# gls
gl_noload = {'noload_subject{}_trial{}'.format(i,j): utils.construct_gl_mass_side(subjectno=i,trialno=j,loadcond='noload') for i in subjects for j in trials_num}
gl_loaded = {'loaded_subject{}_trial{}'.format(i,j): utils.construct_gl_mass_side(subjectno=i,trialno=j,loadcond='loaded') for i in subjects for j in trials_num}
#####################################################################################
# Processing Data
# toe-off
noload_mean_toe_off,_,loaded_mean_toe_off,_ = utils.toe_off_avg_std(gl_noload,gl_loaded)
# muscles metabolic rate
pareto_mean_noload_bi_muscles,pareto_std_noload_bi_muscles = utils.pareto_muscles_metabolic_reduction(assisted_muscles_metabolic_dataset['biarticular_pareto_noload_muscles_metabolic_rate'],unassisted_muscles_metabolic_dataset['noload_muscles_metabolic_rate'])
pareto_mean_loaded_bi_muscles,pareto_std_loaded_bi_muscles = utils.pareto_muscles_metabolic_reduction(assisted_muscles_metabolic_dataset['biarticular_pareto_load_muscles_metabolic_rate'],unassisted_muscles_metabolic_dataset['loaded_muscles_metabolic_rate'])
pareto_mean_noload_mono_muscles,pareto_std_noload_mono_muscles = utils.pareto_muscles_metabolic_reduction(assisted_muscles_metabolic_dataset['monoarticular_pareto_noload_muscles_metabolic_rate'],unassisted_muscles_metabolic_dataset['noload_muscles_metabolic_rate'])
pareto_mean_loaded_mono_muscles,pareto_std_loaded_mono_muscles = utils.pareto_muscles_metabolic_reduction(assisted_muscles_metabolic_dataset['monoarticular_pareto_load_muscles_metabolic_rate'],unassisted_muscles_metabolic_dataset['loaded_muscles_metabolic_rate'])
#####################################################################################
# Paretofront data
# mean & std muscles metabolics cost reduction percents
# noload biarticular
bi_noload_indices = np.array([25,24,23,22,21,19,18,17,13,12,11,1])
paretofront_mean_noload_bi_muscles = utils.manual_filter_muscles_metabolicrate(pareto_mean_noload_bi_muscles,bi_noload_indices)
paretofront_std_noload_bi_muscles = utils.manual_filter_muscles_metabolicrate(pareto_std_noload_bi_muscles,bi_noload_indices)

# noload monoarticular
mono_noload_indices = np.array([25,20,15,14,13,8,7,6,2,1])
paretofront_mean_noload_mono_muscles = utils.manual_filter_muscles_metabolicrate(pareto_mean_noload_mono_muscles,mono_noload_indices)
paretofront_std_noload_mono_muscles = utils.manual_filter_muscles_metabolicrate(pareto_std_noload_mono_muscles,mono_noload_indices)

# loaded biarticular
bi_loaded_indices = np.array([25,24,23,22,21,17,16,13,12,11,6,1])
paretofront_mean_loaded_bi_muscles = utils.manual_filter_muscles_metabolicrate(pareto_mean_loaded_bi_muscles,bi_loaded_indices)
paretofront_std_loaded_bi_muscles = utils.manual_filter_muscles_metabolicrate(pareto_std_loaded_bi_muscles,bi_loaded_indices)

# loaded monoarticular
mono_loaded_indices = np.array([25,20,15,10,5,4,3,2,1])
paretofront_mean_loaded_mono_muscles = utils.manual_filter_muscles_metabolicrate(pareto_mean_loaded_mono_muscles,mono_loaded_indices)
paretofront_std_loaded_mono_muscles = utils.manual_filter_muscles_metabolicrate(pareto_std_loaded_mono_muscles,mono_loaded_indices)
#####################################################################################
# muscles metabolic cost reduction percent plot
# noload biarticular
fig = plt.figure(num='Biarticular Noload Muscles Metabolic Rate',figsize=[25,25])
plot_dic = {'mean_data':paretofront_mean_noload_bi_muscles,'std_data':paretofront_std_noload_bi_muscles,
            'label':'biarticular, noload','weights':bi_noload_indices,'color':mycolors['french rose']}
utils.paretofront_plot_muscles_metabolics(plot_dic)
fig.tight_layout(h_pad=-1, w_pad=-1.5)
fig.subplots_adjust(top=0.98, bottom=0.075, left=0.100, right=0.975,hspace=0.55,wspace=0.15)
plt.show()
fig.savefig('./Figures/Paretofront/Analyses_Pareto/Pareto_Biarticular_Noload_Muscles_MR.pdf',orientation='landscape',bbox_inches='tight')

# noload monoarticular
fig = plt.figure(num='Monoarticular Noload Muscles Metabolic Rate',figsize=[25,25])
plot_dic = {'mean_data':paretofront_mean_noload_mono_muscles,'std_data':paretofront_std_noload_mono_muscles,
            'label':'monoarticular, noload','weights':mono_noload_indices,'color':mycolors['olympic blue']}
utils.paretofront_plot_muscles_metabolics(plot_dic)
fig.tight_layout(h_pad=-1, w_pad=-1.5)
fig.subplots_adjust(top=0.98, bottom=0.075, left=0.100, right=0.975,hspace=0.55,wspace=0.15)
plt.show()
fig.savefig('./Figures/Paretofront/Analyses_Pareto/Pareto_Monoarticular_Noload_Muscles_MR.pdf',orientation='landscape',bbox_inches='tight')

# loaded biarticular
fig = plt.figure(num='Biarticular Loaded Muscles Metabolic Rate',figsize=[25,25])
plot_dic = {'mean_data':paretofront_mean_loaded_bi_muscles,'std_data':paretofront_std_loaded_bi_muscles,
            'label':'biarticular, loaded','weights':bi_loaded_indices,'color':mycolors['crimson red']}
utils.paretofront_plot_muscles_metabolics(plot_dic)
fig.tight_layout(h_pad=-1, w_pad=-1.5)
fig.subplots_adjust(top=0.98, bottom=0.075, left=0.100, right=0.975,hspace=0.55,wspace=0.15)
plt.show()
fig.savefig('./Figures/Paretofront/Analyses_Pareto/Pareto_Biarticular_Loaded_Muscles_MR.pdf',orientation='landscape',bbox_inches='tight')

# loaded monoarticular
fig = plt.figure(num='Monoarticular Loaded Muscles Metabolic Rate',figsize=[25,25])
plot_dic = {'mean_data':paretofront_mean_loaded_mono_muscles,'std_data':paretofront_std_loaded_mono_muscles,
            'label':'monoarticular, loaded','weights':mono_loaded_indices,'color':mycolors['dark purple']}
utils.paretofront_plot_muscles_metabolics(plot_dic)
fig.tight_layout(h_pad=-1, w_pad=-1.5)
fig.subplots_adjust(top=0.98, bottom=0.075, left=0.100, right=0.975,hspace=0.55,wspace=0.15)
plt.show()
fig.savefig('./Figures/Paretofront/Analyses_Pareto/Pareto_Monoarticular_Loaded_Muscles_MR.pdf',orientation='landscape',bbox_inches='tight')


# loaded monoarticular vs biarticular
fig = plt.figure(num='Loaded Muscles Metabolic Rate Comparison',figsize=[25,25])
plot_dic = {'mean_data':paretofront_mean_loaded_mono_muscles,'std_data':paretofront_std_loaded_mono_muscles,
            'label':'monoarticular, loaded','weights':mono_loaded_indices,'color':mycolors['dark purple']}
utils.paretofront_plot_muscles_metabolics(plot_dic)
plot_dic = {'mean_data':paretofront_mean_loaded_bi_muscles,'std_data':paretofront_std_loaded_bi_muscles,
            'label':'biarticular, loaded','weights':bi_loaded_indices,'color':mycolors['crimson red']}
utils.paretofront_plot_muscles_metabolics(plot_dic)
fig.tight_layout(h_pad=-1, w_pad=-1.5)
fig.subplots_adjust(top=0.98, bottom=0.075, left=0.100, right=0.975,hspace=0.55,wspace=0.15)
plt.show()
fig.savefig('./Figures/Paretofront/Analyses_Pareto/Pareto_Loaded_MonoVsBi_Muscles_MR.pdf',orientation='landscape',bbox_inches='tight')


# noload monoarticular vs biarticular
fig = plt.figure(num='Noload Muscles Metabolic Rate Comparison',figsize=[25,25])
plot_dic = {'mean_data':paretofront_mean_noload_mono_muscles,'std_data':paretofront_std_noload_mono_muscles,
            'label':'monoarticular, loaded','weights':mono_noload_indices,'color':mycolors['olympic blue']}
utils.paretofront_plot_muscles_metabolics(plot_dic)
plot_dic = {'mean_data':paretofront_mean_noload_bi_muscles,'std_data':paretofront_std_noload_bi_muscles,
            'label':'biarticular, loaded','weights':bi_noload_indices,'color':mycolors['french rose']}
utils.paretofront_plot_muscles_metabolics(plot_dic)
fig.tight_layout(h_pad=-1, w_pad=-1.5)
fig.subplots_adjust(top=0.98, bottom=0.075, left=0.100, right=0.975,hspace=0.55,wspace=0.15)
plt.show()
fig.savefig('./Figures/Paretofront/Analyses_Pareto/Pareto_Noload_MonoVsBi_Muscles_MR.pdf',orientation='landscape',bbox_inches='tight')


# monoarticular loaded vs noload
fig = plt.figure(num='Loaded Muscles Metabolic Rate Comparison',figsize=[25,25])
plot_dic = {'mean_data':paretofront_mean_noload_mono_muscles,'std_data':paretofront_std_noload_mono_muscles,
            'label':'monoarticular, noload','weights':mono_noload_indices,'color':mycolors['olympic blue']}
utils.paretofront_plot_muscles_metabolics(plot_dic)
plot_dic = {'mean_data':paretofront_mean_loaded_mono_muscles,'std_data':paretofront_std_loaded_mono_muscles,
            'label':'monoarticular, loaded','weights':mono_loaded_indices,'color':mycolors['imperial red']}
utils.paretofront_plot_muscles_metabolics(plot_dic)
fig.tight_layout(h_pad=-1, w_pad=-1.5)
fig.subplots_adjust(top=0.98, bottom=0.075, left=0.100, right=0.975,hspace=0.55,wspace=0.15)
plt.show()
fig.savefig('./Figures/Paretofront/Analyses_Pareto/Pareto_Mono_LoadedVsNoload_Muscles_MR.pdf',orientation='landscape',bbox_inches='tight')

# biarticular loaded vs noload
fig = plt.figure(num='Loaded Muscles Metabolic Rate Comparison',figsize=[25,25])
plot_dic = {'mean_data':paretofront_mean_noload_bi_muscles,'std_data':paretofront_std_noload_bi_muscles,
            'label':'biarticular, noload','weights':bi_noload_indices,'color':mycolors['olympic blue']}
utils.paretofront_plot_muscles_metabolics(plot_dic)
plot_dic = {'mean_data':paretofront_mean_loaded_bi_muscles,'std_data':paretofront_std_loaded_bi_muscles,
            'label':'biarticular, loaded','weights':bi_loaded_indices,'color':mycolors['imperial red']}
utils.paretofront_plot_muscles_metabolics(plot_dic)
fig.tight_layout(h_pad=-1, w_pad=-1.5)
fig.subplots_adjust(top=0.98, bottom=0.075, left=0.100, right=0.975,hspace=0.55,wspace=0.15)
plt.show()
fig.savefig('./Figures/Paretofront/Analyses_Pareto/Pareto_Bi_LoadedVsNoload_Muscles_MR.pdf',orientation='landscape',bbox_inches='tight')
