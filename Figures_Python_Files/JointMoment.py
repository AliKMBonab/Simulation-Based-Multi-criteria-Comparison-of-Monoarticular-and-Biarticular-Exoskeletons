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
# Reading CSV files into a dictionary
directory = 'D:\Ali.K.M.Bonab\Walking_Mass_Inertia_Effect\Data\Data\RRA\*_torque.csv'
files = enumerate(glob.iglob(directory), 1)
dataset = {pathlib.PurePath(f[1]).stem: np.loadtxt(f[1], delimiter=',') for f in files}
#####################################################################################
# Processing Data
gl_noload = {'noload_subject{}_trial{}'.format(i,j): utils.construct_gl_mass_side(subjectno=i,trialno=j,loadcond='noload') for i in subjects for j in trials_num}
gl_loaded = {'loaded_subject{}_trial{}'.format(i,j): utils.construct_gl_mass_side(subjectno=i,trialno=j,loadcond='loaded') for i in subjects for j in trials_num}
noload_mean_toe_off,_,loaded_mean_toe_off,_ = utils.toe_off_avg_std(gl_noload,gl_loaded)
normal_loaded_hipjoint_moment = utils.normalize_direction_data(dataset['loaded_hipjoint_torque'],gl_noload,direction=True)
normal_noload_hipjoint_moment = utils.normalize_direction_data(dataset['noload_hipjoint_torque'],gl_noload,direction=True)
mean_norm_loaded_hipjoint_moment,std_norm_loaded_hipjoint_moment = utils.mean_std_over_subjects(normal_loaded_hipjoint_moment)
mean_norm_noload_hipjoint_moment,std_norm_noload_hipjoint_moment = utils.mean_std_over_subjects(normal_noload_hipjoint_moment)


    

fig, ax = plt.subplots()
plot_dic = {'pgc':gait_cycle,'avg':utils.smooth(mean_norm_loaded_hipjoint_moment,3),'label':'Loaded',
            'std':utils.smooth(std_norm_loaded_hipjoint_moment,3),'avg_toeoff':loaded_mean_toe_off}
utils.plot_shaded_avg(plot_dic=plot_dic,color='k')
plot_dic = {'pgc':gait_cycle,'avg':utils.smooth(mean_norm_noload_hipjoint_moment,3),'label':'Noload',
            'std':utils.smooth(std_norm_noload_hipjoint_moment,3),'avg_toeoff':noload_mean_toe_off}
utils.plot_shaded_avg(plot_dic=plot_dic,toeoff_color='xkcd:shamrock green',color='xkcd:tree green')
plt.legend(loc='best')
plt.ylim((-1,2))
plt.xlabel('gait cycle (%)')
plt.ylabel('hip joint moment (N-m/kg)')
utils.no_top_right(ax)
plt.show()
print('test')


