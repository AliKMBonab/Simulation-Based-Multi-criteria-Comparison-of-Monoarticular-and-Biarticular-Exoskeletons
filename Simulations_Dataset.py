import collections
import copy
import os
import re
import csv
import enum
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
import Functions as fcns
#####################################################################################
#####################################################################################
"""Section 01:
                As the first step of processing data, we need to extract rra simulations
                data and confirm data modification
"""
joints  = ['hip','knee','hip','knee','hip','knee','hip','knee']
suffixes = ['torque','torque','power','power','speed','speed','kinematics','kinematics']
loads = ['noload','loaded']
#***************************
print('Starting to extract files.\n\n')
print('Section 01:\t extracting RRA related files: Torque, Power, Speed.\n')
#***************************
if os.path.exists('./Data/RRA/noload_hipjoint_torque.csv') == True and os.path.exists('./Data/RRA/loaded_hipjoint_torque.csv') == True:
    print('rra files already exist.\n\n')
else:
    print('rra files do not exist. extracting the file.\n')
    for load_type in loads:
        out=fcns.rra_data_subjects(loadcond=load_type)
        for i in range(len(out)):
            np.savetxt('./Data/RRA/{}_{}joint_{}.csv'.format(load_type,joints[i],suffixes[i]), out[i], fmt='%s', delimiter=',')
print('\n\n rra files have been extracted.')

#####################################################################################
#####################################################################################
"""Section 02:
                As the first step of processing data, we need to extract unassised subjects
                and ideal exoskeletons simulations data and confirm modeling and simulations
"""
loads = ['noload','loaded']
middle = ['metabolics','ninemuscles','hip','knee','metabolics_processed']
suffixes = ['energy','activation','musclesmoment','musclesmoment','energy']
#***************************
print('Section 02:\t extracting UnAssist Subjects related files: Metabolic Energy, Muscles Activation, Hip and Knee Muscles Moment.\n')
#***************************
if os.path.exists('./Data/Unassist/noload_ninemuscles_activation.csv') == True and os.path.exists('./Data/Unassist/loaded_ninemuscles_activation.csv') == True:
    print('unassist files already exist.\n\n')
else:
    print('unassist files do not exist. extracting the file.\n')
    for load_type in loads:
        out = fcns.unassist_idealdevice_data_subjects(configuration='UnAssist',loadcond=load_type)
        for i in range(len(out)):
            np.savetxt('./Data/Unassist/{}_{}_{}.csv'.format(load_type,middle[i],suffixes[i]), out[i], fmt='%s', delimiter=',')
#####################################################################################
#####################################################################################
"""Section 03:
                As the first step of processing data, we need to extract unassised subjects
                and ideal exoskeletons simulations data and confirm modeling and simulations
"""
loads = ['noload','loaded']
configs = ['Monoarticular/Ideal','Biarticular/Ideal']
config_names = ['monoarticular','biarticular']
middle =['hipactuator','kneeactuator','hipactuator','kneeactuator',\
         'hipactuator','kneeactuator','hipactuator','kneeactuator',\
         'hipactuator','kneeactuator','metabolics','ninemuscles',\
         'hip','knee','processed_hipactuator','processed_kneeactuator',\
         'processed_hipactuator','processed_kneeactuator','processed_metabolics']
suffixes = ['torque','torque','power','power','speed','speed',\
            'energy','energy','regenrative_energy','regenrative_energy',\
            'energy','activation','musclesmoment','musclesmoment',\
            'energy','energy','regenrative_energy','regenrative_energy',\
            'energy']
#***************************
print('Section 03:\t extracting Ideal exoskeletons related files: Actuators Data, Muscles Activation, Muscles Moment, Metabolic Energy.\n')
#***************************
if  os.path.exists('./Data/Ideal/monoarticular_ideal_noload_hipactuator_torque.csv') == True and os.path.exists('./Data/Ideal/monoarticular_ideal_loaded_hipactuator_torque.csv') == True\
and os.path.exists('./Data/Ideal/biarticular_ideal_noload_hipactuator_torque.csv') == True and os.path.exists('./Data/Ideal/biarticular_ideal_loaded_hipactuator_torque.csv') == True:

    print('ideal biarticular/monoarticular loaded/noload files already exist.\n\n')
else:
    print('ideal biarticular/monoarticular loaded/noload files do not exist. extracting the file.\n')
    for load_type in loads:
        for i in range(2):
            out= fcns.unassist_idealdevice_data_subjects(configuration=configs[i],loadcond= load_type,regenergy=True)
            for k in range(len(out)):
                np.savetxt('./Data/Ideal/{}_ideal_{}_{}_{}.csv'.format(config_names[i],load_type,middle[k],suffixes[k]), out[k], fmt='%s', delimiter=',')
#####################################################################################
#####################################################################################
"""Section 04:
                This section has been dedicated to extracting pareto simulations data for
                loaded and noload walking. The main extracted data are actuators energy,
                metabolic energy, torque profile, power profile.
"""
loads = ['noload','load']
configs = ['Monoarticular','Biarticular']
config_names = ['monoarticular','biarticular']
middle =['hipactuator','kneeactuator','hipactuator','kneeactuator',\
         'hipactuator','kneeactuator','metabolics',\
         'processed_hipactuator','processed_kneeactuator',\
         'processed_metabolics','metabolics','hipregenrative','kneeregenrative',\
         'hipregenrative','kneeregenrative','unsimulated']
suffixes = ['torque','torque','power','power',\
            'energy','energy','energy',\
            'energy','energy','energy','power',\
            'energy','energy','frompower_energy','frompower_energy','unsimulated']
#***************************
print('Section 04:\t extracting Pareto exoskeletons related files: Actuators Data, Muscles Activation, Muscles Moment, Metabolic Energy.\n')
#***************************
if  os.path.exists('./Data/Pareto/monoarticular_pareto_noload_hipactuator_torque.csv') == True and os.path.exists('./Data/Pareto/monoarticular_pareto_load_hipactuator_torque.csv') == True\
and os.path.exists('./Data/Pareto/biarticular_pareto_noload_hipactuator_torque.csv') == True and os.path.exists('./Data/Pareto/biarticular_pareto_load_hipactuator_torque.csv') == True:
    print('pareto biarticular/monoarticular loaded/noload files already exist.\n\n')
else:
    print('pareto biarticular/monoarticular loaded/noload files do not exist. extracting the file.\n')
    for load_type in loads:
        for i in range(2):
            out= fcns.pareto_data_subjects(configuration=configs[i],loadcond= load_type)
            for k in range(len(out)):
                np.savetxt('./Data/Pareto/{}_pareto_{}_{}_{}.csv'.format(config_names[i],load_type,middle[k],suffixes[k]), out[k], fmt='%s', delimiter=',')

#####################################################################################
#####################################################################################
"""Section 05:
                specific weights dataset
"""
loads = ['noload','load']
configs = ['Monoarticular','Biarticular']
config_names = ['monoarticular','biarticular']
middle =['hipactuator','kneeactuator','hipactuator','kneeactuator',\
         'hipactuator','kneeactuator','hipactuator','kneeactuator',\
         'metabolics','ninemuscles','hipmuscles','kneemuscles',\
         'hipactuator','kneeactuator']
suffixes = ['torque','torque','power','power',\
            'speed','speed','energy','energy',\
            'energy','activation','moment','moment',\
            'regenerative_energy','regenerative_energy']
HWs = {'mono_load':[30,70],'bi_load':[30,50,40,40],'mono_noload':[30,40,50],'bi_noload':[40,50,30]}
KWs = {'mono_load':[30,50],'bi_load':[60,60,60,40],'mono_noload':[30,30,50],'bi_noload':[30,40,30]}
labeling = ['mono','bi']
#***************************
print('Section 04:\t extracting Specific Weights of Pareto exoskeletons related files: Actuators Data, Muscles Activation, Muscles Moment, Metabolic Energy.\n')
#***************************
y = input('Specific weights data extraction? (y,n):  ')
if  y.lower() == 'y':
    print('specific weights biarticular/monoarticular loaded/noload files are getting extracted the file.\n')
    for load_type in loads:
        for i in range(2):
            HW = HWs['{}_{}'.format(labeling[i],load_type)]
            KW = KWs['{}_{}'.format(labeling[i],load_type)]
            for j in range(len(HW)):
                out= fcns.specific_weight_data_subjects(configuration=configs[i],HipWeight=HW[j],KneeWeight=KW[j],loadcond=load_type,regenergy=True)
                for k in range(len(out)):
                    np.savetxt('./Data/Specific_Weights/{}_hip{}knee{}_{}_{}_{}.csv'.format(config_names[i],HW[j],KW[j],load_type,middle[k],suffixes[k]), out[k], fmt='%s', delimiter=',')