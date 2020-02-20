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
import pandas as pd
from tabulate import tabulate
from numpy import nanmean, nanstd
from perimysium import postprocessing as pp
from perimysium import dataman

#################################################################
# Essential functions
#****************************************************************
def construct_gl_mass_trial(subjectno,trialno,loadcond='noload'):
    """This function has been designed to construct gl from the dataset. It also returns subject mass
       and trial number to be used on other functions"""
    import Subjects_Dataset as sd
    if loadcond == 'noload':
        data = sd.loaded_dataset["subject{}_noload_trial{}".replace(subjectno,trialno)]
    elif loadcond == 'loaded':
        data = sd.loaded_dataset["subject{}_loaded_trial{}".replace(subjectno,trialno)]
    else:
        raise Exception("load condition is wrong.")
    mass = data["mass"]
    trial_num = data["trial"]
    gl = dataman.GaitLandmarks( primary_leg = data["primary_legs"],
                                cycle_start = data["subjects_cycle_start_time"],
                                cycle_end   = data["subjects_cycle_end_time"],
                                left_strike = data["footstrike_left_leg"],
                                left_toeoff = data["toeoff_time_left_leg"],
                                right_strike= data["footstrike_right_leg"],
                                right_toeoff= data["toeoff_time_right_leg"])
    return gl, mass, trial_num 
#################################################################
# Metabolic Energy Reduction/ Muscles Moment calculation/ Metabolic energy calculations in pareto curve
#****************************************************************
def musclemoment_calc(data,gl,side):
    muscles_name = ['glut_med1_r','glut_med2_r','glut_med3_r','glut_min1_r','glut_min2_r','glut_min3_r','semimem_r',\
                    'semiten_r','bifemlh_r','bifemsh_r','sar_r','add_long_r','add_brev_r','add_mag1_r','add_mag2_r',\
                    'add_mag3_r','tfl_r','pect_r','grac_r','glut_max1_r','glut_max2_r','glut_max3_r','iliacus_r',\
                    'psoas_r','quad_fem_r','gem_r','peri_r','rect_fem_r','vas_med_r','vas_int_r','vas_lat_r','med_gas_r',\
                    'lat_gas_r','soleus_r','tib_post_r','flex_dig_r','flex_hal_r','tib_ant_r','per_brev_r','per_long_r',\
                    'per_tert_r','ext_dig_r','ext_hal_r']
    gait_cycle = np.linspace(0,100,1000)
    time = data['time']
    musclemoment = np.zeros([data.shape[0]])
    for i in range(len(muscles_name)):
        musclemoment += data[muscles_name[i]]
    gpc, musclemoment = pp.data_by_pgc(time,musclemoment,gl,side)
    musclemoment = np.interp(gait_cycle,gpc,musclemoment)
    return musclemoment
def metabolic_power_energy(SubjectNo,TrialNo,subject_mass,Directory,gl):
    """This function has been developed, seperated from the 'pareto_data_extraction' function, 
       to calculate the metabolic power and energy for cases in which we do not have pareto simulations.
       This will be mostly used for simulation of Unassisted subjects.

    ###########################################################################################
    * This function has been generalized for a subject to be utilized in another function.
    * Energy calculations have been normalized with subjects mass.
    """
    gait_cycle = np.linspace(0,100,1000)
    # Directory of each .sto files
    metabolic_power_data_dir = '../subject{}/noloaded/Subject{}_NoLoaded_Dataset/{}/loadedwalking_subject{}_noload_free_trial{}_cmc_Actuation_force.sto'\
                                .format(SubjectNo,SubjectNo,Directory,SubjectNo,TrialNo)
    # Sto to Numpy
    metabolic_power_data = dataman.storage2numpy(metabolic_power_data_dir)
    time = metabolic_power_data['time']
    gait_cycle = np.linspace(time[0],time[-1],1000)
    basal = metabolic_power_data['metabolic_power_BASAL']
    total = metabolic_power_data['metabolic_power_TOTAL']
    main_metabolics = basal + total
    metabolic_cost = np.interp(gait_cycle,time,main_metabolics)
    metabolic_energy = pp.avg_over_gait_cycle(metabolic_power_data['time'], main_metabolics,cycle_duration=gl.cycle_end-gl.cycle_start, cycle_start=gl.cycle_start)
    MetabolicEnergy_Data = metabolic_energy/subject_mass
    return metabolic_cost
def metabolic_energy_reduction(data,unassist_data):
    reduction = np.zeros(len(data))
    for i in range(len(data)):
        reduction[i] = (((unassist_data[i]-data[i])*100)/unassist_data[i])
    return reduction
#################################################################
# Adding mass metabolic cost change/Adding mass Metabolic cost (loaded subjects metabolic cost)
#****************************************************************
def adding_mass_metabolic_change(m_waist,m_thigh,m_shank,I_thigh,I_shank,unassisted_metabolic,I_leg=2.52):
    """ This function has been written according to R.C. Browning et al. paper which
        is calculating the metabolic cost CHANGE during the walking.
    """
    # Waist
    mass_metabolic_waist = 0.045*m_waist
    # Thigh
    mass_metabolic_thigh = 0.075*m_thigh
    I_thigh_ratio = (I_leg + I_thigh)/I_leg
    inertia_metabolic_thigh = ((-0.74 + (1.81*I_thigh_ratio))*unassisted_metabolic)-unassisted_metabolic
    delta_metabolic_thigh = mass_metabolic_thigh + inertia_metabolic_thigh
    # Shank
    mass_metabolic_shank = 0.076*m_shank
    I_shank_ratio = (I_leg + I_shank)/I_leg
    inertia_metabolic_shank = ((0.63749 + (0.40916*I_shank_ratio))*unassisted_metabolic)-unassisted_metabolic
    delta_metabolic_shank = mass_metabolic_shank + inertia_metabolic_shank
    # Total
    delta_metabolic_total = delta_metabolic_shank + delta_metabolic_thigh + mass_metabolic_waist
    
    return  mass_metabolic_waist,delta_metabolic_thigh,delta_metabolic_shank,delta_metabolic_total
def adding_mass_metabolic(m_waist,m_thigh,m_shank,I_thigh,I_shank,I_leg=2.52):
    """ This function has been written according to R.C. Browning et al. paper which
        is calculating the metabolic cost during the walking. 
    """
    # Waist
    mass_metabolic_waist = 2.36+0.045*m_waist
    # Thigh
    mass_metabolic_thigh = 2.38+0.075*m_thigh
    I_thigh_ratio = (I_leg + I_thigh)/I_leg
    inertia_metabolic_thigh = (-0.74 + (1.81*I_thigh_ratio))
    # Shank
    mass_metabolic_shank = 2.34+0.076*m_shank
    I_shank_ratio = (I_leg + I_shank)/I_leg
    inertia_metabolic_shank = (0.63749 + (0.40916*I_shank_ratio))
    return  mass_metabolic_waist,mass_metabolic_thigh,mass_metabolic_shank,inertia_metabolic_thigh,inertia_metabolic_shank
def metabolic_energy_mass_added_pareto(configuration,unassisted_metabolic,m_waist,m_thigh,m_shank,motor_max_torque=2,motor_inertia=0.000506,thigh_com=0.23,shank_com=0.18,leg_inertia=2.52,calc_metabolic_cost=True):
    """This function calculates the following data in a performed pareto simulations:
    - waist metabolic change
    - waist metabolic (optional)
    - thigh metabolic change
    - thigh metabolic (optional)
    - shank metabolic change
    - shank metabolic (optional)
    - the change of inertia in thigh in different maximum required torque (optional)
    - the change of inertia in shank in different maximum required torque (optional)
    #=======================================================================================
    - default values for actuator have been selected from Maxon Motor EC90 250W
    - default values for center of masses have been selected according to the desgin of exoskeletons
    - leg inertia were selected according to the inertia reported by reference paper
    """
    # initialization
    Hip_weights = [70,60,50,40,30]    # Hip Weight may need to be changed according to pareto simulations
    Knee_weights = [70,60,50,40,30]   # Knee Weight may need to be changed according to pareto simulations
    Metabolic_Change_Hip = np.zeros(len(Hip_weights)*len(Knee_weights))
    Metabolic_Change_Thigh = np.zeros(len(Hip_weights)*len(Knee_weights))
    Metabolic_Change_Shank = np.zeros(len(Hip_weights)*len(Knee_weights))
    Total_AddMass_MetabolicChange =  np.zeros(len(Hip_weights)*len(Knee_weights))
    Waist_Metabolic = np.zeros(len(Hip_weights)*len(Knee_weights))
    Thigh_Metabolic = np.zeros(len(Hip_weights)*len(Knee_weights))
    Shank_Metabolic = np.zeros(len(Hip_weights)*len(Knee_weights))
    Inertia_Thigh_Metabolic = np.zeros(len(Hip_weights)*len(Knee_weights))
    Inertia_Shank_Metabolic = np.zeros(len(Hip_weights)*len(Knee_weights))
    Inertia_Thigh = np.zeros(len(Hip_weights)*len(Knee_weights))
    Inertia_Shank = np.zeros(len(Hip_weights)*len(Knee_weights))
    # Due to higher weight of monoarticular in thigh we need to modify the thigh center of mass
    if configuration == 'monoarticular':
        thigh_com = 0.23 + 0.07
    # the masse added two sides
    m_waist = 2*m_waist
    m_thigh = 2*m_thigh
    m_shank = 2*m_shank
    # The metabolic cost change due to adding mass calculation
    c = 0
    for i in range(len(Hip_weights)):
        for j in range(len(Knee_weights)):
            # Gear Ratio max_needed_torque/motor_max_torque
            Hip_ratio = Hip_weights[i]/motor_max_torque
            Knee_ratio = Knee_weights[j]/motor_max_torque
            # I = motor_inertia*(ratio^2) + segment_mass*(segment_com^2)
            I_thigh = motor_inertia*(Hip_ratio**2)+ ((thigh_com**2)*m_thigh)
            I_shank =motor_inertia*(Knee_ratio**2) + ((shank_com**2)*m_shank)
            # loaded leg to unloaded leg inertia ratio
            Inertia_Shank[c] = (I_shank + leg_inertia)/leg_inertia
            Inertia_Thigh[c] = (I_thigh + leg_inertia)/leg_inertia
            # Metabolic change calculation in another function
            metabolic_waist,metabolic_thigh,metabolic_shank,AddMass_MetabolicChange \
                = adding_mass_metabolic_change(m_waist,m_thigh,m_shank,I_thigh,I_shank,unassisted_metabolic=unassisted_metabolic)
            # Storing the data into numpy arrays
            Metabolic_Change_Hip[c] = metabolic_waist
            Metabolic_Change_Thigh[c] = metabolic_thigh
            Metabolic_Change_Shank[c] = metabolic_shank
            Total_AddMass_MetabolicChange[c] =  AddMass_MetabolicChange
            # Metabolic cost calculation in another function
            if calc_metabolic_cost == True :
                mass_metabolic_waist,mass_metabolic_thigh,mass_metabolic_shank,\
                inertia_metabolic_thigh,inertia_metabolic_shank = adding_mass_metabolic(m_waist,m_thigh,m_shank,I_thigh,I_shank)
                Waist_Metabolic[c] = mass_metabolic_waist
                Thigh_Metabolic[c] = mass_metabolic_thigh
                Shank_Metabolic[c] = mass_metabolic_shank
                Inertia_Thigh_Metabolic[c] = inertia_metabolic_thigh
                Inertia_Shank_Metabolic[c] = inertia_metabolic_shank
            c+=1
    if calc_metabolic_cost == True :
        return Metabolic_Change_Hip,Metabolic_Change_Thigh,Metabolic_Change_Shank,Total_AddMass_MetabolicChange,\
               Waist_Metabolic,Thigh_Metabolic,Shank_Metabolic,Inertia_Thigh_Metabolic,Inertia_Shank_Metabolic,Inertia_Thigh,Inertia_Shank
    else:
        return Metabolic_Change_Hip,Metabolic_Change_Thigh,Metabolic_Change_Shank,AddMass_MetabolicChange,Inertia_Shank_Metabolic,Inertia_Thigh,Inertia_Shank
#####################################################################################
# Functions related to pareto data
#****************************************************************
def pareto_data_extraction(SubjectNo,TrialNo,subject_mass,Configuration,gl,calculatenergy=True):
    """This function is designed to get the configuration and optimal force that has been used to perform
    simulations and reporting most of the needed data. This function calculates the following data:
    
    - Torque profiles of actuators
    - Power Profiles of actuators
    - Speed profiles of actuators
    - Instantenous metabolic power of simulated subjects
    - The energy consumption of assistive actuators(optional)
    - Metabolic energy consumption(optional)

    ###########################################################################################
    * This function has been generalized for a subject to be utilized in another function.
    * Energy calculations have been normalized with subjects mass.
    * Some parts of this function can be changed to be used in another simulations.
    """
    # Configurations of Pareto Simulations
    optimal_force = 1000
    hip_list = [70/1000,60/1000,50/1000,40/1000,30/1000]
    knee_list = [70/1000,60/1000,50/1000,40/1000,30/1000]
    # Initialization
    gait_cycle = np.linspace(0,100,1000)
    KneeActuatorEnergy_Data = np.zeros(len(hip_list)*len(knee_list))
    HipActuatorEnergy_Data = np.zeros(len(hip_list)*len(knee_list))
    MetabolicEnergy_Data = np.zeros(len(hip_list)*len(knee_list))
    HipActuator_Torque_Data = np.zeros([1000,len(hip_list)*len(knee_list)])
    KneeActuator_Torque_Data = np.zeros([1000,len(hip_list)*len(knee_list)])
    HipActuator_Power_Data = np.zeros([1000,len(hip_list)*len(knee_list)])
    KneeActuator_Power_Data = np.zeros([1000,len(hip_list)*len(knee_list)])
    MetabolicCost_Data = np.zeros([1000,len(hip_list)*len(knee_list)])
    HipMuscleMoment_Data = np.zeros([1000,len(hip_list)*len(knee_list)])
    KneeMuscleMoment_Data = np.zeros([1000,len(hip_list)*len(knee_list)])
    c = 0
    # Following part extracts the data for a subject
    for hip_max_control in hip_list:
        for knee_max_control in knee_list:
            # Directory of each .sto files
            actuator_torque_data_dir= '../subject{}/noloaded/Subject{}_NoLoaded_Dataset/{}/H{}K{}/loadedwalking_subject{}_noload_free_trial{}_cmc_Actuation_force.sto'\
            .format(SubjectNo,SubjectNo,Configuration,int(hip_max_control*optimal_force),int(knee_max_control*optimal_force),SubjectNo,TrialNo)
            actuator_power_data_dir= '../subject{}/noloaded/Subject{}_NoLoaded_Dataset/{}/H{}K{}/loadedwalking_subject{}_noload_free_trial{}_cmc_Actuation_power.sto'\
            .format(SubjectNo,SubjectNo,Configuration,int(hip_max_control*optimal_force),int(knee_max_control*optimal_force),SubjectNo,TrialNo)
            metabolic_power_data_dir = '../subject{}/noloaded/Subject{}_NoLoaded_Dataset/{}/H{}K{}/loadedwalking_subject{}_noload_free_trial{}_cmc_ProbeReporter_probes.sto'\
            .format(SubjectNo,SubjectNo,Configuration,int(hip_max_control*optimal_force),int(knee_max_control*optimal_force),SubjectNo,TrialNo)
            # Sto to Numpy
            actuator_torque_data = dataman.storage2numpy(actuator_torque_data_dir)
            actuator_power_data = dataman.storage2numpy(actuator_power_data_dir)
            metabolic_power_data = dataman.storage2numpy(metabolic_power_data_dir)
            time = actuator_power_data['time']
            # time and gl class and reading specific columns of data
            knee_actuator_torque = actuator_torque_data['Knee_Right_Actuator']
            hip_actuator_torque = actuator_torque_data['Hip_Right_Actuator']
            knee_actuator_power = actuator_power_data['Knee_Right_Actuator']
            hip_actuator_power = actuator_power_data['Hip_Right_Actuator']
            knee_actuator_torque_l = actuator_torque_data['Knee_Left_Actuator']
            hip_actuator_torque_l = actuator_torque_data['Hip_Left_Actuator']
            knee_actuator_power_l = actuator_power_data['Knee_Left_Actuator']
            hip_actuator_power_l = actuator_power_data['Hip_Left_Actuator']
            # metabolic cost
            total_metabolic_power = metabolic_power_data['metabolic_power_TOTAL'] + metabolic_power_data['metabolic_power_BASAL']
            # Calculate Metabolic energy and the energy consumption of actuators
            if calculatenergy == True:
                hip_actuator_energy = pp.avg_over_gait_cycle(time, np.abs(hip_actuator_power),cycle_duration=gl.cycle_end-gl.cycle_start, cycle_start=gl.cycle_start)
                knee_actuator_energy = pp.avg_over_gait_cycle(time, np.abs(knee_actuator_power),cycle_duration=gl.cycle_end-gl.cycle_start, cycle_start=gl.cycle_start)
                metabolic_energy = pp.avg_over_gait_cycle(metabolic_power_data['time'], total_metabolic_power,cycle_duration=gl.cycle_end-gl.cycle_start, cycle_start=gl.cycle_start)
                HipActuatorEnergy_Data[c] = hip_actuator_energy/subject_mass
                KneeActuatorEnergy_Data[c] = knee_actuator_energy/subject_mass
                MetabolicEnergy_Data[c] = metabolic_energy/subject_mass
            # Shifting raw data to a gait cylce normarlized to pgc
            gpc, knee_actuator_torque = pp.data_by_pgc(time,knee_actuator_torque,gl,side='right')
            gpc, hip_actuator_torque = pp.data_by_pgc(time,hip_actuator_torque,gl,side='right')
            gpc, knee_actuator_power = pp.data_by_pgc(time,knee_actuator_power,gl,side='right')
            gpc, hip_actuator_power = pp.data_by_pgc(time,hip_actuator_power,gl,side='right')
            gpc_l, knee_actuator_torque_l = pp.data_by_pgc(time,knee_actuator_torque_l,gl,side='left')
            gpc_l, hip_actuator_torque_l = pp.data_by_pgc(time,hip_actuator_torque_l,gl,side='left')
            gpc_l, knee_actuator_power_l = pp.data_by_pgc(time,knee_actuator_power_l,gl,side='left')
            gpc_l, hip_actuator_power_l = pp.data_by_pgc(time,hip_actuator_power_l,gl,side='left')
            gpc_metabolic, total_metabolic_power = pp.data_by_pgc(metabolic_power_data['time'],total_metabolic_power,gl,side='right')
            # Interpreting the shifted and normalized data to a single pgc
            knee_actuator_torque = np.interp(gait_cycle,gpc,knee_actuator_torque)
            hip_actuator_torque = np.interp(gait_cycle,gpc,hip_actuator_torque)
            knee_actuator_power = np.interp(gait_cycle,gpc,knee_actuator_power)
            hip_actuator_power = np.interp(gait_cycle,gpc,hip_actuator_power)
            knee_actuator_torque_l = np.interp(gait_cycle,gpc_l,knee_actuator_torque_l)
            hip_actuator_torque_l = np.interp(gait_cycle,gpc_l,hip_actuator_torque_l)
            knee_actuator_power_l = np.interp(gait_cycle,gpc_l,knee_actuator_power_l)
            hip_actuator_power_l = np.interp(gait_cycle,gpc_l,hip_actuator_power_l)
            total_metabolic_power = np.interp(gait_cycle,gpc_metabolic,total_metabolic_power)
            # Storing the processed data into specificed numpy ndarrays
            HipActuator_Torque_Data[:,c] = nanmean([hip_actuator_torque,hip_actuator_torque_l],axis=0)
            KneeActuator_Torque_Data[:,c] = nanmean([knee_actuator_torque,knee_actuator_torque_l],axis=0)
            HipActuator_Power_Data[:,c] = nanmean([hip_actuator_power,hip_actuator_power_l],axis=0)
            KneeActuator_Power_Data[:,c] = nanmean([knee_actuator_power,knee_actuator_power_l],axis=0)
            MetabolicCost_Data[:,c] = total_metabolic_power
            c+=1
    return HipActuator_Torque_Data,KneeActuator_Torque_Data,HipActuator_Power_Data,KneeActuator_Power_Data,\
               MetabolicCost_Data,HipActuatorEnergy_Data,KneeActuatorEnergy_Data,MetabolicEnergy_Data
def pareto_data_subjects(configuration):
    """
    This function has been developed to extract pareto data (written in pareto_data_extraction function)
    for all the simulated subjects and configurations.

    - Torque profiles of actuators for all the subject
    - Power Profiles of actuators for all the subject
    - Speed profiles of actuators for all the subject
    - Instantenous metabolic power of simulated subjects for all the subject
    - The energy consumption of assistive actuators for all the subject
    - Metabolic energy consumption for all the subject

    ###########################################################################################
    * This function has been generalized for a subject to be utilized in another function.
    * Energy calculations have been normalized with subjects mass.
    * Some parts of this function can be changed to be used in another simulations.
    """
    # Configuration
    hip_list = [70/1000,60/1000,50/1000,40/1000,30/1000]
    knee_list = [70/1000,60/1000,50/1000,40/1000,30/1000]
    subjects = ['05','07','09','10','11','12','14']
    trails_num = ['01','02','03']
    # initialization
    KneeActuatorEnergy_Data = np.zeros(len(subjects)*len(trails_num)*len(hip_list)*len(knee_list))
    HipActuatorEnergy_Data = np.zeros(len(subjects)*len(trails_num)*len(hip_list)*len(knee_list))
    MetabolicEnergy_Data = np.zeros(len(subjects)*len(trails_num)*len(hip_list)*len(knee_list))
    HipActuator_Torque_Data = np.zeros([1000,len(subjects)*len(trails_num)*len(hip_list)*len(knee_list)])
    KneeActuator_Torque_Data = np.zeros([1000,len(subjects)*len(trails_num)*len(hip_list)*len(knee_list)])
    HipActuator_Power_Data = np.zeros([1000,len(subjects)*len(trails_num)*len(hip_list)*len(knee_list)])
    KneeActuator_Power_Data = np.zeros([1000,len(subjects)*len(trails_num)*len(hip_list)*len(knee_list)])
    MetabolicCost_Data = np.zeros([1000,len(subjects)*len(trails_num)*len(hip_list)*len(knee_list)])
    HipMuscleMoment_Data = np.zeros([1000,len(subjects)*len(trails_num)*len(hip_list)*len(knee_list)])
    KneeMuscleMoment_Data = np.zeros([1000,len(subjects)*len(trails_num)*len(hip_list)*len(knee_list)])
    c = 0
    for i in subjects:
        for j in trails_num:
            gl,subject_mass,trail = construct_gl_mass_trial(subjectno=i,trailno=j)
            hip_torque,knee_torque,hip_power,knee_power,metabolics,hip_energy,\
            knee_energy,metabolics_energy = pareto_data_extraction(SubjectNo=i,TrialNo=trail,subject_mass=subject_mass,Configuration=configuration,gl=gl)
            HipActuator_Torque_Data[:,c:c+len(hip_list)*len(knee_list)]  = hip_torque
            KneeActuator_Torque_Data[:,c:c+len(hip_list)*len(knee_list)] = knee_torque
            HipActuator_Power_Data[:,c:c+len(hip_list)*len(knee_list)]   = hip_power
            KneeActuator_Power_Data[:,c:c+len(hip_list)*len(knee_list)]  = knee_power
            MetabolicCost_Data[:,c:c+len(hip_list)*len(knee_list)]       = metabolics
            HipActuatorEnergy_Data[c:c+len(hip_list)*len(knee_list)]     = hip_energy
            KneeActuatorEnergy_Data[c:c+len(hip_list)*len(knee_list)]    = knee_energy
            MetabolicEnergy_Data[c:c+len(hip_list)*len(knee_list)]       = metabolics_energy
            c+=len(hip_list)*len(knee_list)+1

    return HipActuator_Torque_Data,KneeActuator_Torque_Data,HipActuator_Power_Data,KneeActuator_Power_Data,\
               MetabolicCost_Data,HipActuatorEnergy_Data,KneeActuatorEnergy_Data,MetabolicEnergy_Data
            
#####################################################################################
#####################################################################################
mono_m_waist = 3
mono_m_thigh = 3
mono_m_shank = 0.9
Mono_Metabolic_Change_Hip,Mono_Metabolic_Change_Thigh,Mono_Metabolic_Change_Shank,Mono_AddMass_MetabolicChange,\
Mono_Hip_Metabolic,Mono_Thigh_Metabolic,Mono_Shank_Metabolic,Mono_Inertia_Thigh_Metabolic,Mono_Inertia_Shank_Metabolic,\
Mono_Inertia_Thigh,Mono_Inertia_Shank= metabolic_energy_mass_added_pareto('monoarticular',unassisted_meabolic_energy,mono_m_waist,mono_m_thigh,mono_m_shank)
bi_m_waist = 6
bi_m_thigh = 1
bi_m_shank = 0.9
Bi_Metabolic_Change_Hip,Bi_Metabolic_Change_Thigh,Bi_Metabolic_Change_Shank,Bi_AddMass_MetabolicChange,\
Bi_Hip_Metabolic,Bi_Thigh_Metabolic,Bi_Shank_Metabolic,Bi_Inertia_Thigh_Metabolic,Bi_Inertia_Shank_Metabolic,\
Bi_Inertia_Thigh,Bi_Inertia_Shank= metabolic_energy_mass_added_pareto('biarticular',unassisted_meabolic_energy,bi_m_waist,bi_m_thigh,bi_m_shank)
Hip_weights = [70,60,50,40,30]
Knee_weights = [70,60,50,40,30]
#####################################################################################
#####################################################################################
labels = []
Hip_weights = [70,60,50,40,30]
Knee_weights = [70,60,50,40,30]
for i in range(len(Hip_weights)):
    for j in range(len(Knee_weights)):
        labels.append('H{}K{}'.format(Hip_weights[i],Knee_weights[j]))
#####################################################################################
#####################################################################################
general_path = 'F:/HMI/Exoskeleton/OpenSim/LoadedWalking_Test/Main_Test/Results/'
