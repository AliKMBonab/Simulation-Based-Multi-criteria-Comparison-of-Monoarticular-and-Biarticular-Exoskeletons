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

#################################################################
# Essential functions
#****************************************************************
def construct_gl_mass_trial(subjectno,trialno,loadcond='noload'):
    """This function has been designed to construct gl from the dataset. It also returns subject mass
       and trial number to be used on other functions"""
    import Subjects_Dataset as sd
    if loadcond == 'noload':
        data = sd.noload_dataset["subject{}_noload_trial{}".format(subjectno,trialno)]
    elif loadcond == 'loaded':
        data = sd.loaded_dataset["subject{}_loaded_trial{}".format(subjectno,trialno)]
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
def data_extraction(Subject_Dic):
    """This function has been developed to extract and modify data.
    """
    # Reading Required Data
    directory = Subject_Dic["Directory"]
    right_param = Subject_Dic["Right_Parameter"]
    left_param = Subject_Dic["Left_Parameter"]
    gl = Subject_Dic["gl"]
    gait_cycle = np.linspace(0,100,1000)
    numpy_data = dataman.storage2numpy(directory)
    time = numpy_data['time']
    right_data = numpy_data[right_param]
    left_data = numpy_data[left_param]
    if gl.primary_leg == 'right':
        gpc_r, shifted_right_data = pp.data_by_pgc(time,right_data,gl,side='right')
        gpc_l, shifted_left_data  = pp.data_by_pgc(time,left_data,gl,side='left')
    elif gl.primary_leg == 'left':
        gpc_r, shifted_right_data = pp.data_by_pgc(time,right_data,gl,side='left')
        gpc_l, shifted_left_data  = pp.data_by_pgc(time,left_data,gl,side='right')
    else:
        raise Exception('Primary leg (right/left) is not correct!!')
    interperted_right_data = np.interp(gait_cycle,gpc_r,shifted_right_data)
    interperted_left_data  = np.interp(gait_cycle,gpc_l,shifted_left_data)
    final_data = nanmean([interperted_right_data,interperted_left_data],axis=0)
    return final_data
def actuators_normal_energy_calc(Subject_Dic,isabs=True,regen=False):
    """This function developed to calculate the consumed energy by actuators"""
    directory = Subject_Dic["Directory"]
    right_param = Subject_Dic["Right_Parameter"]
    left_param = Subject_Dic["Left_Parameter"]
    gl = Subject_Dic["gl"]
    subject_mass = Subject_Dic["Subject_Mass"]
    numpy_data = dataman.storage2numpy(directory)
    time = numpy_data['time']
    right_data = numpy_data[right_param]
    left_data = numpy_data[left_param]
    if isabs == True:
        energy = pp.avg_over_gait_cycle(time, np.abs(right_data),cycle_duration=gl.cycle_end-gl.cycle_start, cycle_start=gl.cycle_start)\
               + pp.avg_over_gait_cycle(time, np.abs(left_data),cycle_duration=gl.cycle_end-gl.cycle_start, cycle_start=gl.cycle_start)
    else:
        energy = pp.avg_over_gait_cycle(time, right_data,cycle_duration=gl.cycle_end-gl.cycle_start, cycle_start=gl.cycle_start)\
               + pp.avg_over_gait_cycle(time,left_data,cycle_duration=gl.cycle_end-gl.cycle_start, cycle_start=gl.cycle_start) 
    normalized_energy = energy/subject_mass
    if regen == True:
        regen_energy = pp.avg_over_gait_cycle(time, np.abs(np.min(0,right_data)),cycle_duration=gl.cycle_end-gl.cycle_start, cycle_start=gl.cycle_start)\
               + pp.avg_over_gait_cycle(time, np.abs(np.min(0,left_data)),cycle_duration=gl.cycle_end-gl.cycle_start, cycle_start=gl.cycle_start)
        normalized_regen_energy = regen_energy/subject_mass
    if regen == True:
        return normalized_energy
    else:
        return normalized_energy, normalized_regen_energy
################################################################# 
# Metabolic Energy Reduction/ Muscles Moment calculation/ Metabolic energy calculations in pareto curve
#****************************************************************
def musclemoment_calc(Subject_Dic):
    """This function returns muscles moment by getting their numpy file for left and right sides.
    """
    muscles_name = ['add_brev','add_long','add_mag3','add_mag4','add_mag2','add_mag1','bifemlh','bifemsh','ext_dig',\
                    'ext_hal','flex_dig','flex_hal','lat_gas','med_gas','glut_max1','glut_max2','glut_max3','glut_med1',\
                    'glut_med2','glut_med3','glut_min1','glut_min2','glut_min3','grac','iliacus','per_brev','per_long',\
                    'peri','psoas','rect_fem','sar','semimem','semiten','soleus','tfl','tib_ant','tib_post','vas_int',\
                    'vas_lat','vas_med']
    gait_cycle = np.linspace(0,100,1000)
    right_directory = Subject_Dic["Right_Directory"]
    left_directory = Subject_Dic["Left_Directory"]
    gl = Subject_Dic["gl"]
    data_r = dataman.storage2numpy(right_directory)
    data_l = dataman.storage2numpy(left_directory)
    time_r = data_r['time']
    time_l = data_l['time']
    musclemoment_r = np.zeros([data_r.shape[0]])
    musclemoment_l = np.zeros([data_l.shape[0]])
    for i in range(len(muscles_name)):
        musclemoment_r += data_r[muscles_name[i]+'_r']
        musclemoment_l += data_l[muscles_name[i]+'_l']
    gpc = np.linspace(0,100,1000)
    gpc_r,shifted_data_r = pp.data_by_pgc(time_r,musclemoment_r,gl,side='right')
    gpc_l,shifted_data_l = pp.data_by_pgc(time_l,musclemoment_l,gl,side='left')
    main_data_r = np.interp(gpc,gpc_r,shifted_data_r)
    main_data_l = np.interp(gpc,gpc_l,shifted_data_l)
    musclemoment = nanmean([main_data_r,main_data_l],axis=0)
    return musclemoment
def metabolic_normal_energy(Subject_Dic):
    """This function has been developed, seperated from the 'pareto_data_extraction' function, 
       to calculate the metabolic power and energy for cases in which we do not have pareto simulations.
       This will be mostly used for simulation of Unassisted subjects.

    ###########################################################################################
    * This function has been generalized for a subject to be utilized in another function.
    * Energy calculations have been normalized with subjects mass.
    """
    metabolic_power_data_dir = Subject_Dic["Directory"]
    gl = Subject_Dic["gl"]
    subject_mass = Subject_Dic["Subject_Mass"]
    gait_cycle = np.linspace(0,100,1000)
    # Sto to Numpy
    metabolic_power_data = dataman.storage2numpy(metabolic_power_data_dir)
    time = metabolic_power_data['time']
    gait_cycle = np.linspace(time[0],time[-1],1000)
    basal = metabolic_power_data['metabolic_power_BASAL']
    total = metabolic_power_data['metabolic_power_TOTAL']
    main_metabolics = basal + total
    metabolic_cost = np.interp(gait_cycle,time,main_metabolics)
    metabolic_energy = pp.avg_over_gait_cycle(metabolic_power_data['time'], main_metabolics,cycle_duration=gl.cycle_end-gl.cycle_start, cycle_start=gl.cycle_start)
    normalized_metabolic_energy = metabolic_energy/subject_mass
    return normalized_metabolic_energy
def metabolic_energy_reduction(data,unassist_data):
    reduction = np.zeros(len(data))
    for i in range(len(data)):
        reduction[i] = (((unassist_data[i]-data[i])*100)/unassist_data[i])
    return reduction
def group_muscles_activation(Subject_Dic,whichgroup='nine',loadcond='noload'):
    """This function returns the activation of the set of muscles that will be determined by user.
       The user can select "hip, knee, and nine" which are standing for set of hip and knee muscles and 
       set of nine important muscles in lower exterimity.
     """
    if whichgroup == 'hip':
    # The name of muscles contributing on hip flexion and extension
        muscles_name = ['add_brev','add_long','add_mag3','add_mag2','add_mag1','bifemlh',\
                            'glut_max1','glut_max2','glut_max3','glut_med1','glut_min1','glut_min3',\
                            'grac','iliacus','psoas','rect_fem','sar','semimem','semiten','tfl']
    elif whichgroup == 'knee':
        # The name of muscles contributing on knee flexion and extension
        muscles_name = ['bifemlh','bifemsh','ext_dig','lat_gas','med_gas','grac',\
                            'rect_fem','sar','semimem','semiten','vas_int','vas_lat','vas_med']
    elif whichgroup == 'nine':
        # The name of nine representitive muscles on lower extermity
        muscles_name = ['bifemsh','glut_max1','psoas','lat_gas','rect_fem','semimem','soleus','tib_ant','vas_lat']
    elif whichgroup == 'both':
         # The name of muscles contributing on knee and hip flexion and extension
        muscles_name = ['add_brev','add_long','add_mag3','add_mag2','add_mag1','bifemlh',\
                            'glut_max1','glut_max2','glut_max3','glut_med1','glut_min1','glut_min3',\
                            'grac','iliacus','psoas','rect_fem','sar','semimem','semiten','tfl','bifemlh',\
                            'bifemsh','ext_dig','lat_gas','med_gas','grac','rect_fem','sar','semimem',\
                            'semiten','vas_int','vas_lat','vas_med']
    else:
        raise Exception('group is not in the list')
    # Establishing directory/initialization/reading data
    Directory = Subject_Dic["Directory"]
    SubjectNo = Subject_Dic["SubjectNo"]
    TrialNo = '02'
    gl = Subject_Dic["gl"]
    data_dir = '../subject{}/{}/{}/loadedwalking_subject{}_{}_free_trial{}_cmc_states.sto'\
                .format(SubjectNo,Directory,SubjectNo,loadcond,TrialNo)
    muscles_activation = np.zeros([1000,len(muscles_name)])
    gait_cycle = np.linspace(0,100,1000)
    data = dataman.storage2numpy(data_dir)
    time = data['time']
    c = 0
    for muscle in muscles_name:
        muscle_r_activation = data[muscle+'_r'+'.activation']
        muscle_l_activation = data[muscle+'_l'+'.activation']
        gpc_r, shifted_muscle_r_activation = pp.data_by_pgc(time,muscle_r_activation,gl,side='right')
        gpc_l, shifted_muscle_l_activation = pp.data_by_pgc(time,muscle_l_activation,gl,side='left')
        muscle_r_activation = np.interp(gait_cycle,gpc_r,shifted_muscle_r_activation)
        muscle_l_activation = np.interp(gait_cycle,gpc_l,shifted_muscle_l_activation)
        muscles_activation[:,c]=nanmean([muscle_r_activation,muscle_l_activation],axis=0)
        c=+1
    return muscles_activation
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
def metabolic_energy_mass_added_pareto(unassisted_metabolic,InertialProp_Dic,calc_metabolic_cost=True):
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
    #=======================================================================================
    * Default: motor_max_torque=2, motor_inertia=0.000506, thigh_com=0.23, shank_com=0.18, leg_inertia=2.52
    """
    # initialization
    m_waist = InertialProp_Dic["m_waist"]
    m_thigh = InertialProp_Dic["m_thigh"]
    m_shank = InertialProp_Dic["m_shank"]
    motor_max_torque = InertialProp_Dic["motor_max_torque"]
    motor_inertia = InertialProp_Dic["motor_inertia"]
    thigh_com =  InertialProp_Dic["thigh_com"]
    shank_com =  InertialProp_Dic["shank_com"]
    leg_inertia = InertialProp_Dic["leg_inertia"]
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
def pareto_data_extraction(Subject_Dic,loadcond='noload',calculatenergy=True):
    """This function is designed to get the configuration and optimal force that has been used to perform
    simulations and reporting most of the needed data. This function calculates the following data:
    
    - Torque profiles of actuators
    - Power Profiles of actuators
    - The energy consumption of assistive actuators(optional)
    - Metabolic energy consumption(optional)

    ###########################################################################################
    * This function has been generalized for a subject to be utilized in another function.
    * Energy calculations have been normalized with subjects mass.
    * Some parts of this function can be changed to be used in another simulations.
    * Performed simulations have labeling issue (trial02 for all subjects), therefore trial has
      been modified without loosing generality of function.
    """
    # Configurations of Pareto Simulations
    Directory = Subject_Dic["Directory"]
    SubjectNo = Subject_Dic["SubjectNo"]
    TrialNo = Subject_Dic["TrialNo"]
    gl = Subject_Dic["gl"]
    subject_mass = Subject_Dic["Subject_Mass"]
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
    c = 0
    # Following part extracts the data for a subject
    for hip_max_control in hip_list:
        for knee_max_control in knee_list:
            # Directory of each .sto files
            actuator_torque_data_dir= '../subject{}/{}/H{}K{}/loadedwalking_subject{}_{}_free_trial{}_cmc_Actuation_force.sto'\
            .format(SubjectNo,Directory,int(hip_max_control*optimal_force),int(knee_max_control*optimal_force),SubjectNo,loadcond,TrialNo)
            actuator_power_data_dir= '../subject{}/{}/H{}K{}/loadedwalking_subject{}_{}_free_trial{}_cmc_Actuation_power.sto'\
            .format(SubjectNo,Directory,int(hip_max_control*optimal_force),int(knee_max_control*optimal_force),SubjectNo,loadcond,TrialNo)
            metabolic_power_data_dir = '../subject{}/{}/H{}K{}/loadedwalking_subject{}_{}_free_trial{}_cmc_ProbeReporter_probes.sto'\
            .format(SubjectNo,Directory,int(hip_max_control*optimal_force),int(knee_max_control*optimal_force),SubjectNo,loadcond,TrialNo)
            data_extraction_dic = {"Directory":actuator_torque_data_dir,
                                 "Right_Parameter":'Hip_Right_Actuator',
                                  "Left_Parameter":'Hip_Left_Actuator',
                                              "gl": gl}
            hip_actuator_torque = data_extraction(Subject_Dic=data_extraction_dic)
            data_extraction_dic = {"Directory":actuator_torque_data_dir,
                                 "Right_Parameter":'Knee_Right_Actuator',
                                  "Left_Parameter":'Knee_Left_Actuator',
                                              "gl": gl}
            knee_actuator_torque = data_extraction(Subject_Dic=data_extraction_dic)
            data_extraction_dic = {"Directory":actuator_power_data_dir,
                                 "Right_Parameter":'Hip_Right_Actuator',
                                  "Left_Parameter":'Hip_Left_Actuator',
                                              "gl": gl}
            hip_actuator_power = data_extraction(Subject_Dic=data_extraction_dic)
            data_extraction_dic = {"Directory":actuator_power_data_dir,
                                 "Right_Parameter":'Knee_Right_Actuator',
                                  "Left_Parameter":'Knee_Left_Actuator',
                                              "gl": gl}
            knee_actuator_power = data_extraction(Subject_Dic=data_extraction_dic)
            
            # Storing the processed data into specificed numpy ndarrays
            HipActuator_Torque_Data[:,c]  = hip_actuator_torque
            KneeActuator_Torque_Data[:,c] = knee_actuator_torque
            HipActuator_Power_Data[:,c]   = hip_actuator_power
            KneeActuator_Power_Data[:,c]  = knee_actuator_power
            # Energy calculations
            if calculatenergy == True:
                energy_dic = {"Directory":actuator_power_data_dir,
                        "Right_Parameter":'Hip_Right_Actuator',
                         "Left_Parameter":'Hip_Left_Actuator',
                                     "gl": gl,
                           "Subject_Mass": subject_mass}
                hip_actuator_energy = actuators_normal_energy_calc(Subject_Dic=energy_dic)
                energy_dic = {"Directory":actuator_power_data_dir,
                        "Right_Parameter":'Knee_Right_Actuator',
                         "Left_Parameter":'Knee_Left_Actuator',
                                     "gl": gl,
                           "Subject_Mass": subject_mass}
                knee_actuator_energy = actuators_normal_energy_calc(Subject_Dic=energy_dic)
                energy_dic = {"Directory":metabolic_power_data_dir,
                                     "gl": gl,
                           "Subject_Mass": subject_mass}
                metabolic_energy = metabolic_normal_energy(energy_dic)
                HipActuatorEnergy_Data[c]  = hip_actuator_energy
                KneeActuatorEnergy_Data[c] = knee_actuator_energy
                MetabolicEnergy_Data[c] = metabolic_energy
            c+=1
    return HipActuator_Torque_Data,KneeActuator_Torque_Data,HipActuator_Power_Data,KneeActuator_Power_Data,\
           HipActuatorEnergy_Data,KneeActuatorEnergy_Data,MetabolicEnergy_Data
def pareto_data_subjects(configuration,loadcond='noload'):
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
    * Performed simulations have labeling issue (trial02 for all subjects), therefore trial has
      been modified without loosing generality of function.

    ###########################################################################################
      The function returns: HipActuator_Torque_Data,KneeActuator_Torque_Data,HipActuator_Power_Data,
      KneeActuator_Power_Data,HipActuatorEnergy_Data,KneeActuatorEnergy_Data,MetabolicEnergy_Data
    """
    # Configuration
    hip_list = [70/1000,60/1000,50/1000,40/1000,30/1000]
    knee_list = [70/1000,60/1000,50/1000,40/1000,30/1000]
    subjects = ['05','07','09','10','11','12','14']
    trails_num = ['01']
    # initialization
    KneeActuatorEnergy_Data = np.zeros(len(subjects)*len(trails_num)*len(hip_list)*len(knee_list))
    HipActuatorEnergy_Data = np.zeros(len(subjects)*len(trails_num)*len(hip_list)*len(knee_list))
    MetabolicEnergy_Data = np.zeros(len(subjects)*len(trails_num)*len(hip_list)*len(knee_list))
    HipActuator_Torque_Data = np.zeros([1000,len(subjects)*len(trails_num)*len(hip_list)*len(knee_list)])
    KneeActuator_Torque_Data = np.zeros([1000,len(subjects)*len(trails_num)*len(hip_list)*len(knee_list)])
    HipActuator_Power_Data = np.zeros([1000,len(subjects)*len(trails_num)*len(hip_list)*len(knee_list)])
    KneeActuator_Power_Data = np.zeros([1000,len(subjects)*len(trails_num)*len(hip_list)*len(knee_list)])
    c = 0
    for i in subjects:
        for j in trails_num:
            # subject/trial/directory construction
            gl,subject_mass,trial = construct_gl_mass_trial(subjectno=i,trialno=j,loadcond=loadcond)
            if loadcond == 'noload':
                files_dir = 'noloaded/Subject{}_NoLoaded_Dataset/{}'.format(i,configuration)
            elif loadcond == 'loaded':
                files_dir = 'loaded/Subject{}_Loaded_Dataset/{}'.format(i,configuration)
            else:
                raise Exception('Invalid load condition!')
            Subject_Dictionary = {"Directory": files_dir,
                                  "SubjectNo": i,
                                    "TrialNo": '02',
                                         "gl": gl,
                               "Subject_Mass":subject_mass}
            
            hip_torque,knee_torque,hip_power,knee_power,hip_energy,\
            knee_energy,metabolics_energy = pareto_data_extraction(Subject_Dictionary,loadcond=loadcond,regen_energy=regenergy)
            # saving data into initialized variables
            HipActuator_Torque_Data[:,c:c+len(hip_list)*len(knee_list)]  = hip_torque
            KneeActuator_Torque_Data[:,c:c+len(hip_list)*len(knee_list)] = knee_torque
            HipActuator_Power_Data[:,c:c+len(hip_list)*len(knee_list)]   = hip_power
            KneeActuator_Power_Data[:,c:c+len(hip_list)*len(knee_list)]  = knee_power
            HipActuatorEnergy_Data[c:c+len(hip_list)*len(knee_list)]     = hip_energy
            KneeActuatorEnergy_Data[c:c+len(hip_list)*len(knee_list)]    = knee_energy
            MetabolicEnergy_Data[c:c+len(hip_list)*len(knee_list)]       = metabolics_energy
            c+=len(hip_list)*len(knee_list)
    return HipActuator_Torque_Data,KneeActuator_Torque_Data,HipActuator_Power_Data,KneeActuator_Power_Data,\
            HipActuatorEnergy_Data,KneeActuatorEnergy_Data,MetabolicEnergy_Data
#####################################################################################
# Functions related to data extraction which will be used for extracting RRA data, Unassisted subjects
# data and finally data for the specific weights. This data extraction functions include muscles moment,
# muscles activation, and other needed data.
#****************************************************************
def specific_weight_data_extraction(Subject_Dic,Hip_Weight,Knee_Weight,loadcond='noload',calculatenergy=True,regen_energy=False):
    """ This function is designed to get the configuration and optimal force that has been used to perform
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
    * Performed simulations have labeling issue (trial02 for all subjects), therefore trial has
      been modified without loosing generality of function.
    """
    # Initialization
    Directory = Subject_Dic["Directory"]
    SubjectNo = Subject_Dic["SubjectNo"]
    TrialNo = '02'
    gl = Subject_Dic["gl"]
    subject_mass = Subject_Dic["subject_mass"]
    gait_cycle = np.linspace(0,100,1000)
    # Directory of each .sto files
    actuator_torque_data_dir= '../subject{}/{}/H{}K{}/loadedwalking_subject{}_{}_free_trial{}_cmc_Actuation_force.sto'\
    .format(SubjectNo,Directory,Hip_Weight,Knee_Weight,SubjectNo,loadcond,TrialNo)
    actuator_power_data_dir= '../subject{}/{}/H{}K{}/loadedwalking_subject{}_{}_free_trial{}_cmc_Actuation_power.sto'\
    .format(SubjectNo,Directory,Hip_Weight,Knee_Weight,SubjectNo,loadcond,TrialNo)
    actuator_speed_data_dir= '../subject{}/{}/H{}K{}/loadedwalking_subject{}_{}_free_trial{}_cmc_Actuation_speed.sto'\
    .format(SubjectNo,Directory,Hip_Weight,Knee_Weight,SubjectNo,loadcond,TrialNo)
    metabolic_power_data_dir = '../subject{}/{}/H{}K{}/loadedwalking_subject{}_{}_free_trial{}_cmc_ProbeReporter_probes.sto'\
    .format(SubjectNo,Directory,Hip_Weight,Knee_Weight,SubjectNo,loadcond,TrialNo)
    data_extraction_dic = {"Directory":actuator_torque_data_dir,
                                 "Right_Parameter":'Hip_Right_Actuator',
                                  "Left_Parameter":'Hip_Left_Actuator',
                                              "gl": gl}
    hip_actuator_torque = data_extraction(Subject_Dic=data_extraction_dic)
    data_extraction_dic = {"Directory":actuator_torque_data_dir,
                                 "Right_Parameter":'Knee_Right_Actuator',
                                  "Left_Parameter":'Knee_Left_Actuator',
                                              "gl": gl}
    knee_actuator_torque = data_extraction(Subject_Dic=data_extraction_dic)
    data_extraction_dic = {"Directory":actuator_power_data_dir,
                                 "Right_Parameter":'Hip_Right_Actuator',
                                  "Left_Parameter":'Hip_Left_Actuator',
                                              "gl": gl}
    hip_actuator_power = data_extraction(Subject_Dic=data_extraction_dic)
    data_extraction_dic = {"Directory":actuator_power_data_dir,
                                 "Right_Parameter":'Knee_Right_Actuator',
                                  "Left_Parameter":'Knee_Left_Actuator',
                                              "gl": gl}
    knee_actuator_power = data_extraction(Subject_Dic=data_extraction_dic)
    data_extraction_dic = {"Directory":actuator_speed_data_dir,
                                 "Right_Parameter":'Hip_Right_Actuator',
                                  "Left_Parameter":'Hip_Left_Actuator',
                                              "gl": gl}
    hip_actuator_speed = data_extraction(Subject_Dic=data_extraction_dic)
    data_extraction_dic = {"Directory":actuator_speed_data_dir,
                                 "Right_Parameter":'Knee_Right_Actuator',
                                  "Left_Parameter":'Knee_Left_Actuator',
                                              "gl": gl}
    knee_actuator_speed = data_extraction(Subject_Dic=data_extraction_dic)
    if calculatenergy == True:
                energy_dic = {"Directory":actuator_power_data_dir,
                        "Right_Parameter":'Hip_Right_Actuator',
                         "Left_Parameter":'Hip_Left_Actuator',
                                     "gl": gl,
                           "Subject_Mass": subject_mass}
                if regen_energy == True:
                    hip_actuator_energy,hip_actuator_regen_energy = actuators_normal_energy_calc(Subject_Dic=energy_dic,regen=regen_energy)
                else:
                    hip_actuator_energy = actuators_normal_energy_calc(Subject_Dic=energy_dic,regen=regen_energy)
                energy_dic = {"Directory":actuator_power_data_dir,
                        "Right_Parameter":'Knee_Right_Actuator',
                         "Left_Parameter":'Knee_Left_Actuator',
                                     "gl": gl,
                           "Subject_Mass": subject_mass}
                if regen_energy == True:
                    knee_actuator_energy,hip_actuator_regen_energy = actuators_normal_energy_calc(Subject_Dic=energy_dic,regen=regen_energy)
                else:
                    knee_actuator_energy = actuators_normal_energy_calc(Subject_Dic=energy_dic,regen=regen_energy)
                energy_dic = {"Directory":metabolic_power_data_dir,
                                     "gl": gl,
                           "Subject_Mass": subject_mass}
                metabolic_energy = metabolic_normal_energy(energy_dic)
    if regen_energy == True:
        return hip_actuator_torque,knee_actuator_torque,hip_actuator_power,knee_actuator_power,\
            hip_actuator_speed,knee_actuator_speed,hip_actuator_energy,knee_actuator_energy,metabolic_energy,hip_actuator_regen_energy,knee_actuator_regen_energy
    else:
        return hip_actuator_torque,knee_actuator_torque,hip_actuator_power,knee_actuator_power,\
            hip_actuator_speed,knee_actuator_speed,hip_actuator_energy,knee_actuator_energy,metabolic_energy
def specific_weight_data_subjects(configuration,HipWeight,KneeWeight,loadcond='noload',musclesmoment=True,musclesactivation=True,regenergy=False):
    """This function generalize the specific_weight_data_extraction for all subjects and additionally it provides muscles activation
    and muscles generated moment.
    -Default setting for muscle activation is nine representitive muscles of the lower extermity. It can be changed to knee/hip/both.
     """
    subjects = ['05','07','09','10','11','12','14']
    trails_num = ['01']
    whichgroup='nine'
    musclesgroup = 9
    # initialization
    KneeActuatorEnergy_Data = np.zeros(len(subjects)*len(trails_num))
    HipActuatorEnergy_Data = np.zeros(len(subjects)*len(trails_num))
    Regen_KneeActuatorEnergy_Data = np.zeros(len(subjects)*len(trails_num))
    Regen_HipActuatorEnergy_Data = np.zeros(len(subjects)*len(trails_num))
    MetabolicEnergy_Data = np.zeros(len(subjects)*len(trails_num))
    HipActuator_Torque_Data = np.zeros([1000,len(subjects)*len(trails_num)])
    KneeActuator_Torque_Data = np.zeros([1000,len(subjects)*len(trails_num)])
    HipActuator_Power_Data = np.zeros([1000,len(subjects)*len(trails_num)])
    KneeActuator_Power_Data = np.zeros([1000,len(subjects)*len(trails_num)])
    HipActuator_Speed_Data = np.zeros([1000,len(subjects)*len(trails_num)])
    KneeActuator_Speed_Data = np.zeros([1000,len(subjects)*len(trails_num)])
    HipMuscleMoment_Data = np.zeros([1000,len(subjects)*len(trails_num)])
    KneeMuscleMoment_Data = np.zeros([1000,len(subjects)*len(trails_num)])
    MuscleActivation_Data = np.zeros([1000,len(subjects)*len(trails_num)*musclesgroup])
    c = 0
    c_m = 0
    for i in subjects:
        for j in trails_num:
            # subject/trial/directory construction
            gl,subject_mass,trial = construct_gl_mass_trial(subjectno=i,trailno=j,loadcond=loadcond)
            if loadcond == 'noload':
                files_dir = 'noloaded/Subject{}_NoLoaded_Dataset/{}'.format(i,configuration)
            else:
                files_dir = 'noloaded/Subject{}_Loaded_Dataset/{}'.format(i,configuration)
            Subject_Dictionary = {"Directory": files_dir,
                                  "SubjectNo": i,
                                    "TrialNo": trial,
                                         "gl": gl,
                               "subject_mass":subject_mass}
            if regenergy == False:
                hip_torque,knee_torque,hip_power,knee_power,hip_speed,\
                knee_speed,hip_energy,knee_energy,metabolics_energy = \
                specific_weight_data_extraction(Subject_Dictionary,Hip_Weight=HipWeight,Knee_weight=KneeWeight,loadcond=loadcond)
            else:
                hip_torque,knee_torque,hip_power,knee_power,hip_speed,\
                knee_speed,hip_energy,knee_energy,metabolics_energy,reg_hip_energy,reg_knee_energy = \
                specific_weight_data_extraction(Subject_Dictionary,Hip_Weight=HipWeight,Knee_weight=KneeWeight,loadcond=loadcond)
                Regen_HipActuatorEnergy_Data[c] = reg_hip_energy
                Regen_KneeActuatorEnergy_Data[c] = reg_knee_energy
            # saving data into initialized variables
            HipActuator_Torque_Data[:,c:c+1]  = hip_torque
            KneeActuator_Torque_Data[:,c:c+1] = knee_torque
            HipActuator_Power_Data[:,c:c+1]   = hip_power
            KneeActuator_Power_Data[:,c:c+1]  = knee_power
            HipActuator_Speed_Data[:,c:c+1]   = hip_speed
            KneeActuator_Speed_Data[:,c:c+1]  = knee_speed
            HipActuatorEnergy_Data[c]     = hip_energy
            KneeActuatorEnergy_Data[c]    = knee_energy
            MetabolicEnergy_Data[c]       = metabolics_energy
            if musclesactivation == True:
                muscles_activation = group_muscles_activation(Subject_Dictionary,whichgroup=whichgroup,loadcond=loadcond)
                MuscleActivation_Data[c_m:c_m+musclesgroup] = muscles_activation
                c_m=musclesgroup+1
            if musclesmoment == True:
                
                hip_r_musclesmoment_dir = '../subject{}/{}/H{}K{}/loadedwalking_subject{}_{}_free_trial{}_analyze_MuscleAnalysis_Moment_hip_flexion_r.sto'\
                                        .format(i,files_dir,HipWeight,KneeWeight,i,loadcond,'02')
                hip_l_musclesmoment_dir = '../subject{}/{}/H{}K{}/loadedwalking_subject{}_{}_free_trial{}_analyze_MuscleAnalysis_Moment_hip_flexion_l.sto'\
                                        .format(i,files_dir,HipWeight,KneeWeight,i,loadcond,'02')
                knee_r_musclesmoment_dir = '../subject{}/{}/H{}K{}/loadedwalking_subject{}_{}_free_trial{}_analyze_MuscleAnalysis_Moment_knee_angle_r.sto'\
                                        .format(i,files_dir,HipWeight,KneeWeight,i,loadcond,'02')
                knee_l_musclesmoment_dir = '../subject{}/{}/H{}K{}/loadedwalking_subject{}_{}_free_trial{}_analyze_MuscleAnalysis_Moment_knee_angle_l.sto'\
                                        .format(i,files_dir,HipWeight,KneeWeight,i,loadcond,'02')
                musclemoment_dic = {"Right_Directory":hip_r_musclesmoment_dir,"Left_Directory":hip_l_musclesmoment_dir,"gl":gl}
                hip_muscle_moment = musclemoment_calc(Subject_Dic=musclemoment_dic)
                musclemoment_dic = {"Right_Directory":knee_r_musclesmoment_dir,"Left_Directory":knee_l_musclesmoment_dir,"gl":gl}
                knee_muscle_moment = musclemoment_calc(Subject_Dic=musclemoment_dic)
                HipMuscleMoment_Data[:,c:c+1] = hip_muscle_moment
                KneeMuscleMoment_Data[:,c:c+1] = knee_muscle_moment
            c+=1
    return HipActuator_Torque_Data,KneeActuator_Torque_Data,HipActuator_Power_Data,KneeActuator_Power_Data,\
           HipActuator_Speed_Data,KneeActuator_Speed_Data,HipActuatorEnergy_Data,KneeActuatorEnergy_Data,\
           MetabolicEnergy_Data,MuscleActivation_Data,HipMuscleMoment_Data,KneeMuscleMoment_Data,Regen_HipActuatorEnergy_Data,Regen_KneeActuatorEnergy_Data
def rra_data_extraction(Subject_Dic,loadcond='noload'):
    """ This function is designed to get the configuration and optimal force that has been used to perform
        simulations and reporting most of the needed data. This function calculates the following data:
    
    - Torque profiles of actuators
    - Power Profiles of actuators
    - Speed profiles of actuators

    ###########################################################################################
    * This function has been generalized for a subject to be utilized in another function.
    * Energy calculations have been normalized with subjects mass.
    * Some parts of this function can be changed to be used in another simulations.
    """
    # Initialization
    Directory = Subject_Dic["Directory"]
    SubjectNo = Subject_Dic["SubjectNo"]
    TrialNo = Subject_Dic["TrialNo"]
    CycleNo = Subject_Dic["CycleNo"]
    gl = Subject_Dic["gl"]
    subject_mass = Subject_Dic["subject_mass"]
    gait_cycle = np.linspace(0,100,1000)
    # Directory of each .sto files
    actuator_torque_data_dir= '../subject{}/{}/Cycle{}/loadedwalking_subject{}_{}_free_trial{}_rratasks_Actuation_force.sto'\
    .format(SubjectNo,Directory,CycleNo,SubjectNo,loadcond,TrialNo)
    actuator_power_data_dir= '../subject{}/{}/Cycle{}/loadedwalking_subject{}_{}_free_trial{}_rratasks_Actuation_power.sto'\
    .format(SubjectNo,Directory,CycleNo,SubjectNo,loadcond,TrialNo)
    actuator_speed_data_dir= '../subject{}/{}/Cycle{}/loadedwalking_subject{}_{}_free_trial{}_rratasks_Actuation_speed.sto'\
    .format(SubjectNo,Directory,CycleNo,SubjectNo,loadcond,TrialNo)
    data_extraction_dic = {"Directory":actuator_torque_data_dir,
                                 "Right_Parameter":'hip_flexion_r',
                                  "Left_Parameter":'hip_flexion_l',
                                              "gl": gl}
    hip_actuator_torque = data_extraction(Subject_Dic=data_extraction_dic)
    data_extraction_dic = {"Directory":actuator_torque_data_dir,
                                 "Right_Parameter":'knee_angle_r',
                                  "Left_Parameter":'knee_angle_l',
                                              "gl": gl}
    knee_actuator_torque = data_extraction(Subject_Dic=data_extraction_dic)
    data_extraction_dic = {"Directory":actuator_power_data_dir,
                                 "Right_Parameter":'hip_flexion_r',
                                  "Left_Parameter":'hip_flexion_l',
                                              "gl": gl}
    hip_actuator_power = data_extraction(Subject_Dic=data_extraction_dic)
    data_extraction_dic = {"Directory":actuator_power_data_dir,
                                 "Right_Parameter":'Knee_Right_Actuator',
                                  "Left_Parameter":'Knee_Left_Actuator',
                                              "gl": gl}
    knee_actuator_power = data_extraction(Subject_Dic=data_extraction_dic)
    data_extraction_dic = {"Directory":actuator_speed_data_dir,
                                 "Right_Parameter":'hip_flexion_r',
                                  "Left_Parameter":'hip_flexion_l',
                                              "gl": gl}
    hip_actuator_speed = data_extraction(Subject_Dic=data_extraction_dic)
    data_extraction_dic = {"Directory":actuator_speed_data_dir,
                                 "Right_Parameter":'Knee_Right_Actuator',
                                  "Left_Parameter":'Knee_Left_Actuator',
                                              "gl": gl}
    knee_actuator_speed = data_extraction(Subject_Dic=data_extraction_dic)
    
    return hip_actuator_torque,knee_actuator_torque,hip_actuator_power,knee_actuator_power,\
           hip_actuator_speed,knee_actuator_speed,hip_actuator_energy,knee_actuator_energy,metabolic_energy
def rra_data_subjects(configuration,loadcond='noload'):
    """This function generalize the rra_data_extraction for all subjects.
    -Default setting for muscle activation is nine representitive muscles of the lower extermity. It can be changed to knee/hip/both.
     """
    subjects = ['05','07','09','10','11','12','14']
    trials_num = ['01']
    whichgroup='nine'
    musclesgroup = 9
    # initialization
    HipActuator_Torque_Data = np.zeros([1000,len(subjects)*len(trails_num)])
    KneeActuator_Torque_Data = np.zeros([1000,len(subjects)*len(trails_num)])
    HipActuator_Power_Data = np.zeros([1000,len(subjects)*len(trails_num)])
    KneeActuator_Power_Data = np.zeros([1000,len(subjects)*len(trails_num)])
    HipActuator_Speed_Data = np.zeros([1000,len(subjects)*len(trails_num)])
    KneeActuator_Speed_Data = np.zeros([1000,len(subjects)*len(trails_num)])
    c = 0
    for i in subjects:
        for j in trials_num:
            # subject/trial/directory construction
            gl,subject_mass,trial = construct_gl_mass_trial(subjectno=i,trailno=j,loadcond=loadcond)
            if loadcond == 'noload':
                files_dir = 'noloaded/RRA'
            else:
                files_dir = 'noloaded/RRA'
            Subject_Dictionary = {"Directory": files_dir,
                                  "SubjectNo": i,
                                    "TrialNo": trial,
                                    "CycleNo": trials_num,
                                         "gl": gl,
                               "subject_mass":subject_mass}
            hip_torque,knee_torque,hip_power,knee_power,hip_speed,knee_speed\
                 = rra_data_extraction(Subject_Dictionary,loadcond=loadcond)
            # saving data into initialized variables
            HipActuator_Torque_Data[:,c:c+1]  = hip_torque
            KneeActuator_Torque_Data[:,c:c+1] = knee_torque
            HipActuator_Power_Data[:,c:c+1]   = hip_power
            KneeActuator_Power_Data[:,c:c+1]  = knee_power
            HipActuator_Speed_Data[:,c:c+1]   = hip_speed
            KneeActuator_Speed_Data[:,c:c+1]  = knee_speed
            c+=1
    return HipActuator_Torque_Data,KneeActuator_Torque_Data,HipActuator_Power_Data,KneeActuator_Power_Data,\
           HipActuator_Speed_Data,KneeActuator_Speed_Data
def idealdevice_data_extraction(Subject_Dic,loadcond='noload',calculatenergy=True,regen_energy=False):
    """ This function is designed to get the configuration and optimal force that has been used to perform
        simulations and reporting most of the needed data. This function calculates the following data:
    
    - Torque profiles of actuators
    - Power Profiles of actuators
    - Speed profiles of actuators
    - The energy consumption of assistive actuators(optional)
    
    ###########################################################################################
    * This function has been generalized for a subject to be utilized in another function.
    * Energy calculations have been normalized with subjects mass.
    * Some parts of this function can be changed to be used in another simulations.
    """
    # Initialization
    Directory = Subject_Dic["Directory"]
    SubjectNo = Subject_Dic["SubjectNo"]
    TrialNo = Subject_Dic["TrialNo"]
    gl = Subject_Dic["gl"]
    subject_mass = Subject_Dic["subject_mass"]
    gait_cycle = np.linspace(0,100,1000)
    # Directory of each .sto files
    actuator_torque_data_dir= '../subject{}/{}/loadedwalking_subject{}_{}_free_trial{}_cmc_Actuation_force.sto'\
    .format(SubjectNo,Directory,SubjectNo,loadcond,TrialNo)
    actuator_power_data_dir= '../subject{}/{}/loadedwalking_subject{}_{}_free_trial{}_cmc_Actuation_power.sto'\
    .format(SubjectNo,Directory,Hip_Weight,Knee_Weight,SubjectNo,loadcond,TrialNo)
    actuator_speed_data_dir= '../subject{}/{}/H{}K{}/loadedwalking_subject{}_{}_free_trial{}_cmc_Actuation_speed.sto'\
    .format(SubjectNo,Directory,Hip_Weight,Knee_Weight,SubjectNo,loadcond,TrialNo)
    metabolic_power_data_dir = '../subject{}/{}/H{}K{}/loadedwalking_subject{}_{}_free_trial{}_cmc_ProbeReporter_probes.sto'\
    .format(SubjectNo,Directory,Hip_Weight,Knee_Weight,SubjectNo,loadcond,TrialNo)
    data_extraction_dic = {"Directory":actuator_torque_data_dir,
                                 "Right_Parameter":'Hip_Right_Actuator',
                                  "Left_Parameter":'Hip_Left_Actuator',
                                              "gl": gl}
    hip_actuator_torque = data_extraction(Subject_Dic=data_extraction_dic)
    data_extraction_dic = {"Directory":actuator_torque_data_dir,
                                 "Right_Parameter":'Knee_Right_Actuator',
                                  "Left_Parameter":'Knee_Left_Actuator',
                                              "gl": gl}
    knee_actuator_torque = data_extraction(Subject_Dic=data_extraction_dic)
    data_extraction_dic = {"Directory":actuator_power_data_dir,
                                 "Right_Parameter":'Hip_Right_Actuator',
                                  "Left_Parameter":'Hip_Left_Actuator',
                                              "gl": gl}
    hip_actuator_power = data_extraction(Subject_Dic=data_extraction_dic)
    data_extraction_dic = {"Directory":actuator_power_data_dir,
                                 "Right_Parameter":'Knee_Right_Actuator',
                                  "Left_Parameter":'Knee_Left_Actuator',
                                              "gl": gl}
    knee_actuator_power = data_extraction(Subject_Dic=data_extraction_dic)
    data_extraction_dic = {"Directory":actuator_speed_data_dir,
                                 "Right_Parameter":'Hip_Right_Actuator',
                                  "Left_Parameter":'Hip_Left_Actuator',
                                              "gl": gl}
    hip_actuator_speed = data_extraction(Subject_Dic=data_extraction_dic)
    data_extraction_dic = {"Directory":actuator_speed_data_dir,
                                 "Right_Parameter":'Knee_Right_Actuator',
                                  "Left_Parameter":'Knee_Left_Actuator',
                                              "gl": gl}
    knee_actuator_speed = data_extraction(Subject_Dic=data_extraction_dic)
    if calculatenergy == True:
                energy_dic = {"Directory":actuator_power_data_dir,
                        "Right_Parameter":'Hip_Right_Actuator',
                         "Left_Parameter":'Hip_Left_Actuator',
                                     "gl": gl,
                           "Subject_Mass": subject_mass}
                if regen_energy == False:
                    hip_actuator_energy = actuators_normal_energy_calc(Subject_Dic=energy_dic,regen=regen_energy)
                else:
                     hip_actuator_energy,hip_actuator_regen_energy = actuators_normal_energy_calc(Subject_Dic=energy_dic,regen=regen_energy)
                energy_dic = {"Directory":actuator_power_data_dir,
                        "Right_Parameter":'Knee_Right_Actuator',
                         "Left_Parameter":'Knee_Left_Actuator',
                                     "gl": gl,
                           "Subject_Mass": subject_mass}
                if regen_energy == False:
                    knee_actuator_energy = actuators_normal_energy_calc(Subject_Dic=energy_dic,regen=regen_energy)
                else:
                     knee_actuator_energy,knee_actuator_regen_energy = actuators_normal_energy_calc(Subject_Dic=energy_dic,regen=regen_energy)
    if regen_energy == False:
        return hip_actuator_torque,knee_actuator_torque,hip_actuator_power,knee_actuator_power,\
            hip_actuator_speed,knee_actuator_speed,hip_actuator_energy,knee_actuator_energy
    else:
        return hip_actuator_torque,knee_actuator_torque,hip_actuator_power,knee_actuator_power,\
            hip_actuator_speed,knee_actuator_speed,hip_actuator_energy,knee_actuator_energy,hip_actuator_regen_energy,knee_actuator_regen_energy
def unassist_idealdevice_data_subjects(configuration,loadcond='noload',metabolicrate=True,musclesmoment=True,musclesactivation=True,regenergy=False):
    """This function generalize the specific_weight_data_extraction for all subjects and additionally it provides muscles activation
    and muscles generated moment.
    -Default setting for muscle activation is nine representitive muscles of the lower extermity. It can be changed to knee/hip/both.
    -Configuration can be: -Biarticular/Ideal -Monoarticular/Ideal -UnAssist
     """
    config_list = ['Biarticular/Ideal','Monoarticular/Ideal','UnAssist']
    if configuration not in config_list:
        raise Exception('Configuration is not valid for this function.')

    subjects = ['05','07','09','10','11','12','14']
    trails_num = ['01','02','03']
    whichgroup='nine'
    musclesgroup = 9
    # initialization
    KneeActuatorEnergy_Data = np.zeros(len(subjects)*len(trails_num))
    HipActuatorEnergy_Data = np.zeros(len(subjects)*len(trails_num))
    Regen_KneeActuatorEnergy_Data = np.zeros(len(subjects)*len(trails_num))
    Regen_HipActuatorEnergy_Data = np.zeros(len(subjects)*len(trails_num))
    MetabolicEnergy_Data = np.zeros(len(subjects)*len(trails_num))
    HipActuator_Torque_Data = np.zeros([1000,len(subjects)*len(trails_num)])
    KneeActuator_Torque_Data = np.zeros([1000,len(subjects)*len(trails_num)])
    HipActuator_Power_Data = np.zeros([1000,len(subjects)*len(trails_num)])
    KneeActuator_Power_Data = np.zeros([1000,len(subjects)*len(trails_num)])
    HipActuator_Speed_Data = np.zeros([1000,len(subjects)*len(trails_num)])
    KneeActuator_Speed_Data = np.zeros([1000,len(subjects)*len(trails_num)])
    HipMuscleMoment_Data = np.zeros([1000,len(subjects)*len(trails_num)])
    KneeMuscleMoment_Data = np.zeros([1000,len(subjects)*len(trails_num)])
    MuscleActivation_Data = np.zeros([1000,len(subjects)*len(trails_num)*musclesgroup])
    c = 0
    c_m = 0
    for i in subjects:
        for j in trails_num:
            # subject/trial/directory construction
            gl,subject_mass,trial = construct_gl_mass_trial(subjectno=i,trailno=j,loadcond=loadcond)
            if loadcond == 'noload':
                files_dir = 'noloaded/Subject{}_NoLoaded_Dataset/{}'.format(i,configuration)
            else:
                files_dir = 'noloaded/Subject{}_Loaded_Dataset/{}'.format(i,configuration)
            Subject_Dictionary = {"Directory": files_dir,
                                  "SubjectNo": i,
                                    "TrialNo": trial,
                                         "gl": gl,
                               "subject_mass":subject_mass}
            if configuration !='UnAssist':
                if regenergy == False:
                    hip_torque,knee_torque,hip_power,knee_power,hip_speed,\
                    knee_speed,hip_energy,knee_energy = \
                    idealdevice_data_extraction(Subject_Dictionary,loadcond=loadcond)
                else:
                    hip_torque,knee_torque,hip_power,knee_power,hip_speed,\
                    knee_speed,hip_energy,knee_energy,hip_regen_energy,knee_regen_energy = \
                    idealdevice_data_extraction(Subject_Dictionary,loadcond=loadcond,regen_energy=regenergy)
                    Regen_HipActuatorEnergy_Data[c]     = hip_regen_energy
                    Regen_KneeActuatorEnergy_Data[c]    = knee_regen_energy
                # saving data into initialized variables
                HipActuator_Torque_Data[:,c:c+1]  = hip_torque
                KneeActuator_Torque_Data[:,c:c+1] = knee_torque
                HipActuator_Power_Data[:,c:c+1]   = hip_power
                KneeActuator_Power_Data[:,c:c+1]  = knee_power
                HipActuator_Speed_Data[:,c:c+1]   = hip_speed
                KneeActuator_Speed_Data[:,c:c+1]  = knee_speed
                HipActuatorEnergy_Data[c]     = hip_energy
                KneeActuatorEnergy_Data[c]    = knee_energy
            if musclesactivation == True:
                muscles_activation = group_muscles_activation(Subject_Dictionary,whichgroup=whichgroup,loadcond=loadcond)
                MuscleActivation_Data[c_m:c_m+musclesgroup] = muscles_activation
                c_m=musclesgroup+1
            if musclesmoment == True:
                hip_r_musclesmoment_dir = '../subject{}/{}/loadedwalking_subject{}_{}_free_trial{}_analyze_MuscleAnalysis_Moment_hip_flexion_r.sto'\
                                        .format(i,files_dir,i,loadcond,trial)
                hip_l_musclesmoment_dir = '../subject{}/{}/loadedwalking_subject{}_{}_free_trial{}_analyze_MuscleAnalysis_Moment_hip_flexion_l.sto'\
                                        .format(i,files_dir,i,loadcond,trial)
                knee_r_musclesmoment_dir = '../subject{}/{}/loadedwalking_subject{}_{}_free_trial{}_analyze_MuscleAnalysis_Moment_knee_angle_r.sto'\
                                        .format(i,files_dir,i,loadcond,trial)
                knee_l_musclesmoment_dir = '../subject{}/{}/loadedwalking_subject{}_{}_free_trial{}_analyze_MuscleAnalysis_Moment_knee_angle_l.sto'\
                                        .format(i,files_dir,i,loadcond,trial)
                musclemoment_dic = {"Right_Directory":hip_r_musclesmoment_dir,"Left_Directory":hip_l_musclesmoment_dir,"gl":gl}
                hip_muscle_moment = musclemoment_calc(Subject_Dic=musclemoment_dic)
                musclemoment_dic = {"Right_Directory":knee_r_musclesmoment_dir,"Left_Directory":knee_l_musclesmoment_dir,"gl":gl}
                knee_muscle_moment = musclemoment_calc(Subject_Dic=musclemoment_dic)
                HipMuscleMoment_Data[:,c:c+1] = hip_muscle_moment
                KneeMuscleMoment_Data[:,c:c+1] = knee_muscle_moment
            if metabolicrate == True:
                metabolic_power_data_dir = '../subject{}/{}/loadedwalking_subject{}_{}_free_trial{}_cmc_ProbeReporter_probes.sto'\
                                            .format(i,files_dir,i,loadcond,trial)
                energy_dic = {"Directory":metabolic_power_data_dir,
                                     "gl": gl,
                           "Subject_Mass": subject_mass}
                metabolic_energy = metabolic_normal_energy(energy_dic)
                MetabolicEnergy_Data[c] = metabolic_energy
            c+=1
    if configuration != 'UnAssist':
        return HipActuator_Torque_Data,KneeActuator_Torque_Data,HipActuator_Power_Data,KneeActuator_Power_Data,\
            HipActuator_Speed_Data,KneeActuator_Speed_Data,HipActuatorEnergy_Data,KneeActuatorEnergy_Data,\
            Regen_HipActuatorEnergy_Data,Regen_KneeActuatorEnergy_Data,MetabolicEnergy_Data,MuscleActivation_Data,\
            HipMuscleMoment_Data,KneeMuscleMoment_Data
    else:
        return MetabolicEnergy_Data,MuscleActivation_Data,HipMuscleMoment_Data,KneeMuscleMoment_Data
#####################################################################################
#####################################################################################

