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
from scipy import integrate
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
    elif loadcond == 'load' or loadcond == 'loaded':
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
    gpc_r, shifted_right_data = pp.data_by_pgc(time,right_data,gl,side='right')
    gpc_l, shifted_left_data  = pp.data_by_pgc(time,left_data,gl,side='left')
    interperted_right_data = np.interp(gait_cycle,gpc_r,shifted_right_data, left=np.nan, right=np.nan)
    interperted_left_data  = np.interp(gait_cycle,gpc_l,shifted_left_data, left=np.nan, right=np.nan)
    final_data = nanmean([interperted_right_data,interperted_left_data],axis=0)
    return final_data


def actuators_energy_calc(Subject_Dic,isabs=True,regen=False):
    """This function developed to calculate the consumed energy by actuators"""
    directory = Subject_Dic["Directory"]
    right_param = Subject_Dic["Right_Parameter"]
    left_param = Subject_Dic["Left_Parameter"]
    subject_mass = Subject_Dic["Subject_Mass"]
    gl = Subject_Dic["gl"]
    numpy_data = dataman.storage2numpy(directory)
    time = numpy_data['time']
    right_data = numpy_data[right_param]
    left_data = numpy_data[left_param]
    if isabs == True:
        if gl.primary_leg == 'left':
            energy = pp.avg_over_gait_cycle(time, np.abs(left_data),cycle_duration=gl.cycle_end-gl.cycle_start,cycle_start=gl.cycle_start)\
              
        else:
            energy = pp.avg_over_gait_cycle(time, np.abs(right_data),cycle_duration=gl.cycle_end-gl.cycle_start,cycle_start=gl.cycle_start)\
              
    else:
        if gl.primary_leg == 'left':
            energy = pp.avg_over_gait_cycle(time, left_data,cycle_duration=gl.cycle_end-gl.cycle_start,cycle_start=gl.cycle_start)\
                
        else:
             energy = pp.avg_over_gait_cycle(time, right_data,cycle_duration=gl.cycle_end-gl.cycle_start,cycle_start=gl.cycle_start)\
                
    if regen == True:
        if gl.primary_leg == 'left':
            regen_energy = pp.avg_over_gait_cycle(time, np.abs(np.minimum(np.zeros(left_data.shape[0]),right_data)),cycle_duration=gl.cycle_end-gl.cycle_start,cycle_start=gl.cycle_start)\
                        
        else:
            regen_energy = pp.avg_over_gait_cycle(time, np.abs(np.minimum(np.zeros(right_data.shape[0]),right_data)),cycle_duration=gl.cycle_end-gl.cycle_start,cycle_start=gl.cycle_start)\

    if regen == True:
        return energy/subject_mass,regen_energy/subject_mass
    else:
        return energy/subject_mass
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
    for i in range(len(muscles_name)):
        musclemoment_r += data_r[muscles_name[i]+'_l']
        musclemoment_l += data_l[muscles_name[i]+'_r']
    gpc = np.linspace(0,100,1000)
    gpc_r,shifted_data_r = pp.data_by_pgc(time_r,musclemoment_r,gl,side='right')
    gpc_l,shifted_data_l = pp.data_by_pgc(time_l,musclemoment_l,gl,side='left')
    main_data_r = np.interp(gpc,gpc_r,shifted_data_r, left=np.nan, right=np.nan)
    main_data_l = np.interp(gpc,gpc_l,shifted_data_l, left=np.nan, right=np.nan)
    musclemoment = nanmean([main_data_r,main_data_l],axis=0)
    return musclemoment


def metabolic_energy_fcn(Subject_Dic):
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
    metabolic_energy = pp.avg_over_gait_cycle(metabolic_power_data['time'], main_metabolics,cycle_duration=gl.cycle_end-gl.cycle_start,cycle_start=gl.cycle_start)
    #metabolic_energy = pp.avg_over_gait_cycle(metabolic_power_data['time'], main_metabolics,cycle_duration=gl.cycle_end-gl.cycle_start)
    return metabolic_energy/subject_mass


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
    TrialNo = Subject_Dic["TrialNo"]
    CycleNo = Subject_Dic["CycleNo"]
    gl = Subject_Dic["gl"]
    data_dir = '../subject{}/{}/Cycle{}/loadedwalking_subject{}_{}_free_trial{}_cmc_states.sto'\
                .format(SubjectNo,Directory,CycleNo,SubjectNo,loadcond,TrialNo)
    muscles_activation = np.zeros([1000,len(muscles_name)])
    gait_cycle = np.linspace(0,100,1000)
    data = dataman.storage2numpy(data_dir)
    time = data['time']
    c = 0
    for muscle in muscles_name:
        muscle_r_activation = data[muscle+'_r'+'activation']
        muscle_l_activation = data[muscle+'_l'+'activation']
        gpc_r, shifted_muscle_r_activation = pp.data_by_pgc(time,muscle_r_activation,gl,side='right')
        gpc_l, shifted_muscle_l_activation = pp.data_by_pgc(time,muscle_l_activation,gl,side='left')
        muscle_r_activation = np.interp(gait_cycle,gpc_r,shifted_muscle_r_activation, left=np.nan, right=np.nan)
        muscle_l_activation = np.interp(gait_cycle,gpc_l,shifted_muscle_l_activation, left=np.nan, right=np.nan)
        muscles_activation[:,c]=nanmean([muscle_r_activation,muscle_l_activation],axis=0)
        c+=1
    return muscles_activation

################################################################# 
# Metabolic and actuators energy calculation using their processed data
#****************************************************************

def actuator_energy_proc_power(Subject_Dic,isabs=True,regen=False):
    """This function developed to calculate the consumed energy by actuators.
    -Attention:
                -The approach of this function is completely different than "actuators_energy_calc"
                 approach for calculating the energy over the gait cycle.
                - This approach uses Simpson's method for integration.
    """
    data_extraction_dic = {"Directory":Subject_Dic["Directory"],
                                    "Right_Parameter": Subject_Dic["Right_Parameter"],
                                     "Left_Parameter": Subject_Dic["Left_Parameter"],
                                                 "gl": Subject_Dic["gl"]}
    power_data = data_extraction(Subject_Dic=data_extraction_dic)
    gl = Subject_Dic["gl"]
    gcp = np.linspace(gl.cycle_start,gl.cycle_end,1000)
    subject_mass = Subject_Dic["Subject_Mass"]
    zeros = np.zeros(1000)
    positive_power_data = np.maximum(zeros,power_data)
    negative_power_data = np.minimum(zeros,power_data)
    positive_energy = integrate.simps(positive_power_data[~np.isnan(positive_power_data)],gcp[~np.isnan(positive_power_data)])
    negative_energy = np.abs(integrate.simps(negative_power_data[~np.isnan(negative_power_data)],gcp[~np.isnan(negative_power_data)]))
    if isabs == True:
        total_energy = positive_energy + negative_energy
    else:
        total_energy = positive_energy - negative_energy
    if regen == False:
        return total_energy/subject_mass
    else:
        return total_energy/subject_mass, negative_energy/subject_mass


def metabolic_energy_instant_power(Subject_Dic):
    """This function sums up the instantaneous power of each muscle at 
    each side and then normalizes them to a gait cycle. The final metabolic
    power then has been integrated using simpson's method and normalized.
    """
    muscles_name = ['add_brev','add_long','add_mag3','add_mag4','add_mag2','add_mag1','bifemlh','bifemsh','ext_dig',\
                    'ext_hal','flex_dig','flex_hal','lat_gas','med_gas','glut_max1','glut_max2','glut_max3','glut_med1',\
                    'glut_med2','glut_med3','glut_min1','glut_min2','glut_min3','grac','iliacus','per_brev','per_long',\
                    'peri','psoas','rect_fem','sar','semimem','semiten','soleus','tfl','tib_ant','tib_post','vas_int',\
                    'vas_lat','vas_med']
    directory = Subject_Dic["Directory"]
    gl = Subject_Dic["gl"]
    subject_mass = Subject_Dic["Subject_Mass"]
    gait_cycle = np.linspace(0,100,1000)
    gcp = np.linspace(gl.cycle_start,gl.cycle_end,1000)
    rawdata = dataman.storage2numpy(directory)
    time = rawdata['time']
    basal = np.mean(rawdata['metabolic_power_BASAL'])*np.ones(1000)
    sum_metabolics_r = np.zeros(rawdata.shape[0])
    sum_metabolics_l = np.zeros(rawdata.shape[0])
    for muscle in muscles_name:
        sum_metabolics_r += rawdata['metabolic_power_'+muscle+'_r']
        sum_metabolics_l += rawdata['metabolic_power_'+muscle+'_l']
    gpc_r, shifted_metabolics_r = pp.data_by_pgc(time,sum_metabolics_r,gl,side='right')
    gpc_l, shifted_metabolics_l = pp.data_by_pgc(time,sum_metabolics_l,gl,side='left')
    interp_metabolics_r = np.interp(gait_cycle,gpc_r,shifted_metabolics_r, left=np.nan, right=np.nan)
    interp_metabolics_l = np.interp(gait_cycle,gpc_l,shifted_metabolics_l, left=np.nan, right=np.nan)
    metabolics = nanmean([interp_metabolics_r,interp_metabolics_l],axis=0) + basal
    normal_metabolics_energy = integrate.simps(metabolics[~np.isnan(metabolics)],gcp[~np.isnan(metabolics)])/subject_mass
    return metabolics, normal_metabolics_energy
    
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
    if Directory == 'noloaded/Subject10_NoLoaded_Dataset/Monoarticular':
        print('trial number has been changed to get match with simulation files in subject 10 noloaded')
        TrialNo = '01'
    elif Directory == 'noloaded/Subject12_NoLoaded_Dataset/Monoarticular' or  Directory == 'noloaded/Subject12_NoLoaded_Dataset/Biarticular':
        print('trial number has been changed to get match with simulation files in subject 12 noloaded')
        TrialNo = '05'
    optimal_force = 1000
    hip_list = [70/1000,60/1000,50/1000,40/1000,30/1000]
    knee_list = [70/1000,60/1000,50/1000,40/1000,30/1000]
    # Initialization
    gait_cycle = np.linspace(0,100,1000)
    unsimulated = []
    KneeActuatorEnergy_Data = np.zeros(len(hip_list)*len(knee_list))
    HipActuatorEnergy_Data = np.zeros(len(hip_list)*len(knee_list))
    MetabolicEnergy_Data = np.zeros(len(hip_list)*len(knee_list))
    KneeActuatorEnergy_FromPower_Data = np.zeros(len(hip_list)*len(knee_list))
    HipActuatorEnergy_FromPower_Data = np.zeros(len(hip_list)*len(knee_list))
    MetabolicEnergy_FromPower_Data = np.zeros(len(hip_list)*len(knee_list))
    HipActuator_Torque_Data = np.zeros([1000,len(hip_list)*len(knee_list)])
    KneeActuator_Torque_Data = np.zeros([1000,len(hip_list)*len(knee_list)])
    HipActuator_Power_Data = np.zeros([1000,len(hip_list)*len(knee_list)])
    KneeActuator_Power_Data = np.zeros([1000,len(hip_list)*len(knee_list)])
    Metabolics_Power_Data = np.zeros([1000,len(hip_list)*len(knee_list)])
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
            
            if os.path.exists(actuator_torque_data_dir) == False or os.path.exists(actuator_power_data_dir) == False or os.path.exists(metabolic_power_data_dir) == False:
                
                print('***Subject{} : H{}K{} has not been simulated***'.format(SubjectNo,int(hip_max_control*optimal_force),int(knee_max_control*optimal_force)))
                unsimulated.append('Subject{}_H{}K{}'.format(SubjectNo,int(hip_max_control*optimal_force),int(knee_max_control*optimal_force)))
            
            else:
            
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
                    hip_actuator_energy = actuators_energy_calc(Subject_Dic=energy_dic)
                    energy_dic = {"Directory":actuator_power_data_dir,
                            "Right_Parameter":'Knee_Right_Actuator',
                            "Left_Parameter":'Knee_Left_Actuator',
                                        "gl": gl,
                            "Subject_Mass": subject_mass}
                    knee_actuator_energy = actuators_energy_calc(Subject_Dic=energy_dic)
                    energy_dic = {"Directory":metabolic_power_data_dir,
                                        "gl": gl,
                            "Subject_Mass": subject_mass}
                    metabolic_energy = metabolic_energy_fcn(energy_dic)
                    energy_dic = {"Directory":actuator_power_data_dir,
                            "Right_Parameter":'Hip_Right_Actuator',
                            "Left_Parameter":'Hip_Left_Actuator',
                                        "gl": gl,
                            "Subject_Mass": subject_mass}
                    hip_actuator_energy_from_power = actuator_energy_proc_power(Subject_Dic=energy_dic)
                    energy_dic = {"Directory":actuator_power_data_dir,
                            "Right_Parameter":'Knee_Right_Actuator',
                            "Left_Parameter":'Knee_Left_Actuator',
                                        "gl": gl,
                            "Subject_Mass": subject_mass}
                    knee_actuator_energy_from_power = actuator_energy_proc_power(Subject_Dic=energy_dic)
                    energy_dic = {"Directory":metabolic_power_data_dir,
                                        "gl": gl,
                            "Subject_Mass": subject_mass}
                    metabolic_power,metabolic_energy_from_power = metabolic_energy_instant_power(energy_dic)
                    HipActuatorEnergy_Data[c]  = hip_actuator_energy
                    KneeActuatorEnergy_Data[c] = knee_actuator_energy
                    MetabolicEnergy_Data[c] = metabolic_energy
                    HipActuatorEnergy_FromPower_Data[c]  = hip_actuator_energy_from_power
                    KneeActuatorEnergy_FromPower_Data[c] = knee_actuator_energy_from_power
                    MetabolicEnergy_FromPower_Data[c] = metabolic_energy_from_power
                    Metabolics_Power_Data[:,c] = metabolic_power
                c+=1
    return HipActuator_Torque_Data,KneeActuator_Torque_Data,HipActuator_Power_Data,KneeActuator_Power_Data,\
           HipActuatorEnergy_Data,KneeActuatorEnergy_Data,MetabolicEnergy_Data,\
           HipActuatorEnergy_FromPower_Data,KneeActuatorEnergy_FromPower_Data,MetabolicEnergy_FromPower_Data,\
           Metabolics_Power_Data,unsimulated


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
    trials_num = ['01']
    # initialization
    unsimulated = []
    KneeActuatorEnergy_Data = np.zeros(len(subjects)*len(trials_num)*len(hip_list)*len(knee_list))
    HipActuatorEnergy_Data = np.zeros(len(subjects)*len(trials_num)*len(hip_list)*len(knee_list))
    MetabolicEnergy_Data = np.zeros(len(subjects)*len(trials_num)*len(hip_list)*len(knee_list))
    KneeActuatorEnergy_fromPower_Data = np.zeros(len(subjects)*len(trials_num)*len(hip_list)*len(knee_list))
    HipActuatorEnergy_fromPower_Data = np.zeros(len(subjects)*len(trials_num)*len(hip_list)*len(knee_list))
    MetabolicEnergy_fromPower_Data = np.zeros(len(subjects)*len(trials_num)*len(hip_list)*len(knee_list))
    HipActuator_Torque_Data = np.zeros([1000,len(subjects)*len(trials_num)*len(hip_list)*len(knee_list)])
    KneeActuator_Torque_Data = np.zeros([1000,len(subjects)*len(trials_num)*len(hip_list)*len(knee_list)])
    HipActuator_Power_Data = np.zeros([1000,len(subjects)*len(trials_num)*len(hip_list)*len(knee_list)])
    KneeActuator_Power_Data = np.zeros([1000,len(subjects)*len(trials_num)*len(hip_list)*len(knee_list)])
    Metabolic_Power_Data = np.zeros([1000,len(subjects)*len(trials_num)*len(hip_list)*len(knee_list)])
    
    c = 0
    for i in subjects:
        for j in trials_num:
            # subject/trial/directory construction
            gl,_,trial = construct_gl_mass_trial(subjectno=i,trialno=j,loadcond=loadcond)
            _,subject_mass,_ = construct_gl_mass_trial(subjectno=i,trialno=j,loadcond='noload')
            if loadcond == 'noload':
                files_dir = 'noloaded/Subject{}_NoLoaded_Dataset/{}'.format(i,configuration)
            elif loadcond == 'load':
                files_dir = 'loaded/Subject{}_Loaded_Dataset/{}'.format(i,configuration)
            else:
                raise Exception('Invalid load condition!')
            Subject_Dictionary = {"Directory": files_dir,
                                  "SubjectNo": i,
                                    "TrialNo": '02',
                                         "gl": gl,
                               "Subject_Mass":subject_mass}
            
            hip_torque,knee_torque,hip_power,knee_power,hip_energy,\
            knee_energy,metabolics_energy,hip_energy_frompower,knee_energy_frompower,\
            metabolics_energy_frompower,metabolic_power,unsuccessful = \
            pareto_data_extraction(Subject_Dictionary,loadcond=loadcond)
            # saving data into initialized variables
            unsimulated.append(unsuccessful)
            HipActuator_Torque_Data[:,c:c+len(hip_list)*len(knee_list)]  = hip_torque
            KneeActuator_Torque_Data[:,c:c+len(hip_list)*len(knee_list)] = knee_torque
            HipActuator_Power_Data[:,c:c+len(hip_list)*len(knee_list)]   = hip_power
            KneeActuator_Power_Data[:,c:c+len(hip_list)*len(knee_list)]  = knee_power
            HipActuatorEnergy_Data[c:c+len(hip_list)*len(knee_list)]     = hip_energy
            KneeActuatorEnergy_Data[c:c+len(hip_list)*len(knee_list)]    = knee_energy
            MetabolicEnergy_Data[c:c+len(hip_list)*len(knee_list)]       = metabolics_energy
            Metabolic_Power_Data [:,c:c+len(hip_list)*len(knee_list)]      = metabolic_power
            HipActuatorEnergy_fromPower_Data[c:c+len(hip_list)*len(knee_list)]     = hip_energy
            KneeActuatorEnergy_fromPower_Data[c:c+len(hip_list)*len(knee_list)]    = knee_energy
            MetabolicEnergy_fromPower_Data[c:c+len(hip_list)*len(knee_list)]       = metabolics_energy
            c+=len(hip_list)*len(knee_list)
    return HipActuator_Torque_Data,KneeActuator_Torque_Data,HipActuator_Power_Data,KneeActuator_Power_Data,\
            HipActuatorEnergy_Data,KneeActuatorEnergy_Data,MetabolicEnergy_Data,\
            HipActuatorEnergy_fromPower_Data,KneeActuatorEnergy_fromPower_Data,MetabolicEnergy_fromPower_Data,\
            Metabolic_Power_Data,unsimulated
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
                    hip_actuator_energy,hip_actuator_regen_energy = actuators_energy_calc(Subject_Dic=energy_dic,regen=regen_energy)
                else:
                    hip_actuator_energy = actuators_energy_calc(Subject_Dic=energy_dic,regen=regen_energy)
                energy_dic = {"Directory":actuator_power_data_dir,
                        "Right_Parameter":'Knee_Right_Actuator',
                         "Left_Parameter":'Knee_Left_Actuator',
                                     "gl": gl,
                           "Subject_Mass": subject_mass}
                if regen_energy == True:
                    knee_actuator_energy,hip_actuator_regen_energy = actuators_energy_calc(Subject_Dic=energy_dic,regen=regen_energy)
                else:
                    knee_actuator_energy = actuators_energy_calc(Subject_Dic=energy_dic,regen=regen_energy)
                energy_dic = {"Directory":metabolic_power_data_dir,
                                     "gl": gl,
                           "Subject_Mass": subject_mass}
                metabolic_energy = metabolic_energy_fcn(energy_dic)
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
    trials_num = ['01']
    whichgroup='nine'
    musclesgroup = 9
    # initialization
    KneeActuatorEnergy_Data = np.zeros(len(subjects)*len(trials_num))
    HipActuatorEnergy_Data = np.zeros(len(subjects)*len(trials_num))
    Regen_KneeActuatorEnergy_Data = np.zeros(len(subjects)*len(trials_num))
    Regen_HipActuatorEnergy_Data = np.zeros(len(subjects)*len(trials_num))
    MetabolicEnergy_Data = np.zeros(len(subjects)*len(trials_num))
    HipActuator_Torque_Data = np.zeros([1000,len(subjects)*len(trials_num)])
    KneeActuator_Torque_Data = np.zeros([1000,len(subjects)*len(trials_num)])
    HipActuator_Power_Data = np.zeros([1000,len(subjects)*len(trials_num)])
    KneeActuator_Power_Data = np.zeros([1000,len(subjects)*len(trials_num)])
    HipActuator_Speed_Data = np.zeros([1000,len(subjects)*len(trials_num)])
    KneeActuator_Speed_Data = np.zeros([1000,len(subjects)*len(trials_num)])
    HipMuscleMoment_Data = np.zeros([1000,len(subjects)*len(trials_num)])
    KneeMuscleMoment_Data = np.zeros([1000,len(subjects)*len(trials_num)])
    MuscleActivation_Data = np.zeros([1000,len(subjects)*len(trials_num)*musclesgroup])
    c = 0
    c_m = 0
    for i in subjects:
        for j in trials_num:
            # subject/trial/directory construction
            gl,_,trial = construct_gl_mass_trial(subjectno=i,trialno=j,loadcond=loadcond)
            _,subject_mass,_ = construct_gl_mass_trial(subjectno=i,trialno=j,loadcond='noload')
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
            HipActuator_Torque_Data[:,c]  = hip_torque
            KneeActuator_Torque_Data[:,c] = knee_torque
            HipActuator_Power_Data[:,c]   = hip_power
            KneeActuator_Power_Data[:,c]  = knee_power
            HipActuator_Speed_Data[:,c]   = hip_speed
            KneeActuator_Speed_Data[:,c]  = knee_speed
            HipActuatorEnergy_Data[c]     = hip_energy
            KneeActuatorEnergy_Data[c]    = knee_energy
            MetabolicEnergy_Data[c]       = metabolics_energy
            if musclesactivation == True:
                muscles_activation = group_muscles_activation(Subject_Dictionary,whichgroup=whichgroup,loadcond=loadcond)
                MuscleActivation_Data[:,c_m:c_m+musclesgroup] = muscles_activation
                c_m+=musclesgroup
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
                HipMuscleMoment_Data[:,c]  = hip_muscle_moment
                KneeMuscleMoment_Data[:,c] = knee_muscle_moment
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
    joint_torque_data_dir= '../subject{}/{}/Cycle{}/loadedwalking_subject{}_{}_free_trial{}_rratasks_Actuation_force.sto'\
    .format(SubjectNo,Directory,CycleNo,SubjectNo,loadcond,TrialNo)
    joint_power_data_dir= '../subject{}/{}/Cycle{}/loadedwalking_subject{}_{}_free_trial{}_rratasks_Actuation_power.sto'\
    .format(SubjectNo,Directory,CycleNo,SubjectNo,loadcond,TrialNo)
    joint_speed_data_dir= '../subject{}/{}/Cycle{}/loadedwalking_subject{}_{}_free_trial{}_rratasks_Actuation_speed.sto'\
    .format(SubjectNo,Directory,CycleNo,SubjectNo,loadcond,TrialNo)
    joint_kinematics_data_dir= '../subject{}/{}/Cycle{}/loadedwalking_subject{}_{}_free_trial{}_rratasks_Kinematics_q.sto'\
    .format(SubjectNo,Directory,CycleNo,SubjectNo,loadcond,TrialNo)
    data_extraction_dic = {"Directory":joint_torque_data_dir,
                                 "Right_Parameter":'hip_flexion_r',
                                  "Left_Parameter":'hip_flexion_l',
                                              "gl": gl}
    hip_joint_torque = data_extraction(Subject_Dic=data_extraction_dic)
    data_extraction_dic = {"Directory":joint_torque_data_dir,
                                 "Right_Parameter":'knee_angle_r',
                                  "Left_Parameter":'knee_angle_l',
                                              "gl": gl}
    knee_joint_torque = data_extraction(Subject_Dic=data_extraction_dic)
    data_extraction_dic = {"Directory":joint_power_data_dir,
                                 "Right_Parameter":'hip_flexion_r',
                                  "Left_Parameter":'hip_flexion_l',
                                              "gl": gl}
    hip_joint_power = data_extraction(Subject_Dic=data_extraction_dic)
    data_extraction_dic = {"Directory":joint_power_data_dir,
                                 "Right_Parameter":'knee_angle_r',
                                  "Left_Parameter":'knee_angle_l',
                                              "gl": gl}
    knee_joint_power = data_extraction(Subject_Dic=data_extraction_dic)
    data_extraction_dic = {"Directory":joint_speed_data_dir,
                                 "Right_Parameter":'hip_flexion_r',
                                  "Left_Parameter":'hip_flexion_l',
                                              "gl": gl}
    hip_joint_speed = data_extraction(Subject_Dic=data_extraction_dic)
    data_extraction_dic = {"Directory":joint_speed_data_dir,
                                 "Right_Parameter":'knee_angle_r',
                                  "Left_Parameter":'knee_angle_l',
                                              "gl": gl}
    knee_joint_speed = data_extraction(Subject_Dic=data_extraction_dic)
    data_extraction_dic = {"Directory":joint_kinematics_data_dir,
                                 "Right_Parameter":'hip_flexion_r',
                                  "Left_Parameter":'hip_flexion_l',
                                              "gl": gl}
    hip_joint_kinematics = data_extraction(Subject_Dic=data_extraction_dic)
    data_extraction_dic = {"Directory":joint_kinematics_data_dir,
                                 "Right_Parameter":'knee_angle_r',
                                  "Left_Parameter":'knee_angle_l',
                                              "gl": gl}
    knee_joint_kinematics = data_extraction(Subject_Dic=data_extraction_dic)
    
    return hip_joint_torque,knee_joint_torque,hip_joint_power,knee_joint_power,\
           hip_joint_speed,knee_joint_speed,hip_joint_kinematics,knee_joint_kinematics


def rra_data_subjects(loadcond='noload'):
    """This function generalize the rra_data_extraction for all subjects.
    -Default setting for muscle activation is nine representitive muscles of the lower extermity. It can be changed to knee/hip/both.
     """
    subjects = ['05','07','09','10','11','12','14']
    trials_num = ['01','02','03']
    whichgroup='nine'
    musclesgroup = 9
    # initialization
    HipJoint_Torque_Data = np.zeros([1000,len(subjects)*len(trials_num)])
    KneeJoint_Torque_Data = np.zeros([1000,len(subjects)*len(trials_num)])
    HipJoint_Power_Data = np.zeros([1000,len(subjects)*len(trials_num)])
    KneeJoint_Power_Data = np.zeros([1000,len(subjects)*len(trials_num)])
    HipJoint_Speed_Data = np.zeros([1000,len(subjects)*len(trials_num)])
    KneeJoint_Speed_Data = np.zeros([1000,len(subjects)*len(trials_num)])
    HipJoint_Kinematics_Data = np.zeros([1000,len(subjects)*len(trials_num)])
    KneeJoint_Kinematics_Data = np.zeros([1000,len(subjects)*len(trials_num)])
    
    c = 0
    for i in subjects:
        for j in trials_num:
            # subject/trial/directory construction
            gl,subject_mass,trial = construct_gl_mass_trial(subjectno=i,trialno=j,loadcond=loadcond)
            if loadcond == 'noload':
                files_dir = 'noloaded/RRA'
            else:
                files_dir = 'loaded/RRA'
            Subject_Dictionary = {"Directory": files_dir,
                                  "SubjectNo": i,
                                    "TrialNo": trial,
                                    "CycleNo": j,
                                         "gl": gl,
                               "subject_mass":subject_mass}
            hip_torque,knee_torque,hip_power,knee_power,hip_speed,knee_speed,\
            hip_kinematics,knee_kinematics = rra_data_extraction(Subject_Dictionary,loadcond=loadcond)
            # saving data into initialized variables
            HipJoint_Torque_Data[:,c]  = hip_torque
            KneeJoint_Torque_Data[:,c] = knee_torque
            HipJoint_Power_Data[:,c]   = hip_power
            KneeJoint_Power_Data[:,c]  = knee_power
            HipJoint_Speed_Data[:,c]   = hip_speed
            KneeJoint_Speed_Data[:,c]  = knee_speed
            HipJoint_Kinematics_Data[:,c]   = hip_kinematics
            KneeJoint_Kinematics_Data[:,c]  = knee_kinematics
            c+=1
    return HipJoint_Torque_Data,KneeJoint_Torque_Data,HipJoint_Power_Data,KneeJoint_Power_Data,\
           HipJoint_Speed_Data,KneeJoint_Speed_Data,HipJoint_Kinematics_Data,KneeJoint_Kinematics_Data


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
    CycleNo = Subject_Dic["CycleNo"]
    gl = Subject_Dic["gl"]
    subject_mass = Subject_Dic["subject_mass"]
    gait_cycle = np.linspace(0,100,1000)
    # Directory of each .sto files
    actuator_torque_data_dir= '../subject{}/{}/Cycle{}/loadedwalking_subject{}_{}_free_trial{}_cmc_Actuation_force.sto'\
    .format(SubjectNo,Directory,CycleNo,SubjectNo,loadcond,TrialNo)
    actuator_power_data_dir= '../subject{}/{}/Cycle{}/loadedwalking_subject{}_{}_free_trial{}_cmc_Actuation_power.sto'\
    .format(SubjectNo,Directory,CycleNo,SubjectNo,loadcond,TrialNo)
    actuator_speed_data_dir= '../subject{}/{}/Cycle{}/loadedwalking_subject{}_{}_free_trial{}_cmc_Actuation_speed.sto'\
    .format(SubjectNo,Directory,CycleNo,SubjectNo,loadcond,TrialNo)
    metabolic_power_data_dir = '../subject{}/{}/Cycle{}/loadedwalking_subject{}_{}_free_trial{}_cmc_ProbeReporter_probes.sto'\
    .format(SubjectNo,Directory,CycleNo,SubjectNo,loadcond,TrialNo)
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
                    hip_actuator_energy = actuators_energy_calc(Subject_Dic=energy_dic,regen=regen_energy)
                    hip_proc_actuator_energy = actuator_energy_proc_power(Subject_Dic=energy_dic,regen=regen_energy)
                else:
                    hip_actuator_energy,hip_actuator_regen_energy = actuators_energy_calc(Subject_Dic=energy_dic,regen=regen_energy)
                    hip_proc_actuator_energy,hip_proc_actuator_regen_energy = actuator_energy_proc_power(Subject_Dic=energy_dic,regen=regen_energy)
                energy_dic = {"Directory":actuator_power_data_dir,
                        "Right_Parameter":'Knee_Right_Actuator',
                         "Left_Parameter":'Knee_Left_Actuator',
                                     "gl": gl,
                           "Subject_Mass": subject_mass}
                if regen_energy == False:
                    knee_actuator_energy = actuators_energy_calc(Subject_Dic=energy_dic,regen=regen_energy)
                    knee_proc_actuator_energy = actuator_energy_proc_power(Subject_Dic=energy_dic,regen=regen_energy)
                else:
                     knee_actuator_energy,knee_actuator_regen_energy = actuators_energy_calc(Subject_Dic=energy_dic,regen=regen_energy)
                     knee_proc_actuator_energy,knee_proc_actuator_regen_energy = actuator_energy_proc_power(Subject_Dic=energy_dic,regen=regen_energy)
    if regen_energy == False:
        return hip_actuator_torque,knee_actuator_torque,hip_actuator_power,knee_actuator_power,\
            hip_actuator_speed,knee_actuator_speed,hip_actuator_energy,knee_actuator_energy,hip_proc_actuator_energy,knee_proc_actuator_energy
    else:
        return hip_actuator_torque,knee_actuator_torque,hip_actuator_power,knee_actuator_power,\
            hip_actuator_speed,knee_actuator_speed,hip_actuator_energy,knee_actuator_energy,\
            hip_actuator_regen_energy,knee_actuator_regen_energy,hip_proc_actuator_energy,knee_proc_actuator_energy,\
            hip_proc_actuator_regen_energy,knee_proc_actuator_regen_energy


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
    trials_num = ['01','02','03']
    whichgroup='nine'
    musclesgroup = 9
    # initialization
    KneeActuatorEnergy_Data = np.zeros(len(subjects)*len(trials_num))
    HipActuatorEnergy_Data = np.zeros(len(subjects)*len(trials_num))
    Regen_KneeActuatorEnergy_Data = np.zeros(len(subjects)*len(trials_num))
    Regen_HipActuatorEnergy_Data = np.zeros(len(subjects)*len(trials_num))
    MetabolicEnergy_Data = np.zeros(len(subjects)*len(trials_num))
    KneeActuatorEnergy_Proc_Data = np.zeros(len(subjects)*len(trials_num))
    HipActuatorEnergy_Proc_Data = np.zeros(len(subjects)*len(trials_num))
    Regen_KneeActuatorEnergy_Proc_Data = np.zeros(len(subjects)*len(trials_num))
    Regen_HipActuatorEnergy_Proc_Data = np.zeros(len(subjects)*len(trials_num))
    MetabolicEnergy_Proc_Data = np.zeros(len(subjects)*len(trials_num))
    HipActuator_Torque_Data = np.zeros([1000,len(subjects)*len(trials_num)])
    KneeActuator_Torque_Data = np.zeros([1000,len(subjects)*len(trials_num)])
    HipActuator_Power_Data = np.zeros([1000,len(subjects)*len(trials_num)])
    KneeActuator_Power_Data = np.zeros([1000,len(subjects)*len(trials_num)])
    Metabolics_Power_Data = np.zeros([1000,len(subjects)*len(trials_num)])
    HipActuator_Speed_Data = np.zeros([1000,len(subjects)*len(trials_num)])
    KneeActuator_Speed_Data = np.zeros([1000,len(subjects)*len(trials_num)])
    HipMuscleMoment_Data = np.zeros([1000,len(subjects)*len(trials_num)])
    KneeMuscleMoment_Data = np.zeros([1000,len(subjects)*len(trials_num)])
    MuscleActivation_Data = np.zeros([1000,len(subjects)*len(trials_num)*musclesgroup])
    c = 0
    c_m = 0
    for i in subjects:
        for j in trials_num:
            # subject/trial/directory construction
            gl,_,trial = construct_gl_mass_trial(subjectno=i,trialno=j,loadcond=loadcond)
            _,subject_mass,_ = construct_gl_mass_trial(subjectno=i,trialno=j,loadcond='noload')
            if loadcond == 'noload':
                files_dir = 'noloaded/Subject{}_NoLoaded_Dataset/{}'.format(i,configuration)
            else:
                files_dir = 'loaded/Subject{}_Loaded_Dataset/{}'.format(i,configuration)
            Subject_Dictionary = {"Directory": files_dir,
                                  "SubjectNo": i,
                                    "TrialNo": trial,
                                    "CycleNo": j,
                                         "gl": gl,
                               "subject_mass":subject_mass}
            if configuration !='UnAssist':
                if regenergy == False:
                    hip_torque,knee_torque,hip_power,knee_power,hip_speed,\
                    knee_speed,hip_energy,knee_energy,hip_proc_energy,knee_proc_energy = \
                    idealdevice_data_extraction(Subject_Dictionary,loadcond=loadcond)
                else:
                    hip_torque,knee_torque,hip_power,knee_power,hip_speed,\
                    knee_speed,hip_energy,knee_energy,hip_regen_energy,knee_regen_energy,\
                    hip_proc_energy,knee_proc_energy,hip_proc_regen_energy,knee_proc_regen_energy = \
                    idealdevice_data_extraction(Subject_Dictionary,loadcond=loadcond,regen_energy=regenergy)
                    Regen_HipActuatorEnergy_Data[c]     = hip_regen_energy
                    Regen_KneeActuatorEnergy_Data[c]    = knee_regen_energy
                    Regen_HipActuatorEnergy_Proc_Data[c]  = hip_proc_regen_energy
                    Regen_KneeActuatorEnergy_Proc_Data[c] = knee_proc_regen_energy
                    
                # saving data into initialized variables
                HipActuator_Torque_Data[:,c]  = hip_torque
                KneeActuator_Torque_Data[:,c] = knee_torque
                HipActuator_Power_Data[:,c]   = hip_power
                KneeActuator_Power_Data[:,c]  = knee_power
                HipActuator_Speed_Data[:,c]   = hip_speed
                KneeActuator_Speed_Data[:,c]  = knee_speed
                HipActuatorEnergy_Data[c]     = hip_energy
                KneeActuatorEnergy_Data[c]    = knee_energy
                HipActuatorEnergy_Proc_Data[c]     = hip_proc_energy
                KneeActuatorEnergy_Proc_Data[c]    = knee_proc_energy
            if musclesactivation == True:
                muscles_activation = group_muscles_activation(Subject_Dictionary,whichgroup=whichgroup,loadcond=loadcond)
                MuscleActivation_Data[:,c_m:c_m+musclesgroup] = muscles_activation
                c_m+=musclesgroup
            if musclesmoment == True:
                hip_r_musclesmoment_dir = '../subject{}/{}/Cycle{}/loadedwalking_subject{}_{}_free_trial{}_analyze_MuscleAnalysis_Moment_hip_flexion_r.sto'\
                                        .format(i,files_dir,j,i,loadcond,trial)
                hip_l_musclesmoment_dir = '../subject{}/{}/Cycle{}/loadedwalking_subject{}_{}_free_trial{}_analyze_MuscleAnalysis_Moment_hip_flexion_l.sto'\
                                        .format(i,files_dir,j,i,loadcond,trial)
                knee_r_musclesmoment_dir = '../subject{}/{}/Cycle{}/loadedwalking_subject{}_{}_free_trial{}_analyze_MuscleAnalysis_Moment_knee_angle_r.sto'\
                                        .format(i,files_dir,j,i,loadcond,trial)
                knee_l_musclesmoment_dir = '../subject{}/{}/Cycle{}/loadedwalking_subject{}_{}_free_trial{}_analyze_MuscleAnalysis_Moment_knee_angle_l.sto'\
                                        .format(i,files_dir,j,i,loadcond,trial)
                musclemoment_dic = {"Right_Directory":hip_r_musclesmoment_dir,"Left_Directory":hip_l_musclesmoment_dir,"gl":gl}
                hip_muscle_moment = musclemoment_calc(Subject_Dic=musclemoment_dic)
                musclemoment_dic = {"Right_Directory":knee_r_musclesmoment_dir,"Left_Directory":knee_l_musclesmoment_dir,"gl":gl}
                knee_muscle_moment = musclemoment_calc(Subject_Dic=musclemoment_dic)
                HipMuscleMoment_Data[:,c]  = hip_muscle_moment
                KneeMuscleMoment_Data[:,c] = knee_muscle_moment
            if metabolicrate == True:
                metabolic_power_data_dir = '../subject{}/{}/Cycle{}/loadedwalking_subject{}_{}_free_trial{}_cmc_ProbeReporter_probes.sto'\
                                            .format(i,files_dir,j,i,loadcond,trial)
                energy_dic = {"Directory":metabolic_power_data_dir,
                                     "gl": gl,
                           "Subject_Mass": subject_mass}
                metabolic_energy = metabolic_energy_fcn(energy_dic)
                _,metabolic_proc_energy = metabolic_energy_instant_power(energy_dic)
                MetabolicEnergy_Proc_Data[c] = metabolic_proc_energy
                MetabolicEnergy_Data[c] = metabolic_energy
            c+=1
    if configuration != 'UnAssist':
        return HipActuator_Torque_Data,KneeActuator_Torque_Data,HipActuator_Power_Data,KneeActuator_Power_Data,\
            HipActuator_Speed_Data,KneeActuator_Speed_Data,HipActuatorEnergy_Data,KneeActuatorEnergy_Data,\
            Regen_HipActuatorEnergy_Data,Regen_KneeActuatorEnergy_Data,MetabolicEnergy_Data,MuscleActivation_Data,\
            HipMuscleMoment_Data,KneeMuscleMoment_Data,\
            HipActuatorEnergy_Proc_Data,KneeActuatorEnergy_Proc_Data,\
            Regen_HipActuatorEnergy_Proc_Data,Regen_KneeActuatorEnergy_Proc_Data,MetabolicEnergy_Proc_Data
    else:
        return MetabolicEnergy_Data,MuscleActivation_Data,HipMuscleMoment_Data,KneeMuscleMoment_Data,MetabolicEnergy_Proc_Data
#####################################################################################
#####################################################################################
# TODO reserve/residual forces and pErr control.
