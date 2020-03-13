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
from scipy import integrate
import importlib
from tabulate import tabulate
from numpy import nanmean, nanstd
from perimysium import postprocessing as pp
from perimysium import dataman
import pathlib
#######################################################################
#######################################################################
# Data saving and reading related functions

def listToString(s):
    """fmt = ",".join(["%s"] + ["%s"] * (Hip_JointMoment.shape[1]-1))
    numpy.savetxt, at least as of numpy 1.6.2, writes bytes
    to file, which doesn't work with a file open in text mode.  To
    work around this deficiency, open the file in binary mode, and
    write out the header as bytes."""
    # initialize an empty string 
    str1 = ""  
    # traverse in the string   
    for ele in s:  
        str1 += (ele + ",")    
    # return string   
    return str1  

def csv2numpy(datname):
    """it performs storage2numpy task for csv files with headers."""
    f = open(datname, 'r')
    # read the headers
    line = f.readline()
    # making list of headers seperated by ','
    column_name = line.split(',')
    # eleminating last column name which is '\n'
    column_name.pop()
    f.close()
    data = np.genfromtxt(datname,delimiter= ',', names=column_name,skip_header=1)
    return data

def recover_muscledata(data,prefix,whichgroup='nine'):
    headers = muscles_header(prefix,whichgroup=whichgroup)
    recovered_data = np.zeros((data.shape[0],len(headers)))
    c=0
    for header in headers:
        recovered_data[:,c] = data[header]
        c+=1
    return recovered_data

def vec2mat(Data,matrix_cols=0,num_matrix=0):
    """This function concatenates vectors to establish matrix""" 
    datanp = np.zeros((1000,len(Data)+num_matrix*matrix_cols-num_matrix))
    c=0
    for i in range(len(Data)):
        if Data[i].size > 1000 :
            datanp[:,c:c+matrix_cols]=Data[i]
            c+=matrix_cols
        else:
            datanp[:,c]=Data[i]
            c+=1
    return datanp

def muscles_header(prefix,whichgroup='nine'):
    """This function has been established to generate headers for muscles related files."""
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
    header = []
    for musc_name in muscles_name:
        header.append(prefix+'_'+musc_name)
    return header

######################################################################
# Data processing related functions

def normalize_direction_data(data, gl, normalize=True, direction=False):
    c=0
    norm_data = np.zeros([data.shape[0],data.shape[1]])
    for key in gl.keys():
        if direction== True:
            norm_data[:,c] = (-1*data[:,c])/gl[key][1]
        else:
            norm_data[:,c] = (data[:,c])/gl[key][1]
        c+=1
    if normalize == False:
        if direction== True:
            reverse_data = -1*data
        else:
            reverse_data = data
        return reverse_data
    else:
        return norm_data

def cmc_revise_data(data, gl):
    c=0
    modified_data = np.zeros([data.shape[0],data.shape[1]])
    for key in gl.keys():
        if gl[key][2] == 'left':
            modified_data[:,c] = -1*data[:,c]
        elif gl[key][2] == 'right':
            modified_data[:,c] = data[:,c]
        else:
            raise Exception('primary leg does not match!')
        c+=1
    return modified_data

def toeoff_pgc(gl, side):
    if side == 'right':
        toeoff = gl.right_toeoff
        strike = gl.right_strike
    elif side == 'left':
        toeoff = gl.left_toeoff
        strike = gl.left_strike
    cycle_duration = (gl.cycle_end - gl.cycle_start)
    while toeoff < strike:
        toeoff += cycle_duration
    while toeoff > strike + cycle_duration:
        toeoff -= cycle_duration
    return pp.percent_duration_single(toeoff,strike,strike + cycle_duration)

def construct_gl_mass_side(subjectno,trialno,loadcond):
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
    side = data["primary_legs"]
    gl = dataman.GaitLandmarks( primary_leg = data['primary_legs'],
                                cycle_start = data['subjects_cycle_start_time'],
                                cycle_end   = data['subjects_cycle_end_time'],
                                left_strike = data['footstrike_left_leg'],
                                left_toeoff = data['toeoff_time_left_leg'],
                                right_strike= data['footstrike_right_leg'],
                                right_toeoff= data['toeoff_time_right_leg'])
    return gl,mass,side

def mean_std_over_subjects(data,ax=1):
    mean = np.nanmean(data,axis=ax)
    std = np.nanstd(data,axis=ax)
    return mean,std

def mean_std_muscles_subjects(data,muscles_num=9):
    mean_data = np.zeros((data.shape[0],muscles_num))
    std_data  = np.zeros((data.shape[0],muscles_num))
    for i in range(muscles_num):
        cols = np.arange(i,data.shape[1],muscles_num)
        mean_data[:,i] = np.nanmean(data[:,cols],axis=1)
        std_data [:,i] = np.nanstd (data[:,cols],axis=1)
    return mean_data,std_data

def toe_off_avg_std(gl_noload,gl_loaded):
    '''This function returns the mean toe off percentage for loaded and noloaded subjects
        parameters:
            gl_noload: a dictionary of noload subjects gait landmark
            gl_loaded: a dictionary of loaded subjects gait landmark
        output:
            np.mean(noload_toe_off),np.std(noload_toe_off),np.mean(loaded_toe_off),np.std(loaded_toe_off)
        '''
    noload_toe_off= np.zeros(21)
    loaded_toe_off= np.zeros(21)
    c0 = 0
    c1 = 0
    for key in gl_noload.keys():
        noload_toe_off[c0] =toeoff_pgc(gl=gl_noload[key][0],side= gl_noload[key][2])
        c0+=1
    for key in gl_loaded.keys():
        loaded_toe_off[c1] =toeoff_pgc(gl=gl_loaded[key][0],side= gl_loaded[key][2])
        c1+=1
    return np.mean(noload_toe_off),np.std(noload_toe_off),np.mean(loaded_toe_off),np.std(loaded_toe_off)

def smooth(a,WSZ):
    """
    a: NumPy 1-D array containing the data to be smoothed
    WSZ: smoothing window size needs, which must be odd number,
    as in the original MATLAB implementation.
    """
    out0 = np.convolve(a,np.ones(WSZ,dtype=int),'valid')/WSZ    
    r = np.arange(1,WSZ-1,2)
    start = np.cumsum(a[:WSZ-1])[::2]/r
    stop = (np.cumsum(a[:-WSZ:-1])[::2]/r)[::-1]
    return np.concatenate((  start , out0, stop  ))

def reduction_calc(data1,data2):
    """ Please assign data according to the formula: (data1-data2)100/data1."""
    reduction = np.zeros(len(data2))
    for i in range(len(data1)):
        reduction[i] = (((data1[i]-data2[i])*100)/data1[i])
    return reduction

######################################################################
# Mass and inertia effect functions

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

######################################################################
# Plot related functions
def beautiful_boxplot(bp):
    ## change outline color, fill color and linewidth of the boxes
    for box in bp['boxes']:
        # change outline color
        box.set( color='#00008b', linewidth=1)
        # change fill color
        box.set( facecolor = '#ffffff' )

    ## change color and linewidth of the whiskers
    for whisker in bp['whiskers']:
        whisker.set(color='#000000', linestyle='--', linewidth=1)

    ## change color and linewidth of the caps
    for cap in bp['caps']:
        cap.set(color='#000000', linewidth=1)

    ## change color and linewidth of the medians
    for median in bp['medians']:
        median.set(color='#ff0000', linewidth=1)

    ## change the style of fliers and their fill
    for flier in bp['fliers']:
        flier.set(marker='o', color='#000000', alpha=0.5)

def no_top_right(ax):
    """box off equivalent in python"""
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')

def plot_shaded_avg(plot_dic,toeoff_color='xkcd:medium grey',toeoff_alpha=1.0,
    lw=2.0,ls='-',alpha=0.35,fill_std=True,fill_lw=0,*args, **kwargs):

    pgc = plot_dic['pgc']
    avg = plot_dic['avg']
    std= plot_dic['std']
    label = plot_dic['label']
    avg_toeoff = plot_dic['avg_toeoff']
    
    #axes setting
    plt.xticks([0,20,40,60,80,100])
    plt.xlim([0,100])
    # plot
    if 'load' in plot_dic:
        load = plot_dic['load']
        if load == 'noload':
            plt.axvline(avg_toeoff, lw=lw, color='xkcd:shamrock green', zorder=0, alpha=toeoff_alpha) #vertical line
    else:
        plt.axvline(avg_toeoff, lw=lw, color=toeoff_color, zorder=0, alpha=toeoff_alpha) #vertical line

    plt.axhline(0, lw=lw, color='grey', zorder=0, alpha=0.75) # horizontal line
    plt.fill_between(pgc, avg + std, avg - std, alpha=alpha,linewidth=fill_lw, *args, **kwargs) # shaded std
    return plt.plot(pgc, avg, *args, lw=lw, ls=ls, label=label, **kwargs) # mean

def plot_muscles_avg(plot_dic,toeoff_color='xkcd:medium grey',
                     toeoff_alpha=1.0,row_num=3,col_num=3,
                     lw=2.0,ls='-',alpha=0.2,fill_lw=0,
                     is_std = False,is_smooth=True,WS=3,fill_std=True,*args, **kwargs):

    pgc = plot_dic['pgc']
    avg = plot_dic['avg']
    std= plot_dic['std']
    label = plot_dic['label']
    avg_toeoff = plot_dic['avg_toeoff']
    muscle_group = plot_dic['muscle_group']
    import Muscles_Group as mgn
    muscles_name = mgn.muscle_group_name[muscle_group]
    # smoothing data
    smooth_avg = np.zeros((avg.shape[0],avg.shape[1]))
    smooth_std = np.zeros((std.shape[0],std.shape[1]))
    if is_smooth == True:
        for i in range(len(muscles_name)):
            smooth_avg[:,i] = smooth(avg[:,i],WSZ=WS)
            smooth_std[:,i] = smooth(std[:,i],WSZ=WS)
    else:
        pass
    avg = smooth_avg
    std = smooth_std
    # plots
    for i in range(len(muscles_name)):
        ax = plt.subplot(row_num,col_num,i+1)
        no_top_right(ax)
        plt.tight_layout()
        plt.title(muscles_name[i])
        plt.xticks([0,20,40,60,80,100])
        plt.xlim([0,100])
        plt.yticks((0,0.2,0.4,0.6,0.8,1))
        ax.axvline(avg_toeoff, lw=lw, color=toeoff_color, zorder=0, alpha=toeoff_alpha) #vertical line
        if is_std == True:
            ax.fill_between(pgc, avg[:,i] + std[:,i], avg[:,i] - std[:,i], alpha=alpha,linewidth=fill_lw, *args, **kwargs) # shaded std
        else:
            pass
        ax.plot(pgc, avg[:,i], *args, lw=lw, ls=ls,label=label,**kwargs) # mean

def plot_joint_muscle_exo (nrows,ncols,plot_dic,color_dic,ylabel,legend_loc=[0,1],thirdplot=True,y_ticks = [-2,-1,0,1,2]):
    '''Note: please note that since it is in the for loop, if some data is
    needed to plot several times it should be repeated in the lists.  '''

    # reading data
    plot_1_list = plot_dic['plot_1_list']
    plot_2_list = plot_dic['plot_2_list']
    color_1_list = color_dic['color_1_list']
    color_2_list = color_dic['color_2_list']
    plot_titles = plot_dic['plot_titles']
    if thirdplot == True:
        color_3_list = color_dic['color_3_list']
        plot_3_list = plot_dic['plot_3_list']
    #plot
    for i in range(nrows*ncols):
        ax = plt.subplot(nrows,ncols,i+1)
        plot_shaded_avg(plot_dic=plot_1_list[i],color=color_1_list[i])
        plot_shaded_avg(plot_dic=plot_2_list[i],color=color_2_list[i])
        if thirdplot == True:
            plot_shaded_avg(plot_dic=plot_3_list[i],color=color_3_list[i])
        plt.yticks(y_ticks)
        plt.title(plot_titles[i])
        no_top_right(ax)
        if i in legend_loc:
            plt.legend(loc='upper right',frameon=False)
        if i in range((nrows*ncols)-nrows,(nrows*ncols)):
            plt.xlabel('gait cycle (%)')
        if i not in np.arange(1,nrows*ncols,ncols):
            plt.ylabel(ylabel)

######################################################################
######################################################################
# Data Processing related functions for Pareto Simulations
def pareto_from_mean_power(power_mean,power_std):
    if power_mean.shape[0] != power_std.shape[0]:
        raise Exception('input dataset do not match!')
    mean_energy = np.zeros(power_mean.shape[1])
    std_energy  = np.zeros( power_std.shape[1])
    gcp = np.linspace(0,100,power_mean.shape[0])
    for i in range(power_mean.shape[1]):
        mean_energy[i] = integrate.simps(np.abs(power_mean[:,i]),gcp)
        std_energy[i]  = np.abs(integrate.simps(np.abs(power_mean[:,i]+power_std[:,i]),gcp) - integrate.simps(np.abs(power_mean[:,i]),gcp))
    return mean_energy, std_energy
def pareto_metabolics_reduction(assist_data,unassist_data,simulation_num=25,subject_num=7):
    reshaped_assisted_data = np.reshape(assist_data,(simulation_num,subject_num),order='F')
    reduction = np.zeros((simulation_num,subject_num))
    c=0
    for i in np.arange(0,21,3):
        reduction[:,c] = np.true_divide((np.ones(25)*unassist_data[i] - reshaped_assisted_data[:,c])*100,np.ones(25)*unassist_data[i])
        c+=1
    return reduction
    
def pareto_avg_std_energy(data,simulation_num=25,subject_num=7,reshape=True):
    if reshape == True:
        reshaped_data = np.reshape(data,(simulation_num,subject_num),order='F')
    else:
        reshaped_data = data
    avg = np.mean(reshaped_data,axis=1)
    std = np.std(reshaped_data,axis=1)
    return avg,std

def pareto_profiles_avg_std(data,gl,simulation_num=25,subject_num=7,change_direction=True):
    avg = np.zeros((data.shape[0],simulation_num))
    std = np.zeros((data.shape[0],simulation_num))
    normal_data = np.zeros((data.shape[0],data.shape[1]))
    c = 0
    subjects = ['05','07','09','10','11','12','14']
    for i in range(subject_num):
        selected_data = data[:,c:c+simulation_num]
        if change_direction == True:
            normal_selected_data = np.true_divide(-1*selected_data,gl['{}_subject{}_trial01'.format('noload',subjects[i])][1])
        else:
            normal_selected_data = np.true_divide(selected_data,gl['{}_subject{}_trial01'.format('noload',subjects[i])][1])
        normal_data[:,c:c+simulation_num] = normal_selected_data
        c+=simulation_num
    c = 0
    for i in range(simulation_num):
        cols = np.arange(i,(simulation_num*subject_num-simulation_num)+1+i,simulation_num)
        selected_data = normal_data[:,cols]
        avg[:,c] = np.nanmean(selected_data,axis=1)
        std[:,c] = np.nanstd(selected_data,axis=1)
        c+=1
    return avg,std

def energy_processed_power(data,gl,simulation_num=25,subject_num=7):
    c = 0
    for i in range(simulation_num):
        cols = np.arange(i,(simulation_num*subject_num-simulation_num)+i,simulation_num)
        selected_data = data[:,cols]
######################################################################
# Plot related functions for Pareto Simulations

def gen_paretocurve_label(H= [70,60,50,40,30],K= [70,60,50,40,30]):
    labels = []
    for i in H:
        for j in K:
            labels.append('H{}K{}'.format(i,j))
    return labels

def label_datapoints(x,y,labels,xytext=(0,0),ha='right',fontsize=10, *args, **kwargs):
    c = 0
    for x,y in zip(x,y):
        plt.annotate(labels[c], (x,y),textcoords="offset points",xytext=xytext,ha=ha,fontsize=fontsize)
        c+=1

def plot_pareto_avg_curve (plot_dic,loadcond,legend_loc=0,labels=None,label_on=True,errbar_on=True,*args, **kwargs):
    x1_data = plot_dic['x1_data']
    x2_data = plot_dic['x2_data']
    y1_data = plot_dic['y1_data']
    y2_data = plot_dic['y2_data']
    x1err_data = plot_dic['x1err_data']
    x2err_data = plot_dic['x2err_data']
    y1err_data = plot_dic['y1err_data']
    y2err_data = plot_dic['y2err_data']
    color_1 = plot_dic['color_1']
    color_2 = plot_dic['color_2']
    # handle labels
    if labels == None:
        labels = np.arange(1,26,1)
    # handle legends
    if 'legend_1' and 'legend_2' not in plot_dic:
        legend_1 = 'biarticular,{}'.format(loadcond)
        legend_2 = 'monoaricular,{}'.format(loadcond)
    else:
        legend_1 = plot_dic['legend_1']
        legend_2 = plot_dic['legend_2']
    # main plot
    plt.scatter(x1_data,y1_data,marker="o",color=color_1,label=legend_1,*args, **kwargs)
    plt.scatter(x2_data,y2_data,marker="v",color=color_2,label=legend_2,*args, **kwargs)
    if errbar_on == True:
        plt.errorbar(x1_data,y1_data,xerr=x1err_data,yerr=y1err_data,fmt='o',ecolor=color_1,alpha=0.15)
        plt.errorbar(x2_data,y2_data,xerr=x2err_data,yerr=y2err_data,fmt='v',ecolor=color_2,alpha=0.15)
    if label_on == True:
        label_datapoints(x1_data,y1_data,labels,*args, **kwargs)
        label_datapoints(x2_data,y2_data,labels,ha='left',*args, **kwargs)
    
def plot_pareto_curve_subjects (nrows,ncols,nplot,plot_dic,loadcond,legend_loc=[0],labels=None,label_on=True,*args, **kwargs):
    x1_data = plot_dic['x1_data']
    x2_data = plot_dic['x2_data']
    y1_data = plot_dic['y1_data']
    y2_data = plot_dic['y2_data']
    color_1 = plot_dic['color_1']
    color_2 = plot_dic['color_2']
    ylabel = plot_dic ['ylabel']
    xlabel = plot_dic ['xlabel']
    # handle labels
    if labels == None:
        labels = labels = np.arange(1,26,1)
    # handle titles
    if 'plot_titles' not in plot_dic:
        subjects = ['05','07','09','10','11','12','14']
        plot_titles = ['subject{},{}'.format(i,loadcond) for i in subjects]
    else:
        plot_titles = plot_dic['plot_titles']
    # handle legends
    if 'legend_1' and 'legend_2' not in plot_dic:
        legend_1 = 'biarticular,{}'.format(loadcond)
        legend_2 = 'monoaricular,{}'.format(loadcond)
    else:
        legend_1 = plot_dic['legend_1']
        legend_2 = plot_dic['legend_2']
    # main plots
    for i in range(nplot):
        ax = plt.subplot(nrows,ncols,i+1)
        plt.scatter(x1_data[:,i],y1_data[:,i],marker="o",color=color_1,label=legend_1,*args, **kwargs)
        plt.scatter(x2_data[:,i],y2_data[:,i],marker="v",color=color_2,label=legend_2,*args, **kwargs)
        if label_on == True:
            label_datapoints(x1_data[:,i],y1_data[:,i],labels,*args, **kwargs)
            label_datapoints(x2_data[:,i],y2_data[:,i],labels,ha='left',*args, **kwargs)
        plt.title(plot_titles[i])
        no_top_right(ax)
        if i in legend_loc:
            plt.legend(loc='best',frameon=False)
        if i in range((nrows*ncols)-nrows,(nrows*ncols)):
            plt.xlabel(xlabel)
        if i in np.arange(0,nrows*ncols,ncols):
            plt.ylabel(ylabel)
        plt.tight_layout()

def plot_pareto_shaded_avg(plot_dic,loadcond,toeoff_color='xkcd:medium grey',toeoff_alpha=1.0,
    lw=2.0,ls='-',alpha=0.35,fill_std=True,plot_toeoff=True,plot_joint=True,fill_lw=0,
    nrows=5,ncols=5,nplots=25,legend_loc=[0],*args, **kwargs):

    pgc = np.linspace(0,100,1000)
    joint_avg = plot_dic['joint_avg']
    joint_std = plot_dic['joint_std']
    avg_1 = plot_dic['avg_1']
    std_1 = plot_dic['std_1']
    avg_2 = plot_dic['avg_2']
    std_2 = plot_dic['std_2']
    color_1 = plot_dic['color_1']
    color_2 = plot_dic['color_2']
    ylabel = plot_dic['ylabel']
    plot_titles = gen_paretocurve_label()
    if 'legend_1' and 'legend_2' not in plot_dic:
        legend_1 = 'biarticular,{}'.format(loadcond)
        legend_2 = 'monoaricular,{}'.format(loadcond)
    else:
        legend_1 = plot_dic['legend_1']
        legend_2 = plot_dic['legend_2']
    #axes setting
    plt.xticks([0,20,40,60,80,100])
    plt.xlim([0,100])
    # plot
    for i in range (nplots):
        plt.subplot(nrows,ncols,i+1)
        plt.axhline(0, lw=lw, color='grey', zorder=0, alpha=0.75) # horizontal line
        #  joint shaded std and mean
        if plot_joint == True:
            plt.fill_between(pgc, joint_avg + joint_std, joint_avg - joint_std,color='k', alpha=0.20,linewidth=fill_lw, *args, **kwargs) 
            plt.plot(pgc, joint_avg, *args, lw=lw, ls=ls, label='joint',color='k', **kwargs)
        if plot_toeoff == True:
            avg_toeoff = plot_dic['avg_toeoff']
            plt.axvline(avg_toeoff, lw=lw, color=toeoff_color, zorder=0, alpha=toeoff_alpha) #vertical line
        # pareto shaded std and mean
        plt.fill_between(pgc, avg_1[:,i] + std_1[:,i], avg_1[:,i] - std_1[:,i], color=color_1, alpha=alpha,linewidth=fill_lw, *args, **kwargs) # shaded std
        plt.plot(pgc, avg_1[:,i], *args, lw=lw, ls=ls, label=legend_1,color=color_1, **kwargs) # mean
        plt.fill_between(pgc, avg_2[:,i] + std_2[:,i], avg_2[:,i] - std_2[:,i], color=color_2, alpha=alpha,linewidth=fill_lw, *args, **kwargs) # shaded std
        plt.plot(pgc, avg_2[:,i], *args, lw=lw, ls=ls, label=legend_2,color=color_2, **kwargs) # mean
        plt.title(plot_titles[i])
        ax = plt.gca()
        no_top_right(ax)
        if i in legend_loc:
            plt.legend(loc='best',frameon=False)
        if i in range((nrows*ncols)-nrows,(nrows*ncols)):
            plt.xlabel('gait cycle (%)')
        if i in np.arange(0,nrows*ncols,ncols):
            plt.ylabel(ylabel)
