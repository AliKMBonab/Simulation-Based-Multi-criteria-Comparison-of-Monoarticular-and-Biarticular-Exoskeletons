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


def mean_std_over_subjects(data):
    mean = np.nanmean(data,axis=1)
    std = np.nanstd(data,axis=1)
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
# Plot related functions
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
        