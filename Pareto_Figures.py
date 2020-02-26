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
import Functions as fcns
#####################################################################################
#####################################################################################
"""Section 01:
                As the first step of processing data, we need to extract unassised subjects
                and ideal exoskeletons simulations data and confirm modeling and simulations"""
#####################################################################################
#####################################################################################
"""Section 02:
                This section has been dedicated to extracting pareto simulations data for
                loaded and noload walking. The main extracted data are actuators energy,
                metabolic energy, torque profile, power profile.
"""
# Biarticular/Noload
biarticular_noload_hipactuator_torque,biarticular_noload_kneeactuator_torque,\
biarticular_noload_hipactuator_power,biarticular_noload_kneeactuator_power,\
biarticular_noload_hipactuator_energy,biarticular_noload_kneeactuator_energy,\
biarticular_noload_metabolicenergy =fcns.pareto_data_subjects(configuration='Biarticular',loadcond='noload')

# Monoarticular/Noload
monoarticular_noload_hipactuator_torque,monoarticular_noload_kneeactuator_torque,\
monoarticular_noload_hipactuator_power,monoarticular_noload_kneeactuator_power,\
monoarticular_noload_hipactuator_energy,monoarticular_noload_kneeactuator_energy,\
monoarticular_noload_metabolicenergy =fcns.pareto_data_subjects(configuration='Monoarticular',loadcond='noload')

# Biarticular/Loaded
biarticular_loaded_hipactuator_torque,biarticular_loaded_kneeactuator_torque,\
biarticular_loaded_hipactuator_power,biarticular_loaded_kneeactuator_power,\
biarticular_loaded_hipactuator_energy,biarticular_loaded_kneeactuator_energy,\
biarticular_loaded_metabolicenergy =fcns.pareto_data_subjects(configuration='Biarticular',loadcond='loaded')

# Monoarticular/Loaded
monoarticular_loaded_hipactuator_torque,monoarticular_loaded_kneeactuator_torque,\
monoarticular_loaded_hipactuator_power,monoarticular_loaded_kneeactuator_power,\
monoarticular_loaded_hipactuator_energy,monoarticular_loaded_kneeactuator_energy,\
monoarticular_loaded_metabolicenergy =fcns.pareto_data_subjects(configuration='Monoarticular',loadcond='loaded')

#####################################################################################
