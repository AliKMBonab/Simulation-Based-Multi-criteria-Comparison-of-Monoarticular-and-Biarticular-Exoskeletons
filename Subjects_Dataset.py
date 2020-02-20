import numpy as np
"""
-This is a dataset that has been established to be used in another files to process data.
"""
noload_subjects_cycle_start_time = np.array([0.462,0.416,0.379,0.356,0.377,0.431,0.245,0.302,0.407,0.311,0.606,0.652,0.286,0.517,0.511,0.248,0.273,0.551,0.630,0.444,0.474]),
loaded_subjects_cycle_start_time = np.array([0.277,0.25,0.408,0.307,0.277,0.288,0.199,0.233,0.367,1.056,1.254,0.952,0.191,0.470,0.365,0.230,0.130,0.365,0.271,0.581,0.581])

noload_subjects_cycle_end_time = np.array([1.614,1.578,1.552,1.512,1.5,1.56,1.471,1.506,1.659,1.385,1.67,1.7,1.429,1.645,1.639,1.402,1.416,1.707,1.707,1.526,1.571])
noload_subjects_cycle_end_time = np.array([1.475,1.427,1.52,1.595,1.560,1.528,1.562,1.541,1.669,2.103,2.315,2.002,1.315,1.607,1.516,1.510,1.438,1.683,1.422,1.691,1.772])

noload_footstrike_right_leg = np.array([0.462,0.416,0.379,1.512,0.377,0.431,1.471,1.506,1.659,0.840,1.127,1.175,1.429,1.645,1.639,1.402,0.273,0.551,0.630,0.444,0.474])
loaded_footstrike_right_leg = np.array([1.475,1.427,1.520,1.595,1.560,1.528,1.562,1.541,1.669,0.909,1.625,1.329,1.315,1.607,1.516,1.510,1.438,1.683,0.849,1.123,1.185])
                        
noload_toeoff_time_right_leg = np.array([1.179,1.156,1.107,1.071,1.091,1.137,1.033,1.069,1.198,1.521,1.802,1.815,1.028,1.244,1.249,0.951,0.981,1.255,1.286,1.105,1.142])
loaded_toeoff_time_right_leg = np.array([1.089,1.015,1.133,1.135,1.108,1.087,1.111,1.111,1.222,1.569,1.254,0.952,0.938,1.232,1.148,1.047,0.973,1.212,1.579,1.853,1.931])

noload_footstrike_left_leg = np.array([1.048,1.018,0.969,0.935,0.965,1.012,0.867,0.896,1.035,0.311,0.606,0.652,0.855,1.073,1.079,0.817,0.860,1.128,1.167,0.993,1.021])
loaded_footstrike_left_leg = np.array([0.891,0.835,0.957,0.953,0.928,0.911,0.884,0.891,1.010,1.431,1.097,0.798,0.746,1.044,0.944,0.874,0.782,1.022,1.422,1.691,1.772])
                      
noload_toeoff_time_left_leg = np.array([1.764,1.719,1.702,1.643,1.662,1.701,1.623,1.667,1.813,0.954,1.242,1.286,1.586,1.798,1.790,1.524,1.550,1.845,1.831,1.658,1.700])
loaded_toeoff_time_left_leg = np.array([1.653,1.602,1.696,1.778,1.748,1.696,1.774,1.734,1.866,1.056,1.777,1.480,1.494,1.794,1.7,1.625,1.618,1.877,1.012,1.274,1.347]) 

noload_primary_legs = ['right','right','right','right','right','right','right','right','right','left','left','left','right','right','right','right','right','right','right','right','right']
loaded_primary_legs = ['right','right','right','right','right','right','right','right','right','left','right','right','right','right','right','right','right','right','left','left','left']

noload_secondary_legs = ['left','left','left','left','left','left','left','left','left','right','right','right','left','left','left','left','left','left','left','left','left']
noload_secondary_legs = ['left','left','left','left','left','left','left','left','left','right','left','left','left','left','left','left','left','left','right','right','right']

noload_subjects_mass = np.array([112.43,112.43,112.43,89.23,89.23,89.23,86.90,86.90,86.90,64.03,64.03,64.03,85.06,85.06,85.06,67.14,67.14,67.14,83.79,83.79,83.79])
loaded_subjects_mass = np.array([149.86,149.86,149.86,126.94,126.94,126.94,125.09,125.09,125.09,102.03,102.03,102.03,122.81,122.81,122.81,105.23,105.23,105.23,122.52,122.52,122.52])