clear
clc
sample  = csvread('D:\Ali.K.M.Bonab\Walking_Mass_Inertia_Effect\Data\Pareto_Related_Files\sample_data.csv');

[f, idxs] = ParetoFront(sample)
