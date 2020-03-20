clc;clear;close all;

%% Loading Data
path = 'D:\Ali.K.M.Bonab\Walking_Mass_Inertia_Effect\Data\Data\' ;
% noload-biarticular
biarticular_noload_hip_energy  = csvread(strcat(path,'Pareto\biarticular_pareto_noload_hipactuator_energy.csv'));
biarticular_noload_knee_energy = csvread(strcat(path,'Pareto\biarticular_pareto_noload_kneeactuator_energy.csv'));
biarticular_noload_exo_energy = biarticular_noload_hip_energy + biarticular_noload_knee_energy;
biarticular_noload_metabolic_energy = csvread(strcat(path,'Pareto\biarticular_pareto_noload_metabolics_energy.csv'));
% loaded-biarticular
biarticular_loaded_hip_energy  = csvread(strcat(path,'Pareto\biarticular_pareto_load_hipactuator_energy.csv'));
biarticular_loaded_knee_energy = csvread(strcat(path,'Pareto\biarticular_pareto_load_kneeactuator_energy.csv'));
biarticular_loaded_exo_energy = biarticular_loaded_hip_energy + biarticular_loaded_knee_energy;
biarticular_loaded_metabolic_energy = csvread(strcat(path,'Pareto\biarticular_pareto_load_metabolics_energy.csv'));
% noload-monoarticular
monoarticular_noload_hip_energy  = csvread(strcat(path,'Pareto\monoarticular_pareto_noload_hipactuator_energy.csv'));
monoarticular_noload_knee_energy = csvread(strcat(path,'Pareto\monoarticular_pareto_noload_kneeactuator_energy.csv'));
monoarticular_noload_exo_energy = monoarticular_noload_hip_energy + monoarticular_noload_knee_energy;
monoarticular_noload_metabolic_energy = csvread(strcat(path,'Pareto\monoarticular_pareto_noload_metabolics_energy.csv'));
% loaded-monoarticular
monoarticular_loaded_hip_energy  = csvread(strcat(path,'Pareto\monoarticular_pareto_load_hipactuator_energy.csv'));
monoarticular_loaded_knee_energy = csvread(strcat(path,'Pareto\monoarticular_pareto_load_kneeactuator_energy.csv'));
monoarticular_loaded_exo_energy = monoarticular_loaded_hip_energy + monoarticular_loaded_knee_energy;
monoarticular_loaded_metabolic_energy = csvread(strcat(path,'Pareto\monoarticular_pareto_load_metabolics_energy.csv'));
% unassist metabolic
unassisted_noload_metabolic_energy = csvread(strcat(path,'Unassist\noload_metabolics_energy.csv'));
unassisted_loaded_metabolic_energy = csvread(strcat(path,'Unassist\loaded_metabolics_energy.csv'));
idx = [1;4;7;10;13;16;19];
%% Post Processing 
% noload/biarticular
bi_noload_exo_energy = reshape(biarticular_noload_exo_energy,[25,7]);
bi_noload_metabolic_energy = reshape(biarticular_noload_metabolic_energy,[25,7]);
% noload/monoarticular
mono_noload_exo_energy = reshape(monoarticular_loaded_exo_energy,[25,7]);
mono_noload_metabolic_energy = reshape(monoarticular_noload_metabolic_energy,[25,7]);
% loaded/biarticular
bi_loaded_exo_energy = reshape(biarticular_loaded_exo_energy,[25,7]);
bi_loaded_metabolic_energy = reshape(biarticular_loaded_metabolic_energy,[25,7]);
% loaded/monoarticular
mono_loaded_exo_energy = reshape(monoarticular_loaded_exo_energy,[25,7]);
mono_loaded_metabolic_energy = reshape(monoarticular_loaded_metabolic_energy,[25,7]);
% unassist
unassist_noload_metabolic_energy = unassisted_noload_metabolic_energy(idx);
unassist_loaded_metabolic_energy = unassisted_loaded_metabolic_energy(idx);

%% Pareto Filtering
for i = 1:1:7
    % prepare data for paretofront function
    noload_bi_data = [bi_noload_metabolic_energy(:,i),bi_noload_exo_energy(:,i)];
    loaded_bi_data = [bi_loaded_metabolic_energy(:,i),bi_loaded_exo_energy(:,i)];
    noload_mono_data = [mono_noload_metabolic_energy(:,i),mono_noload_exo_energy(:,i)];
    loaded_mono_data = [mono_loaded_metabolic_energy(:,i),mono_loaded_exo_energy(:,i)];
    % filter the paretocurve data
    [data,~] = ParetoFront(noload_bi_data);
    bi_noload_paretofront_exo_energy{i} = data(:,2);
    bi_noload_paretofront_metabolic_energy{i} = data(:,1);
    bi_noload_paretofront_metabolic_percent{i} = ((unassist_noload_metabolic_energy(i)*ones(size(data(:,1)))-data(:,1))*100)./unassist_noload_metabolic_energy(i);
    [data,~] = ParetoFront(loaded_bi_data);
    bi_loaded_paretofront_exo_energy{i} = data(:,2);
    bi_loaded_paretofront_metabolic_energy{i} = data(:,1);
    bi_loaded_paretofront_metabolic_percent{i} = ((unassist_loaded_metabolic_energy(i)*ones(size(data(:,1)))-data(:,1))*100)./unassist_noload_metabolic_energy(i);
    [data,~] = ParetoFront(noload_mono_data);
    mono_noload_paretofront_exo_energy{i} = data(:,2);
    mono_noload_paretofront_metabolic_energy{i} = data(:,1);
    mono_noload_paretofront_metabolic_percent{i} = ((unassist_noload_metabolic_energy(i)*ones(size(data(:,1)))-data(:,1))*100)./unassist_noload_metabolic_energy(i);
    [data,~] = ParetoFront(loaded_mono_data);
    mono_loaded_paretofront_exo_energy{i} = data(:,2);
    mono_loaded_paretofront_metabolic_energy{i} = data(:,1);
    mono_loaded_paretofront_metabolic_energy{i} = ((unassist_loaded_metabolic_energy(i)*ones(size(data(:,1)))-data(:,1))*100)./unassist_noload_metabolic_energy(i);
end

%% Convert Metabolic Energy to Reduction Percent
for i=1:1:7
    bi_noload_metabolic_percent(:,i) = ((unassist_noload_metabolic_energy(i)*ones(25,1)-bi_noload_metabolic_energy(:,i))*100)./unassist_noload_metabolic_energy(i);
    bi_loaded_metabolic_percent(:,i) = ((unassist_loaded_metabolic_energy(i)*ones(25,1)-bi_loaded_metabolic_energy(:,i))*100)./unassist_loaded_metabolic_energy(i);
    mono_noload_metabolic_percent(:,i) = ((unassist_noload_metabolic_energy(i)*ones(25,1)-mono_noload_metabolic_energy(:,i))*100)./unassist_noload_metabolic_energy(i);
    mono_loaded_metabolic_percent(:,i) = ((unassist_loaded_metabolic_energy(i)*ones(25,1)-mono_loaded_metabolic_energy(:,i))*100)./unassist_loaded_metabolic_energy(i);
end
%% Pareto Curve and Pareto Front Plots
figure('Name','Paretofront_noload')
for i=1:1:7
    subplot(3,3,i)
    plot(bi_noload_paretofront_metabolic_percent{i},bi_noload_paretofront_exo_energy{i},'o','Color','r')
    hold on
    plot(bi_noload_metabolic_percent(:,i),bi_noload_exo_energy(:,i),'*','Color','r')
    hold on
    plot(mono_noload_paretofront_metabolic_percent{i},mono_noload_paretofront_exo_energy{i},'o','Color','b')
    hold on
    plot(mono_noload_metabolic_percent(:,i),mono_noload_exo_energy(:,i),'*','Color','b')
    if ismember(i,[1,4,7])==1
        ylabel('exo energy')
    end
    if ismember(i,[7,5,6])==1
        xlabel('metabolic energy')
    end
    if ismember(i,[1])==1
        legend('biarticular','monoarticular')
    end
    box off
end

%% Pareto Front Plots
figure('Name','Paretofront_noload')
for i=1:1:7
    subplot(3,3,i)
    plot(bi_noload_paretofront_metabolic_percent{i},bi_noload_paretofront_exo_energy{i},'o','Color','r')
    hold on
    plot(bi_noload_metabolic_percent(:,i),bi_noload_exo_energy(:,i),'*','Color','r')
    hold on
    plot(mono_noload_paretofront_metabolic_percent{i},mono_noload_paretofront_exo_energy{i},'o','Color','b')
    hold on
    plot(mono_noload_metabolic_percent(:,i),mono_noload_exo_energy(:,i),'*','Color','b')
    if ismember(i,[1,4,7])==1
        ylabel('exo energy')
    end
    if ismember(i,[7,5,6])==1
        xlabel('metabolic energy')
    end
    if ismember(i,[1])==1
        legend('biarticular','monoarticular')
    end
    box off
end
