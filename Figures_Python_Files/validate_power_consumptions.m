clc; clear; close all;
gcp = linspace(0,1,1000);
%% Biarticular
% Torque profiles
bi_hip_torque = readmatrix('../Data/Ideal/biarticular_ideal_noload_hipactuator_torque.csv');
bi_knee_torque = readmatrix('../Data/Ideal/biarticular_ideal_noload_kneeactuator_torque.csv');
bi_hip_torque_avg = mean(bi_hip_torque,2);
bi_hip_torque_std = std(bi_hip_torque,0,2);
bi_knee_torque_avg = mean(bi_knee_torque,2);
bi_knee_torque_std = std(bi_knee_torque,0,2);

% Velocity profiles
bi_hip_velocity = readmatrix('../Data/Ideal/biarticular_ideal_noload_hipactuator_speed.csv');
bi_knee_velocity = readmatrix('../Data/Ideal/biarticular_ideal_noload_kneeactuator_speed.csv');
bi_hip_velocity_avg = mean(bi_hip_velocity,2);
bi_hip_velocity_std = std(bi_hip_velocity,0,2);
bi_knee_velocity_avg = mean(bi_knee_velocity,2);
bi_knee_velocity_std = std(bi_knee_velocity,0,2);

% Power profiles
bi_hip_power = readmatrix('../Data/Ideal/biarticular_ideal_noload_hipactuator_power.csv');
bi_knee_power = readmatrix('../Data/Ideal/biarticular_ideal_noload_kneeactuator_power.csv');
bi_hip_power_avg = mean(bi_hip_power,2);
bi_hip_power_std = std(bi_hip_power,0,2);
bi_knee_power_avg = mean(bi_knee_power,2);
bi_knee_power_std = std(bi_knee_power,0,2);

%% Monoarticular
% Torque profiles
mono_hip_torque = readmatrix('../Data/Ideal/monoarticular_ideal_noload_hipactuator_torque.csv');
mono_knee_torque = readmatrix('../Data/Ideal/monoarticular_ideal_noload_kneeactuator_torque.csv');
mono_hip_torque_avg = mean(mono_hip_torque,2);
mono_hip_torque_std = std(mono_hip_torque,0,2);
mono_knee_torque_avg = mean(mono_knee_torque,2);
mono_knee_torque_std = std(mono_knee_torque,0,2);

% Velocity profiles
mono_hip_velocity = readmatrix('../Data/Ideal/monoarticular_ideal_noload_hipactuator_speed.csv');
mono_knee_velocity = readmatrix('../Data/Ideal/monoarticular_ideal_noload_kneeactuator_speed.csv');
mono_hip_velocity_avg = mean(mono_hip_velocity,2);
mono_hip_velocity_std = std(mono_hip_velocity,0,2);
mono_knee_velocity_avg = mean(mono_knee_velocity,2);
mono_knee_velocity_std = std(mono_knee_velocity,0,2);

% Power profiles
mono_hip_power = readmatrix('../Data/Ideal/monoarticular_ideal_noload_hipactuator_power.csv');
mono_knee_power = readmatrix('../Data/Ideal/monoarticular_ideal_noload_kneeactuator_power.csv');
mono_hip_power_avg = mean(mono_hip_power,2);
mono_hip_power_std = std(mono_hip_power,0,2);
mono_knee_power_avg = mean(mono_knee_power,2);
mono_knee_power_std = std(mono_knee_power,0,2);

%% Power profiles calculated based on the linear mapping
% biarticular
bi_hip_power_calculated = (mono_hip_torque-mono_knee_torque).*bi_hip_velocity;
bi_knee_power_calculated = bi_knee_torque.*bi_knee_velocity;
bi_hip_power_calculated_avg = mean(bi_hip_power_calculated,2);
bi_hip_power_calculated_std = std(bi_hip_power_calculated,0,2);
bi_knee_power_calculated_avg = mean(bi_knee_power_calculated,2);
bi_knee_power_calculated_std = std(bi_knee_power_calculated,0,2);

% monoarticular
mono_hip_power_calculated = mono_hip_torque.*mono_hip_velocity;
mono_knee_power_calculated = mono_knee_torque.*(bi_knee_velocity-bi_hip_velocity);
mono_hip_power_calculated_avg = mean(mono_hip_power_calculated,2);
mono_hip_power_calculated_std = std(mono_hip_power_calculated,0,2);
mono_knee_power_calculated_avg = mean(mono_knee_power_calculated,2);
mono_knee_power_calculated_std = std(mono_knee_power_calculated,0,2);

%% Energy Consumption
% biarticular
bi_hip_energy_consumption = trapz(gcp',max(0,bi_hip_power),1) + abs(trapz(gcp',min(0,bi_hip_power),1));
bi_knee_energy_consumption = trapz(gcp',max(0,bi_knee_power),1) + abs(trapz(gcp',min(0,bi_knee_power),1));

% monoarticular
mono_hip_energy_consumption = trapz(gcp',max(0,mono_hip_power),1) + abs(trapz(gcp',min(0,mono_hip_power),1));
mono_knee_energy_consumption = trapz(gcp',max(0,mono_knee_power),1) + abs(trapz(gcp',min(0,mono_knee_power),1));

%% Comparison Figures
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Power comparison
figure('Name','power comparisons')
subplot(221)
plot(gcp,bi_hip_power_avg,'LineWidth',2,'Color','r')
hold on
plot(gcp,bi_hip_power_calculated_avg,'LineWidth',2,'Color','b','LineStyle','--')
title('biarticular hip')
subplot(222)
plot(gcp,bi_knee_power_avg,'LineWidth',2,'Color','r')
hold on
plot(gcp,bi_knee_power_calculated_avg,'LineWidth',2,'Color','b','LineStyle','--')
title('biarticular knee')
subplot(223)
plot(gcp,mono_hip_power_avg,'LineWidth',2,'Color','r')
hold on
plot(gcp,mono_hip_power_calculated_avg,'LineWidth',2,'Color','b','LineStyle','--')
title('monoarticular hip')
subplot(224)
plot(gcp,mono_knee_power_avg,'LineWidth',2,'Color','r')
hold on
plot(gcp,mono_knee_power_calculated_avg,'LineWidth',2,'Color','b','LineStyle','--')
title('monoarticular knee')
legend('main profile','calculated profile')
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Energy consumption
figure('Name','Energy conumption comparisons')
subplot(131)
boxplot([bi_hip_energy_consumption',mono_hip_energy_consumption'],'Notch','off','Labels',{'bi','mono'})
title('hip energy consumption')
subplot(132)
boxplot([bi_knee_energy_consumption',mono_knee_energy_consumption'],'Notch','off','Labels',{'bi','mono'})
title('knee energy consumption')
subplot(133)
boxplot([bi_hip_energy_consumption'+bi_knee_energy_consumption',mono_hip_energy_consumption'+mono_knee_energy_consumption'],...
        'Notch','off','Labels',{'bi','mono'})
title('hip energy consumption')

