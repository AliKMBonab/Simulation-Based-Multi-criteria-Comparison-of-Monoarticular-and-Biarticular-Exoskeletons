clc; clear; close all;
gcp = linspace(0,100,1000);
set(0,'defaultAxesFontSize',14)
%% loading data
path = 'D:\Ali.K.M.Bonab\Walking_Mass_Inertia_Effect\Data\' ;
noloadhipjointpower = csvread(strcat(path,'Data\RRA\noload_hipjoint_power.csv'));
noloadkneejointpower = csvread(strcat(path,'Data\RRA\noload_kneejoint_power.csv'));
loadedhipjointpower = csvread(strcat(path,'Data\RRA\loaded_hipjoint_power.csv'));
loadedkneejointpower = csvread(strcat(path,'Data\RRA\loaded_kneejoint_power.csv'));
noloadhipjointtorque = csvread(strcat(path,'Data\RRA\noload_hipjoint_torque.csv'));
noloadkneejointtorque = csvread(strcat(path,'Data\RRA\noload_kneejoint_torque.csv'));
loadedhipjointtorque = csvread(strcat(path,'Data\RRA\loaded_hipjoint_torque.csv'));
loadedkneejointtorque = csvread(strcat(path,'Data\RRA\loaded_kneejoint_torque.csv'));

%% figures
figure('Name','Subject 05 vs 11')
subplot(2,2,1)
plot(gcp,noloadhipjointpower(:,1),'LineWidth',2,'Color','r')
hold on
plot(gcp,noloadhipjointpower(:,13),'LineWidth',2,'Color','b')
title('Subject 05 vs 11-Hip Joint Power')
xlabel('gait cycle')
ylabel('power')
box off
subplot(2,2,2)
plot(gcp,noloadkneejointpower(:,1),'LineWidth',2,'Color','r')
hold on
plot(gcp,noloadkneejointpower(:,13),'LineWidth',2,'Color','b')
title('Subject 05 vs 11-Knee Joint Power')
xlabel('gait cycle')
ylabel('power')
box off
subplot(2,2,3)
plot(gcp,-noloadhipjointtorque(:,1),'LineWidth',2,'Color','r')
hold on
plot(gcp,-noloadhipjointtorque(:,13),'LineWidth',2,'Color','b')
title('Subject 05 vs 11-Hip Joint Torque')
xlabel('gait cycle')
ylabel('torque')
box off
subplot(2,2,4)
plot(gcp,-noloadkneejointtorque(:,1),'LineWidth',2,'Color','r')
hold on
plot(gcp,-noloadkneejointtorque(:,13),'LineWidth',2,'Color','b')
title('Subject 05 vs 11-Knee Joint Torque')
xlabel('gait cycle')
ylabel('torque')
legend('subject 05','subject 11')
box off


figure('Name','Subject 05 vs 10')
subplot(2,2,1)
plot(gcp,noloadhipjointpower(:,1),'LineWidth',2,'Color','r')
hold on
plot(gcp,noloadhipjointpower(:,10),'LineWidth',2,'Color','g')
title('Subject 05 vs 10-Hip Joint Power')
xlabel('gait cycle')
ylabel('power')
box off
subplot(2,2,2)
plot(gcp,noloadkneejointpower(:,1),'LineWidth',2,'Color','r')
hold on
plot(gcp,noloadkneejointpower(:,10),'LineWidth',2,'Color','g')
title('Subject 05 vs 10-Knee Joint Power')
xlabel('gait cycle')
ylabel('power')
box off
subplot(2,2,3)
plot(gcp,-noloadhipjointtorque(:,1),'LineWidth',2,'Color','r')
hold on
plot(gcp,-noloadhipjointtorque(:,10),'LineWidth',2,'Color','g')
title('Subject 05 vs 10-Hip Joint Torque')
xlabel('gait cycle')
ylabel('torque')
box off
subplot(2,2,4)
plot(gcp,-noloadkneejointtorque(:,1),'LineWidth',2,'Color','r')
hold on
plot(gcp,-noloadkneejointtorque(:,10),'LineWidth',2,'Color','g')
title('Subject 05 vs 10-Knee Joint Torque')
xlabel('gait cycle')
ylabel('torque')
legend('subject 05','subject 10')
box off