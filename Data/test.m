clc;clear
path = 'F:\HMI\Exoskeleton\OpenSim\Walking_Mass_Inertia_Effect\Data\Data\pareto\';
hip_torque = csvread(strcat(path,'biarticular_pareto_load_hipactuator_power.csv'));
hip_torque = hip_torque(:,250);
left_torque = circshift(hip_torque,600);
gait_cycle = linspace(0,100,1000);
figure()
plot(gait_cycle,left_torque,'ko')
hold on
plot(gait_cycle,hip_torque,'ro')
hold on
plot([0,100],[0,0],'k')
hold on
plot([60,60],[min(hip_torque),max(hip_torque)],'g')

