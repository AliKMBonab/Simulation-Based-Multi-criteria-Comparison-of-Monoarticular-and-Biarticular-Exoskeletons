clc; clear; close all;
dataset = csvread('D:\Ali.K.M.Bonab\Walking_Mass_Inertia_Effect\Data\Data\Unassist\unassist_loaded_reaction_moments.csv');
joint = input("Jont: ['back','duct_tape','hip','knee','patellofemoral','ankle']:    ",'s');
joint = convertCharsToStrings(joint);
if joint == 'back'
    c = 1;
elseif joint == 'duct_tape'
    c = 64;
elseif joint == 'hip'
    c = 128;
elseif joint == 'knee'
    c = 192;
elseif joint == 'patellofemoral'
    c = 256;
elseif joint == 'ankle'
    c = 320;
end
  
force_1 = zeros(1000,21);
force_2 = zeros(1000,21);
force_3 = zeros(1000,21);
gpc = repmat(transpose(linspace(0,100,1000)),1,21);
for i= 1:1:21
    if mod(i,3)== 0
        force_1(:,i) = -dataset(:,c);
        force_2(:,i) = -dataset(:,c+1);
        force_3(:,i) = -dataset(:,c+2);
        c=c+1;
    else
        force_1(:,i) = dataset(:,c);
        force_2(:,i) = dataset(:,c+1);
        force_3(:,i) = dataset(:,c+2);
        c=c+1;
    end
end
figure('Name','test')
subplot(3,1,1)
plot(gpc,force_1)
subplot(3,1,2)
plot(gpc,force_2)
subplot(3,1,3)
plot(gpc,force_3)

figure('Name','test 2')
subjects = {'05','07','09','10','11','12','14'};
s=1;
t=1;
for i=1:1:21
    subplot(7,3,i)
    plot(gpc(:,1),force_3(:,i),'*')
    title(sprintf('S%s-T%d',subjects{s},t));
    t=t+1;
    if mod(i,3)==0
        s=s+1;
        t=1;
    end
end
mean_force_3 = nanmean(force_3,2);
std_force_3 = nanmean(force_3,2);
figure('Name','avg')
plot(gpc(:,1),mean_force_3,'LineWidth',3,'Color','b')
fill([gpc(:,1); gpc(end:-1:1,1)], [mean_force_3 + std_force_3; ...
        mean_force_3(end:-1:1) - std_force_3(end:-1:1)], ...
        'b', 'edgecolor','b','facealpha',0.1, 'edgealpha',0.1)
