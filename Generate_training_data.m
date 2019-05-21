close all
clc
clear

%% Simulation Parameters
dt = 0.001;
total_time = 100;
t = 1:dt:total_time;

%% System Setup
%state variable
y = zeros(length(t),1);
sample_rate = 0.5;
sample_time = [];
samples = [];

%process noise covariance 
Q = 0.0025;

%control
u = zeros(length(t),1);
umin = -1.3;
umax = 1.3;
control_inpute_rate = 5;

%% Simulating the system
for i = 1:length(t)-1
    
    %Input at input_rate
    if mod(i,round(control_inpute_rate/dt)) == 1
        u(i) = umin+rand()*(umax-umin);
    else
        u(i) = u(i-1);
    end
    
    y(i+1) = dynamics(y(i),u(i),dt);
    
    if mod(i,round(sample_rate/dt)) == 0
        sample_time = [sample_time t(i)];
        samples = [samples [y(i)+sqrt(Q)*randn;u(i)]];
    end   
    
end

save('Training_data.mat','samples')