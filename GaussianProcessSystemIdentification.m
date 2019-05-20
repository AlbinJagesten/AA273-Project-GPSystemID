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

%% Getting kernel
K = CovFunc(samples(:,1:end-1),samples(:,1:end-1));
noisy_K_inv = pinv(K + 0.0001*(eye(size(K)))); 

%% Prediction

sim_time = 0:0.5:total_time;
sim_y = zeros(length(sim_time),1);
sim_y(1) = y(1);
sigma_pred = zeros(length(sim_time),1);

v0 = 0.0045;

for i = 1:length(sim_time)-1
    
    [~,idx] = min(abs(sample_time-sim_time(i)));
    
    u = samples(2,idx);
    
    k_vec =  CovFunc(samples(:,1:end-1), [sim_y(i);u]);
    
    sim_y(i+1) = k_vec' * (noisy_K_inv * samples(1,2:end)');
    
    k = CovFunc([sim_y(i);u],[sim_y(i);u]);
    
    sigma_pred(i) = k - k_vec' * noisy_K_inv * k_vec+v0;
    
end

%% Plots
hold on;
plot(t,y);
plot(sim_time,sim_y,'-.');
plot(sample_time,samples(1,:),'k.');
plot(sim_time,sim_y+sqrt(sigma_pred),'r',sim_time,sim_y-sqrt(sigma_pred),'r');
legend('Noiseless True trajectory', 'Predicted trajectory', 'Samples used for training', 'One sigma error bound');
