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


%% Finding optimum hyperparameters
train_samples_output = samples(1,2:end);
train_samples_input = samples(:,1:end-1);
population_size = 40;                               %classic setting = 10x num of hyperparameters
maxIter = 200;
min_hyperparam = [0; 0; 0; 0.0005];
max_hyperparam = [7; 7; 7; 7];
F_weight = 0.8;                                     %classic setting = 0.8
CR = 0.9;                                           %classic setting = 0.9

opt_hyp_param = RunDiffEvolutionOpt(train_samples_output,train_samples_input, population_size, maxIter, min_hyperparam, max_hyperparam, F_weight, CR);



%% Getting kernel
 w = [0.2948; 0.1323];
    v1 = 6.2618;
    v0 = 0.0045;
%opt_hyp_param = [w; v1; v0];
K = CovFunc(samples(:,1:end-1), samples(:,1:end-1), opt_hyp_param);
log_likelihood_K = LogLikelihood(K, samples(1,2:end)) 

%% Prediction

K_inv = pinv(K);

dt = 0.5;
sim_time = 0:dt:total_time;
sim_y = zeros(length(sim_time),1);
sigma_pred = zeros(length(sim_time),1);

u = umin+rand()*(umax-umin);

y = zeros(length(sim_time),1);

for i = 1:length(sim_time)-1
    
    if mod(i,round(control_inpute_rate/dt)) == 0
        u = umin+rand()*(umax-umin);
    end    
    
    k_vec =  CovFunc(samples(:,1:end-1), [sim_y(i);u], opt_hyp_param);
    
    sim_y(i+1) = k_vec' * (K_inv * samples(1,2:end)');
    
    k = CovFunc([sim_y(i);u],[sim_y(i);u], opt_hyp_param);
    
    sigma_pred(i+1) = k - k_vec' * K_inv * k_vec;
    
    y(i+1) = dynamics(y(i),u,dt);
    
end

%% Plots
close all
hold on;
plot(sim_time,y);
plot(sim_time,sim_y,'-.');
plot(sim_time,sim_y+2*sqrt(sigma_pred),'r',sim_time,sim_y-2*sqrt(sigma_pred),'r');
legend('True trajectory', 'Predicted trajectory', 'Samples used for training', 'One sigma error bound');
