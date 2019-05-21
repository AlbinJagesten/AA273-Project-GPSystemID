close all
clear
clc

%% Finding optimum hyperparameters

load('Training_data.mat')

train_samples_output = samples(1,2:end);
train_samples_input = samples(:,1:end-1);
population_size = 40;                               %classic setting = 10x num of hyperparameters
maxIter = 100;
min_hyperparam = [0; 0; 0; 0.0005];
max_hyperparam = [7; 7; 7; 7];
F_weight = 0.8;                                     %classic setting = 0.8
CR = 0.9;                                           %classic setting = 0.9

opt_hyp_param = RunDiffEvolutionOpt(train_samples_output,train_samples_input, population_size, maxIter, min_hyperparam, max_hyperparam, F_weight, CR);

min_hyperparam = opt_hyp_param - 0.2 * opt_hyp_param;
max_hyperparam = opt_hyp_param + 0.2 * opt_hyp_param;

opt_hyp_param = RunDiffEvolutionOpt(train_samples_output,train_samples_input, population_size, maxIter, min_hyperparam, max_hyperparam, F_weight, CR);

%% Getting kernel
% w = [0.2948; 0.1323];
%      v1 = 6.2618;
%      v0 = 0.0045;
% opt_hyp_param = [w; v1; v0];

K = CovFunc(samples(:,1:end-1), samples(:,1:end-1), opt_hyp_param);
log_likelihood_K = LogLikelihood(K, samples(1,2:end)) 

%% Prediction

K_inv = pinv(K);

load('Validation_data.mat')

dt = 0.5;
sim_time = 0:dt:total_time;
sim_y = zeros(length(sim_time),1);
sigma_pred = zeros(length(sim_time),1);
y = zeros(length(sim_time),1);

for i = 1:length(sim_time)-1  
    
    u = control_input(i);
    
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
