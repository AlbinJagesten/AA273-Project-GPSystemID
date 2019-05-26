close all
clc
clear

Q = 0.0025;
sample_rate = 0.5;
control_inpute_rate = 5;
control_imit = 1.3;
control_length = 21;
control_sequence = 2*control_imit*rand(control_length,1)-control_imit;

[samples sample_time] = ...
    sample_system(@dynamics, sample_rate, control_inpute_rate, ...
                  control_sequence, Q);

%% Finding optimum hyperparameters
%set inputs and outputs
train_samples_output = samples(1,2:end);
train_samples_input = samples(:,1:end-1);

%set population size for the DE algorithm; approx 10x number of
%hyperparameters is a classic setting; however >40 shows limited improvement
population_size = 40; 

%number of iterations after which the algorithm will stop
maxIter = 200;

%range for the hyperparameters we are tuning; the algorithm as written will ONLY
%search in this range
min_hyperparam = [0; 0; 0; 0.0005];
max_hyperparam = [7; 7; 7; 7];

%Paramters to tune for the DE algorithm; if RandomizeF_weight is set then a
%new ranomd F_weight between 0.5 and 1 is selected each iteration
RandomizeF_weight = true;
F_weight = 0.8;                                     %classic setting = 0.8
CR = 0.9;                                           %classic setting = 0.9

%calling the DE opt function
opt_hyp_param = RunDiffEvolutionOpt(train_samples_output,train_samples_input, population_size, maxIter, min_hyperparam, max_hyperparam, F_weight, RandomizeF_weight, CR);



%% Hardcode Hyperparameters
% to hardcode the hyperparameters for testing purposes
% w = [0.2948; 0.1323];
% v1 = 6.2618;
% v0 = 0.0045;
% opt_hyp_param = [w; v1; v0];


%% Prediction

control_inpute_rate = 5;
control_imit = 1.3;
control_length = 101;
control_sequence = 2*control_imit*rand(control_length,1)-control_imit;

[sim_time, sim_y, sigma2, y] = simulate_system(@CovFunc, [0.1;0.1;0.1;0.1], ...
    samples, sample_rate, control_sequence, control_inpute_rate,@dynamics);


%% Plots
close all
hold on;
plot(sim_time,y);
plot(sim_time,sim_y,'-.');
%plot(sim_time,sim_y+2*sqrt(sigma2),'r',sim_time,sim_y-2*sqrt(sigma2),'r');
%legend('True trajectory', 'Predicted trajectory', 'Samples used for training', 'One sigma error bound');
