%% Simulation Setup

close all
clc
clear


%choosing dynamical system
dynamical_sys = @robot_dyn;
sys_dim = 3;

%choosing covariance function
global hyper_param_from_paper; hyper_param_from_paper = 0;
global four_hyper_params; four_hyper_params = 1;
global three_hyper_params; three_hyper_params = 2;
global two_hyper_params; two_hyper_params = 3;
global Locally_Periodic_Kernel; Locally_Periodic_Kernel = 4;
global Periodic_Kernel; Periodic_Kernel = 5;

cov_fn_mode = [4, 4, 4];

%choosing time step for simulation
dt = 0.001;


%% Sample System
sample_rate = 0.5;
control_inpute_rate = 1;
Q = 0.0025*eye(sys_dim);

control_index = linspace(0,10,20);
control_sequence = [sin(control_index)];

[samples, sample_time] = ...
    sample_system(dynamical_sys, sample_rate, control_inpute_rate, ...
                  control_sequence, Q, dt);

              
%% Finding optimum hyperparameters using our optimizer

    train_samples_input = samples(:,1:end-1);

    hyper_params = [];
    
    %finding optimum hyperparameters for each element of output vector
    for i = 1:sys_dim
       
        train_samples_output = samples(i,2:end);
        hyper_param = find_param(train_samples_output, train_samples_input, cov_fn_mode(i));
        hyper_params = [hyper_params hyper_param];
        
    end

%% Finding optimum hyperparameters using MATLAB toolbox
gpr_model = cell(1,sys_dim);
for i = 1:sys_dim
    train_samples_output = samples(i,2:end);
    gpr_model{i} = fitrgp(train_samples_input', train_samples_output', 'KernelFunction', 'ardrationalquadratic');
end


%% Prediction

pred_control_input_rate = 1;
control_index = linspace(0,10,20);
pred_control_sequence = [cos(control_index)];

[sim_time, pred_y, toolbox_pred_y, sigma2, true_y, t] = simulate_system(hyper_params, gpr_model,...
    samples, sample_rate, pred_control_sequence, pred_control_input_rate, dynamical_sys, dt, cov_fn_mode);


%% Plots
close all
for i = 1:sys_dim
    subplot(sys_dim,1,i);
    hold on;
    plot(t, true_y(i,:));
    plot(sim_time, pred_y(i,:),'-.');
    plot(sim_time, toolbox_pred_y(i,:),'--');
    %plot(sample_time, samples(i,:),'-.');
    legend('True trajectory', 'Predicted trajectory using our GPR with DE optimizer', 'Predicted trajectory using MATLAB toolbox', 'Initial Training Samples');
end

figure
hold on
plot(true_y(1,:),true_y(2,:))
plot(toolbox_pred_y(1,:), toolbox_pred_y(2,:),'--')

%load gong.mat;
%sound(y);
