%% Simulation Setup

close all
clc
clear


%choosing dynamical system
dynamical_sys = @robot_dyn;
sys_dim = 3;

%choosing covariance function
cov_fn = @CovFunc;

%choosing time step for simulation
dt = 0.001;

%% Sample System
sample_rate = 0.5;
control_inpute_rate = 5;
Q = 0.0025*eye(sys_dim);

control_sequence = [ones(1,5), -ones(1,5)];

[samples, sample_time] = ...
    sample_system(dynamical_sys, sample_rate, control_inpute_rate, ...
                  control_sequence, Q, dt);

              
%% Finding optimum hyperparameters using our optimizer

    train_samples_input = samples(:,1:end-1);

    hyper_params = [];
    
    %finding optimum hyperparameters for each element of output vector
    for i = 1:sys_dim
       
        train_samples_output = samples(i,2:end);
        hyper_param = find_param(train_samples_output, train_samples_input, cov_fn);
        hyper_params = [hyper_params hyper_param];
        
    end

%% Finding optimum hyperparameters using MATLAB toolbox
gpr_model = cell(1,sys_dim);
for i = 1:sys_dim
    train_samples_output = samples(i,2:end);
    gpr_model{i} = fitrgp(train_samples_input', train_samples_output', 'KernelFunction', 'ardrationalquadratic');
end


%% Prediction

pred_control_input_rate = 5;
pred_control_sequence = [ones(1,5), ones(1,5)];

[sim_time, pred_y, toolbox_pred_y, sigma2, true_y, t] = simulate_system(cov_fn, hyper_params, gpr_model,...
    samples, sample_rate, pred_control_sequence, pred_control_input_rate, dynamical_sys, dt);


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

% figure
% hold on
% plot(y(1,:),y(2,:))
% plot(sim_y(1,:),sim_y(2,:),'--')
% 
% load gong.mat;
% sound(y);
