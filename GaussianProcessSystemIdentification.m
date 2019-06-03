%% Simulation Setup

close all
clc
clear


%choosing dynamical system
dynamical_sys = @robot_dyn;
sys_dim = 2;

%choosing covariance function
global hyper_param_from_paper; hyper_param_from_paper = 0;
global ArdSquaredExp_four_hyper_params; ArdSquaredExp_four_hyper_params = 1;
global ArdSquaredExp_three_hyper_params; ArdSquaredExp_three_hyper_params = 2;
global ArdSquaredExp_two_hyper_params; ArdSquaredExp_two_hyper_params = 3;
global Locally_Periodic_Kernel; Locally_Periodic_Kernel = 4;
global Periodic_Kernel; Periodic_Kernel = 5;
global ArdRationalQuadratic; ArdRationalQuadratic = 6;

cov_fn_mode = [6 6];

%choosing time step for simulation
dt = 0.001;


%% Sample System
sample_rate = 0.5;
control_input_rate = 1;
Q = 0.0025*eye(sys_dim);

%control_index = linspace(0,10,20);
%control_sequence = [sin(control_index)];
control_sequence = cumsum(sign(randn(1,50)));

%control_sequence = 2.6*rand(1,20)-1.3;

[samples, sample_time] = ...
    sample_system(dynamical_sys, sample_rate, control_input_rate, ...
                  control_sequence, Q, dt);

              
%% Finding optimum hyperparameters using our optimizer

    train_samples_input = samples(:,1:end-1);

    hyper_params = [];
    
    %finding optimum hyperparameters for each element of output vector
    for i = 1:sys_dim
       
        train_samples_output = samples(i,2:end);
        hyper_param = find_param(train_samples_output, train_samples_input, cov_fn_mode(i));
        hyper_params = [hyper_params, hyper_param];
        
    end

%% Finding optimum hyperparameters using MATLAB toolbox
gpr_model = cell(1,sys_dim);

for i = 1:sys_dim

    train_samples_output = samples(i,2:end);
    gpr_model{i} = fitrgp(train_samples_input', train_samples_output', 'KernelFunction', 'ardrationalquadratic');

end


%% Prediction

pred_control_input_rate = 1;
%control_index = linspace(0,10,20);
%pred_control_sequence = [cos(control_index)];
%pred_control_sequence = 2.6*rand(1,20)-1.3;
pred_control_sequence = cumsum(0.25*sign(randn(1,25)));

[sim_time, pred_y, toolbox_pred_y, sigma2, true_y, t] = simulate_system(hyper_params, gpr_model,...
    samples, sample_rate, pred_control_sequence, pred_control_input_rate, dynamical_sys, dt, cov_fn_mode);


%% Plots
close all
for i = 1:sys_dim
    subplot(sys_dim,1,i);
    hold on;
    plot(t, true_y(i,:),'k','LineWidth',1.3);
    plot(sim_time, pred_y(i,:),'r--','LineWidth',1.3);
    plot(sim_time, toolbox_pred_y(i,:),'r-.','LineWidth',1.3);
    legend('True trajectory', 'Predicted trajectory using our GPR with DE optimizer', 'Predicted trajectory using MATLAB toolbox');
    title(['Prediction; train control rate = ', num2str(control_input_rate),', test control rate = ', num2str(pred_control_input_rate)])
    xlabel('time')
    ylabel('Output')
end

% figure
% plot(sample_time, samples(i,:),'-.');

%load gong.mat;
%sound(y);

%% 
% figure
% hold on;
% plot(true_y(1,:), true_y(2,:),'k','LineWidth',1.3);
% plot(toolbox_pred_y(1,:), toolbox_pred_y(2,:),'r--','LineWidth',1.3);

%% Subplots
% close all
% figure
% 
% subplot(2,1,1);
% plot(sample_time, samples(1,:),'-.');
% title(['Training data; train control rate = ', num2str(control_input_rate)])
% xlabel('time')
% 
% 
% subplot(2,1,2);
% hold on;
% plot(t, true_y(1,:),'k','LineWidth',1.3);
% plot(sim_time, pred_y(1,:),'b--','LineWidth',1.3);
% plot(sim_time, toolbox_pred_y(1,:),'r-.','LineWidth',1.3);
% legend('True trajectory', 'Predicted trajectory using our GPR with DE optimizer', 'Predicted trajectory using MATLAB toolbox');
% title(['Prediction; test control rate = ', num2str(pred_control_input_rate)])
% xlabel('time')
