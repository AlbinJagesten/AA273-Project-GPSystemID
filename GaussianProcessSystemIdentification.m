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
global ardsquaredexponential; ardsquaredexponential = 6;

cov_fn = @ArdSquaredExpCov;

%choosing time step for simulation
dt = 0.001;


%% Sample System
sample_rate = 0.5;
control_input_rate = 1;
Q = 0.0025*eye(sys_dim);

control_index = linspace(0,10,50);
control_sequence = [sin(control_index)];

[samples, sample_time] = ...
    sample_system(dynamical_sys, sample_rate, control_input_rate, ...
                  control_sequence, Q, dt);
              
[train_samples_input, train_samples_output] = GetTrainData(samples,sys_dim);
              

%% Finding optimum hyperparameters using our optimizer

    hyper_params = [];

    %finding optimum hyperparameters for each element of output vector
    for i = 1:sys_dim

        %hyper_param = find_param(train_samples_output, train_samples_input, cov_fn_mode(i));
        hyper_param = Rprop(train_samples_output(i,:), train_samples_input,cov_fn);
        hyper_params = [hyper_params hyper_param];
        K = cov_fn(train_samples_input,train_samples_input,train_samples_output(i,:),hyper_params(:,i),'cov');
        LogLikelihood(K,train_samples_output(i,:))
        
    end

%% Finding optimum hyperparameters using MATLAB toolbox
gpr_model = cell(1,sys_dim);

for i = 1:sys_dim

    gpr_model{i} = fitrgp(train_samples_input', train_samples_output(i,:)', 'KernelFunction', 'ardexponential');

end

%% Prediction

pred_control_input_rate = 1;
control_index = linspace(5,15,50);
pred_control_sequence = -[cos(control_index)];

[sim_time, pred_y, toolbox_pred_y, sigma2, true_y, t] = simulate_system(hyper_params, gpr_model,...
    samples, sample_rate, pred_control_sequence, pred_control_input_rate, dynamical_sys, dt, cov_fn);


%% Plots
close all
for i = 1:sys_dim
    subplot(sys_dim,1,i);
    hold on;
    plot(t, true_y(i,:),'k','LineWidth',1.3);
    plot(sim_time, pred_y(i,:),'b--','LineWidth',1.3);
    plot(sim_time, toolbox_pred_y(i,:),'r-.','LineWidth',1.3);
    legend('True trajectory', 'Predicted trajectory using our GPR with Rprop optimizer', 'Predicted trajectory using MATLAB toolbox');
    title(['Prediction; train control rate = ', num2str(control_input_rate),', test control rate = ', num2str(pred_control_input_rate)])
    xlabel('time')
    ylabel('Output')
end
return
%% trajectory plot

figure
hold on;
plot(true_y(1,:), true_y(2,:),'k','LineWidth',1.3);
plot(pred_y(1,:), pred_y(2,:),'b--','LineWidth',1.3);
plot(toolbox_pred_y(1,:), toolbox_pred_y(2,:),'r-.','LineWidth',1.3);
legend('True trajectory', 'Predicted trajectory using our GPR with Rprop optimizer', 'Predicted trajectory using MATLAB toolbox');
title('Prediction robot trajectory')
xlabel('x')
ylabel('y')

