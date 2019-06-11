%% Simulation Setup

%close all
rng(1)
clc
%clear

%choosing dynamical system
% dynamical_sys = @robot_dyn;
% sys_dim = 2;
dynamical_sys = @dynamics;
sys_dim = 1;
control_dim = 1;
Rprop = false;

%ArdSquaredExpCov
%ArdRationalQuadraticCov
%ExponentialKernelCov

cov_fn = @ArdSquaredExpCov;
delta = false;

%choosing time step for simulation
dt = 0.001;

numhypArdSqExp = 2 + sys_dim + control_dim;
numhypArdRaQu = 3 + control_dim+sys_dim;
numhypExpKer = 3;

numhyp = [numhypArdSqExp];


%% Sample System
sample_rate = 0.5;
Q = 0.0025*eye(sys_dim);

% control_input_rate = 1;
% control_index = linspace(0,10,50);
% control_sequence = [sin(control_index+rand(1,50))];

control_input_rate = 5;
control_sequence = 2.6*(rand(1,20)-0.5);

[samples, sample_time] = ...
    sample_system(dynamical_sys, sample_rate, control_input_rate, ...
                  control_sequence, Q, dt);
              
[train_samples_input, train_samples_output] = GetTrainData(samples,sys_dim,delta);
              

%% Finding optimum hyperparameters using our optimizer

    hyper_params = [];
    
    %finding optimum hyperparameters for each element of output vector
    for i = 1:sys_dim

        if Rprop
            hyper_param = Rprop(train_samples_output(i,:), train_samples_input,cov_fn,numhyp);
        else
            hyper_param = find_param(train_samples_output, train_samples_input, cov_fn,numhyp);
        end
        
        hyper_params = [hyper_params hyper_param];
        K = cov_fn(train_samples_input,train_samples_input,train_samples_output(i,:),hyper_params(:,i),'cov');
        LogLikelihood(K,train_samples_output(i,:))

    end


%% Finding optimum hyperparameters using MATLAB toolbox
gpr_model = cell(1,sys_dim);

% Functions
% ardrationalquadratic
% ardexponential

for i = 1:sys_dim

    gpr_model{i} = fitrgp(train_samples_input', train_samples_output(i,:)', 'KernelFunction', 'ardrationalquadratic');

end

%% Prediction

% pred_control_input_rate = 1;
% control_index = linspace(7,17,50);
% pred_control_sequence = -[cos(control_index)+1];

pred_control_input_rate = 5;
pred_control_sequence = 2.6*(rand(1,20)-0.5);

[sim_time, pred_y, toolbox_pred_y, sigma2, true_y, t] = simulate_system(hyper_params, gpr_model,...
    samples, sample_rate, pred_control_sequence, pred_control_input_rate, dynamical_sys, dt, cov_fn,sys_dim,delta);


%% Plots

figure

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

