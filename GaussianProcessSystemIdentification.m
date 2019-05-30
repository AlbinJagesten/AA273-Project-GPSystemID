%% Simulation Setup

close all
clc
clear

dynamical_sys = @robot_dyn;
sys_dim = 3;
control_dim = 2;
cov_fn = @CovFunc;


%% Sample System

sample_rate = 0.5;
control_inpute_rate = 5;
Q = 0.0025*eye(sys_dim);

control_sequence = [ones(1,5), -ones(1,5)];

[samples, sample_time] = ...
    sample_system(dynamical_sys, sample_rate, control_inpute_rate, ...
                  control_sequence, Q);

              
%% Finding optimum hyperparameters

    train_samples_input = samples(:,1:end-1);
    
    hyper_params = [];
    
    %finding optimum hyperparameters for each element of output vector
    for i = 1:sys_dim
       
        train_samples_output = samples(i,2:end);
        hyper_param = find_param(train_samples_output, train_samples_input, cov_fn);
        hyper_params = [hyper_params hyper_param];
        
    end


%% Prediction

pred_control_inpute_rate = 5;
pred_control_sequence = [ones(1,5), -ones(1,5)];

[sim_time, sim_y, sigma2, y] = simulate_system(cov_fn, hyper_params, ...
    samples, sample_rate, pred_control_sequence, pred_control_inpute_rate, dynamical_sys);


%% Plots
close all
for i = 1:sys_dim
    figure
    hold on;
    plot(sim_time,y(i,:));
    plot(sim_time,sim_y(i,:),'-.');
    %plot(sim_time,sim_y(i,:)+2*sqrt(sigma2(i,:)),'r--');
    %plot(sim_time,sim_y(i,:)-2*sqrt(sigma2(i,:)),'r--');
    %legend('True trajectory', 'Predicted trajectory', 'Samples used for training', 'One sigma error bound');
end

% figure
% hold on
% plot(y(1,:),y(2,:))
% plot(sim_y(1,:),sim_y(2,:),'--')
% 
% load gong.mat;
% sound(y);
