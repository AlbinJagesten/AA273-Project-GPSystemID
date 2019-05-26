close all
clc
clear

sample_rate = 0.5;
control_inpute_rate = 5;
Q = 0*eye(3);
control_sequence = [ones(1,20);randn(1,20)];

[samples sample_time] = ...
    sample_system(@robot_dyn, sample_rate, control_inpute_rate, ...
                  control_sequence, Q);

%% Finding optimum hyperparameters

    train_samples_input = samples(:,1:end-1);
    
    hyper_params = [];
    
    for i = 1:length(Q)
       
        train_samples_output = samples(i,2:end);
        hyper_param = find_param(train_samples_output,train_samples_input,@CovFunc)
        hyper_params = [hyper_params hyper_param];
        
    end


%% Prediction

control_inpute_rate = 5;
control_sequence = [ones(1,20);randn(1,20)];

[sim_time, sim_y, sigma2, y] = simulate_system(@CovFunc, hyper_params, ...
    samples, sample_rate, control_sequence, control_inpute_rate,@robot_dyn);


%% Plots
close all
for i = 1:3
    figure
    hold on;
    plot(sim_time,y(i,:));
    plot(sim_time,sim_y(i,:),'-.');
    %plot(sim_time,sim_y(i,:)+2*sqrt(sigma2(i,:)),'r--');
    %plot(sim_time,sim_y(i,:)-2*sqrt(sigma2(i,:)),'r--');
    %legend('True trajectory', 'Predicted trajectory', 'Samples used for training', 'One sigma error bound');
end

figure
hold on
plot(y(1,:),y(2,:))
plot(sim_y(1,:),sim_y(2,:),'--')
