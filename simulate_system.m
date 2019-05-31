function [sim_time, pred_y, toolbox_pred_y, sigma2, true_y, t] = ...
    simulate_system(opt_hyp_params, gpr_model, samples, sample_rate, ...
    control_sequence, control_input_rate, dynamics, dt, cov_fn_mode)
    
    % Simulation Parameters
    total_time = control_input_rate * (length(control_sequence)-1);
    t = 0:dt:total_time;    
    sim_time = 0:sample_rate:total_time;
    
    K_dim = size(samples,2)-1;
    num_param_sets = size(opt_hyp_params,2);
    K_inv = zeros(K_dim,K_dim,num_param_sets);
    
    for i = 1:num_param_sets

        K = CovFunc(samples(:,1:end-1), samples(:,1:end-1), opt_hyp_params(:,i), cov_fn_mode(i));
        K_inv(:,:,i) = pinv(K);
    
    end

    pred_y = zeros(num_param_sets,length(sim_time));
    toolbox_pred_y = zeros(num_param_sets,length(sim_time));
    sigma2 = zeros(num_param_sets,length(sim_time));
    
    state = zeros(num_param_sets,1);
    true_y = zeros(num_param_sets,length(sim_time));
    
    
    for i = 1:length(t)-1 
        
        idx = round(dt*i/control_input_rate) + 1;
        u = control_sequence(:,idx);
                
        %ADVANCING THE "TRUE" STATE 
        state = dynamics(state, u, dt);
        true_y(:, i+1) = state;
        
        if mod(i,round(sample_rate/dt)) == 0
            index = (i/round(sample_rate/dt));
            %PREDICTING FOR EACH ELEMENT OF OUTPUT Y FOR THIS TIME STEP
            for j = 1:num_param_sets

                %PREDICTING USING OUR MODEL
                %finding the new kernel values using the new point
                k_vec =  CovFunc(samples(:,1:end-1), [pred_y(:,index);u], opt_hyp_params(:,j), cov_fn_mode(j));
                k = CovFunc([pred_y(:,index);u],[pred_y(:,index);u], opt_hyp_params(:,j), cov_fn_mode(j));

                %predicting y and obtaining the variance sigma 
                pred_y(j,index+1) = k_vec' * (K_inv(:,:,j) * samples(j,2:end)');
                sigma2(j,index+1) = k - k_vec' * K_inv(:,:,j) * k_vec;

                %PREDICTING USING MATLAB TOOLBOX GPR
                toolbox_pred_y(j, index+1) = predict(gpr_model{j}, [toolbox_pred_y(:,index)', u']);
            end
        end

    end

end