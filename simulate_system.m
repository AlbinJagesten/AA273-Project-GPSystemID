function [sim_time, pred_y, toolbox_pred_y, sigma2, true_y, t] = ...
    simulate_system(opt_hyp_params, gpr_model, samples, sample_rate, ...
    control_sequence, control_input_rate, dynamics, dt, cov_fn,sys_dim,delta)
    
    % Simulation Parameters
    total_time = control_input_rate * (length(control_sequence)-1);
    t = 0:dt:total_time;    
    sim_time = 0:sample_rate:total_time;
    
    K_dim = size(samples,2)-1;
    num_param_sets = size(opt_hyp_params,2);
    K_inv = zeros(K_dim,K_dim,num_param_sets);
    
    [sys_input,sys_output] = GetTrainData(samples,sys_dim,delta);
    
    for i = 1:num_param_sets

        K = cov_fn(sys_input,sys_input,0,opt_hyp_params(:,i),'cov');
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
                sys_star = [pred_y(:,index);u];
                k_vec =  cov_fn(sys_input,sys_star,0,opt_hyp_params(:,j),'corr');
                k =  cov_fn(sys_star,sys_star,0,opt_hyp_params(:,j),'corr');

                %predicting y and obtaining the variance sigma 
                pred_y(j,index+1) = k_vec' * (K_inv(:,:,j) * sys_output(j,:)');
                sigma2(j,index+1) = k - k_vec' * K_inv(:,:,j) * k_vec;

                %PREDICTING USING MATLAB TOOLBOX GPR
                toolbox_pred_y(j,index+1) = predict(gpr_model{j}, [toolbox_pred_y(:,index)' , u']);

                if delta
                    pred_y(j,index+1) = pred_y(j,index+1)+pred_y(j,index);
                    toolbox_pred_y(j,index+1)=toolbox_pred_y(j,index+1)+toolbox_pred_y(j,index)';
                end

            end
        end

    end

end