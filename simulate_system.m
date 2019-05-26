function [sim_time, sim_y, sigma2, y] = ...
    simulate_system(CovFunc, opt_hyp_params, samples, sample_rate, ...
    control_sequence, control_inpute_rate, dynamics)

    total_time = control_inpute_rate * (length(control_sequence)-1);
    
    K_dim = size(samples,2)-1;
    num_params = size(opt_hyp_params,2);
    K_inv = zeros(K_dim,K_dim,num_params);
    
    for i = 1:num_params

        K = CovFunc(samples(:,1:end-1), samples(:,1:end-1), opt_hyp_params(:,i));
        K_inv(:,:,i) = pinv(K);
    
    end

    sim_time = 0:sample_rate:total_time;
    sim_y = zeros(num_params,length(sim_time));
    sigma2 = zeros(num_params,length(sim_time));
    state = zeros(num_params,1);

    y = zeros(num_params,length(sim_time));

    for i = 1:length(sim_time)-1 
        
        idx = round(sample_rate*i/control_inpute_rate) + 1;
        u = control_sequence(:,idx);
        
        for j = 1:num_params

            k_vec =  CovFunc(samples(:,1:end-1), [sim_y(:,i);u], opt_hyp_params(:,j));

            sim_y(j,i+1) = k_vec' * (K_inv(:,:,j) * samples(j,2:end)');

            k = CovFunc([sim_y(:,i);u],[sim_y(:,i);u], opt_hyp_params(:,j));

            sigma2(j,i+1) = k - k_vec' * K_inv(:,:,j) * k_vec;
        
        end
        
        state = dynamics(state,u,sample_rate);

        y(:,i+1) = state;

    end

end