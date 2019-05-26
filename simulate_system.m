function [sim_time, sim_y, sigma2, y] = ...
    simulate_system(CovFunc, opt_hyp_param, samples, sample_rate, ...
    control_sequence, control_inpute_rate, dynamics)

    total_time = control_inpute_rate * (length(control_sequence)-1);

    K = CovFunc(samples(:,1:end-1), samples(:,1:end-1), opt_hyp_param);
    K_inv = pinv(K);

    sim_time = 0:sample_rate:total_time;
    sim_y = zeros(length(sim_time),1);
    sigma2 = zeros(length(sim_time),1);

    y = zeros(length(sim_time),1);

    for i = 1:length(sim_time)-1 
        
        idx = round(sample_rate*i/control_inpute_rate) + 1;
        u = control_sequence(idx);

        k_vec =  CovFunc(samples(:,1:end-1), [sim_y(i);u], opt_hyp_param);

        sim_y(i+1) = k_vec' * (K_inv * samples(1,2:end)');

        k = CovFunc([sim_y(i);u],[sim_y(i);u], opt_hyp_param);

        sigma2(i+1) = k - k_vec' * K_inv * k_vec;

        y(i+1) = dynamics(y(i),u,sample_rate);

    end

end