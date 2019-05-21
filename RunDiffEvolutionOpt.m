function opt_hyp_param = RunDiffEvolutionOpt(train_samples_output,train_samples_input, population_size, maxIter, min_hyperparam, max_hyperparam, F_weight, CR, Q)
    
    %% variable declaration and initializations
    
    num_param = size(min_hyperparam, 1);
  
    current_pop = zeros(num_param, population_size);
    current_log_likelihood = zeros(1, population_size);
    
    opt_hyp_param = (min_hyperparam + max_hyperparam)/2;        %arbitrary
    K_temp = CovFunc(train_samples_input, train_samples_input, opt_hyp_param);
    opt_log_lik = LogLikelihood(K_temp, train_samples_output, Q);
    
    
    %% generating initial population
    
    for i = 1:population_size
        current_pop(:,i) = min_hyperparam + rand(num_param,1).*(max_hyperparam - min_hyperparam);
    end
    
    
    
    %% evaluating initial optimum
    
    for i = 1:population_size
        K = CovFunc(train_samples_input, train_samples_input, current_pop(:,i));
        current_log_likelihood(i) = LogLikelihood(K, train_samples_output, Q);
        if current_log_likelihood(i) > opt_log_lik
            opt_log_lik = LogLikelihood(K, train_samples_output, Q);
            opt_hyp_param = current_pop(:,i);
        end
    end
    
    
    
    %% iterating to find optimum parameters
    
    for iter = 1:maxIter
        iter
        % GENERATING NEW CANDIDATE POPULATION
        %scrambling the population and creating 3 new sets offsets are used to make 1 set of scrambled indices into 3 by
        %shifting the indices by this constant offset; this is better than using randperm 3 times as this ensures that 
        %3 populations are necessarily different
        scrambled_indices_1 = randperm(population_size);
        index_offset_1 = round(rand()*population_size) + 1; %1 is added to ensure that the min value is not zero                 
        index_offset_2 = round(rand()*population_size) + 1; 
        if (index_offset_1 == index_offset_2)
            index_offset_2 = index_offset_1 + population_size/2; %arbitrary heuristic
        end
        scrambled_indices_2 = 1 + rem((scrambled_indices_1 + index_offset_1), population_size);
        scrambled_indices_3 = 1 + rem((scrambled_indices_1 + index_offset_2), population_size);
        
        a = current_pop(:, scrambled_indices_1);
        b = current_pop(:, scrambled_indices_2);
        c = current_pop(:, scrambled_indices_3);
        
        new_pop = a + F_weight*(b - c);   
    
        %PERFORMING CROSSOVER BETWEEN OLD AND NEW POPULATIONS
        %generating random masks in order to select members from new population
        % and remaining from old population
        new_pop_mask = rand(num_param, population_size) < CR;  % all random numbers < F_CR are 1, 0 otherwise
        old_pop_mask = new_pop_mask < 0.5;                     % inverse mask of new_pop_mask
        
        new_pop = new_pop_mask.*new_pop + old_pop_mask.*current_pop;
        
        %BOUNDING THE POPULATION AND DECIDING WHICH MEMBERS ARE CARRIED OVER
        %checking that the log likelihood of the new member is better than the older counterpart
        %boundary constraints via bounce back: if we do not bound it, things that aren't (say) supposed to go -ve
        %could go negative and result in numerical issues
        for i = 1:population_size
            for j = 1:num_param             
               if (new_pop(j,i) > max_hyperparam(j))
                  new_pop(j,i) = max_hyperparam(j) + rand()*(current_pop(j,i) - max_hyperparam(j));
               end
               if (new_pop(j,i) < min_hyperparam(j))
                  new_pop(j,i) = min_hyperparam(j) + rand()*(current_pop(j,i) - min_hyperparam(j));
               end   
            end
            
            K = CovFunc(train_samples_input, train_samples_input, new_pop(:,i));
            new_log_likelihood = LogLikelihood(K, train_samples_output, Q);
            if new_log_likelihood > current_log_likelihood(i)
                current_pop(:,i) = new_pop(:,i);
                current_log_likelihood(i) = new_log_likelihood;
            end
            
        end
        
        %FINDING OPTIMUM PARAMETERS IN NEW POPULATION
        for i = 1:population_size
            if current_log_likelihood(i) > opt_log_lik
                opt_log_lik = current_log_likelihood(i);
                opt_hyp_param = current_pop(:,i);
            end
        end
        
        current_pop = new_pop;
end