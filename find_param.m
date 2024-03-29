function hyper_param = find_param(train_samples_output,train_samples_input,covfn,numhyp)

    %set inputs and outputs

    %set population size for the DE algorithm; approx 10x number of
    %hyperparameters is a classic setting; however >40 shows limited improvement
    population_size = numhyp*10;

    %number of iterations after which the algorithm will stop
    maxIter = 100;

    %range for the hyperparameters we are tuning; the algorithm as written will ONLY
    %search in this range
    min_hyperparam = zeros(numhyp,1);
    max_hyperparam = 7*ones(numhyp,1);

    %Paramters to tune for the DE algorithm; if RandomizeF_weight is set then a
    %new ranomd F_weight between 0.5 and 1 is selected each iteration
    RandomizeF_weight = true;
    F_weight = 0.8;                                     %classic setting = 0.8
    CR = 0.9;                                           %classic setting = 0.9

    %calling the DE opt function
    hyper_param = RunDiffEvolutionOpt(train_samples_output,...
        train_samples_input, population_size, maxIter, min_hyperparam, ...
        max_hyperparam, F_weight, RandomizeF_weight, CR, covfn);

end
