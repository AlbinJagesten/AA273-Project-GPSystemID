function [train_samples_input, train_samples_output] = ...
    GetTrainData(samples,sys_dim)

delta_samples = samples(1:sys_dim,2:end) - samples(1:sys_dim,1:end-1);

train_samples_input = samples(:,1:end-1);
train_samples_output = samples(1:sys_dim,2:end) - samples(1:sys_dim,1:end-1);

end