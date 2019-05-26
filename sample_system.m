function [samples,sample_time] = ...
    sample_system(dynamics, sample_rate, control_inpute_rate, ...
                  control_sequence, Q)

    % Simulation Parameters
    dt = 0.001;
    total_time = control_inpute_rate * (length(control_sequence) - 1);
    t = 1:dt:total_time;

    % System Setup

    %state variable
    sample_time = [];
    samples = [];
    state_dim = length(Q);
    state = zeros(state_dim,1);
    Q_sqrt = sqrtm(Q);

    % Simulating the system
    for i = 1:length(t)-1

        idx = round(dt*i/control_inpute_rate) + 1;

        state = dynamics(state,control_sequence(:,idx),dt);

        if mod(i,round(sample_rate/dt)) == 0
            sample_time = [sample_time t(i)];
            samples = [samples [state+Q_sqrt*randn(state_dim,1);...
                                control_sequence(:,idx)]];
        end   

    end

end