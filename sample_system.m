function [samples sample_time] = ...
    sample_system(dynamics, sample_rate, control_inpute_rate, ...
                  control_sequence, Q)

    % Simulation Parameters
    dt = 0.001;
    total_time = control_inpute_rate * (length(control_sequence) - 1);
    t = 1:dt:total_time;

    % System Setup

    %state variable
    y = zeros(length(t),1);
    sample_time = [];
    samples = [];

    % Simulating the system
    for i = 1:length(t)-1

        idx = round(dt*i/control_inpute_rate) + 1;

        y(i+1) = dynamics(y(i),control_sequence(idx),dt);

        if mod(i,round(sample_rate/dt)) == 0
            sample_time = [sample_time t(i)];
            samples = [samples [y(i)+sqrt(Q)*randn;control_sequence(idx)]];
        end   

    end

end