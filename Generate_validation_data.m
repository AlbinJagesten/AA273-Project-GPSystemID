close all
clc
clear

dt = 0.5;
total_time = 100;
umin = -1.3;
umax = 1.3;
control_inpute_rate = 5;

sim_time = 0:dt:total_time;
control_input = zeros(length(sim_time)-1,1);

u = umin+rand()*(umax-umin);

for i = 1:length(sim_time)-1
    
    if mod(i,round(control_inpute_rate/dt)) == 0
        u = umin+rand()*(umax-umin);
    end    
    
    control_input(i) = u;
    
end

save('Validation_data.mat','control_input')