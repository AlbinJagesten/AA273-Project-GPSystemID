close all
clc
clear

dt = 0.001;
total_time = 100;
t = 1:dt:total_time;

y = zeros(length(t),1);

Q = 0.0025;

xmin = -1.3;
xmax = 1.3;
u = xmin+rand()*(xmax-xmin);
inpute_rate = 5;
sample_rate = 0.5;
sample_time = [];
samples = [];

for i = 1:length(t)-1
    
    %Input at inpute_rate
    if mod(i,round(inpute_rate/dt)) == 0
        u = xmin+rand()*(xmax-xmin);
    end    
   
    y(i+1) = dynamics(y(i),u,dt);
    
    if mod(i,round(sample_rate/dt)) == 0
        sample_time = [sample_time t(i)];
        samples = [samples [y(i)+sqrt(Q)*randn;u]];
    end   
    
end

K = CovFunc(samples(:,1:end-1),samples(:,1:end-1));

%K_inv = pinv(K+1e-8*eye(size(K)));
K_inv = pinv(K);
    
sim_time = 0:0.25:total_time;
sim_y = zeros(length(sim_time),1);
sim_y(1) = y(1);
sigma2 = zeros(length(sim_time),1);

for i = 1:length(sim_time)-1
    
    [~,idx] = min(abs(sample_time-sim_time(i)));
    
    u = samples(2,idx);
    
    k_vec =  CovFunc(samples(:,1:end-1), [sim_y(i);u]);
    
    sim_y(i+1) = k_vec' * (K_inv * samples(1,2:end)');
    
    k = CovFunc([sim_y(i);u],[sim_y(i);u]);
    
    sigma2(i) = k - k_vec' * K_inv * k_vec;
    
end

hold on
plot(t,y)
plot(sim_time,sim_y,'.-')
plot(sample_time,samples(1,:),'k.')
%plot(sim_time,sim_y+sigma2,'b--')

function new_y = dynamics(y,u,dt)

    new_y = y-dt*tanh(y+u^3);

end