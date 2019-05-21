close all
clc
clear

w = [0.2948; 0.1323];
v1 = 6.2618;
v0 = 0.0045;
opt_hyp_param = [w; v1; v0];

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

K = CovFunc(samples(:,1:end-1),samples(:,1:end-1),opt_hyp_param);

%K_inv = pinv(K+1e-8*eye(size(K)));
K_inv = pinv(K);
    

dt = 0.5;
sim_time = 0:dt:total_time;
sim_y = zeros(length(sim_time),1);
sigma2 = zeros(length(sim_time),1);

v0 = 0.0045;

u = xmin+rand()*(xmax-xmin);

y = zeros(length(sim_time),1);
y(1) = 0;

for i = 1:length(sim_time)-1
    
    if mod(i,round(inpute_rate/dt)) == 0
        u = xmin+rand()*(xmax-xmin);
    end    
    
    k_vec =  CovFunc(samples(:,1:end-1), [sim_y(i);u],opt_hyp_param);
    
    sim_y(i+1) = k_vec' * (K_inv * samples(1,2:end)');
    
    k = CovFunc([sim_y(i);u],[sim_y(i);u],opt_hyp_param);
    
    sigma2(i+1) = k - k_vec' * K_inv * k_vec+v0;
    
    y(i+1) = dynamics(y(i),u,dt);
    
end

hold on
plot(sim_time,sim_y+sqrt(sigma2),'r--')
plot(sim_time,sim_y-sqrt(sigma2),'r--')
plot(sim_time,y,'c')
plot(sim_time,sim_y,'k--')

function new_y = dynamics(y,u,dt)

    new_y = y-dt*tanh(y+u^3);

end