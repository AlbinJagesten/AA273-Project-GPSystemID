close all
clc
clear

dt = 0.001;
t = 1:dt:100;

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

w = [0.1323;0.2948];
v0 = 6.2618;
%v0 = 0;
v1 = 0.0045;

C = @(x,y) v0*exp(-1/2*w'*(x-y).^2)+v1;

K = zeros(length(sample_time)-1);

for i = 1:length(sample_time)-1
    for j = 1:length(sample_time)-1

        K(i,j) = C(samples(:,i),samples(:,j));

    end
end

K = K + 0.1 * eye(size(K));
K_inv = pinv(K);

sim_time = 0:0.1:100;
sim_y = zeros(length(sim_time),1);

for i = 1:length(sim_time)-1
    
    [~,idx] = min(abs(sample_time-sim_time(i)));
    
    u = samples(2,idx);
    
    k_vec =  zeros(length(sample_time)-1,1);
   
    for j = 1:length(sample_time)-1
        k_vec(j) = C(samples(:,j),[sim_y(i);u]);
    end
    
    sim_y(i+1) = k_vec' * (K_inv * samples(1,2:end)');
    
end

hold on
 plot(t,y)
plot(sim_time,sim_y)


function new_y = dynamics(y,u,dt)

    new_y = y-dt*tanh(y+u^3);

end