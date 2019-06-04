function hyper_param = Rprop(y,X)

%Parameters
num_iter = 1000;
eta_p = 1.2;
eta_n = 0.5;
delta = 0.1*ones(5,1);
min_delta = 1e-6;
max_delta = 500;
dJold = zeros(5,1);
hyper_param = log(0.1*rand(5,1));
epsilon = 1e-3;

for i = 1:num_iter
    
    dJ = find_grad(y,X,hyper_param);

%     if mod(i,100) == 0
%         fprintf('Iteration %d, norm %.3f\n',i,norm(dJ - dJold))
%     end
    
    delta = delta .* (eta_p .* (dJ .* dJold > 0) + eta_n .* (dJ .* dJold < 0) + (dJ .* dJold == 0));
    delta = max(delta,min_delta);
    delta = min(delta,max_delta);
    
    if norm(dJ - dJold) < epsilon
        disp('Done')
       break 
    end
    
    dJold = dJ;
    
    hyper_param = hyper_param - sign(dJ) .* delta;
    
    if i == num_iter
        disp('Did not converge')
    end
    
end

hyper_param = exp(hyper_param);

end