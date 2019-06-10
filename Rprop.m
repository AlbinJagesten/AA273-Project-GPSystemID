function hyper_param = Rprop(y,X,cov_fn,numhyp)

%Parameters
num_iter = 200;
eta_p = 1.2;
eta_n = 0.5;
delta = 0.1*ones(numhyp,1);
min_delta = 1e-6;
max_delta = 500;
dJold = zeros(numhyp,1);
hyper_param = log(rand(numhyp,1));
epsilon = 1e-3;

for i = 1:num_iter

    dJ = cov_fn(X,X,y,exp(hyper_param),'grad');
    
    delta = delta .* (eta_p .* (dJ .* dJold > 0) + eta_n .* (dJ .* dJold < 0) + (dJ .* dJold == 0));
    delta = max(delta,min_delta);
    delta = min(delta,max_delta);
    
    dJold = dJ;
    
    hyper_param = hyper_param - sign(dJ) .* delta;

    if norm(dJ) < epsilon
        disp('Converged')
        break
    end
    
end

hyper_param = exp(hyper_param);

end