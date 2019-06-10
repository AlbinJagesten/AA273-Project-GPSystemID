function grad = num_grad_cov(y,X,hyp,cov_fn)

    %hyp = exp(hyp);
    dt = 1e-4;
    grad = zeros(length(hyp),1);
    
    for i = 1:length(hyp) 

        dhyp = hyp;
        dhyp(i) = hyp(i)+dt;
        C1 = cov_fn(X,X,y, hyp, 'cov');
        C2 = cov_fn(X,X,y, dhyp, 'cov');
        grad(i) = (LogLikelihood(C2,y)-LogLikelihood(C1,y))/dt;
        
    end
    
    grad = -grad;
    
end