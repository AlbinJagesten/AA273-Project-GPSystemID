function grad = num_grad_cov(y,X,hyp,mode)

    hyp = exp(hyp);
    dt = 1e-4;
    grad = zeros(length(hyp),1);
    
    for i = 1:length(hyp) 

        dhyp = hyp;
        dhyp(i) = hyp(i)+dt;
        C1 = CovFunc(X,X, hyp, mode);
        C2 = CovFunc(X,X, dhyp, mode);
        grad(i) = (LogLikelihood(C2,y)-LogLikelihood(C1,y))/dt;
        
    end
    
    grad = -grad;
    
end