function output = ExponentialKernelCov(X,Y,y,hyper_param,mode)

    % ARD Squared Exponential Kernel
    %
    % mode : 'cov' - return covariance matrix (+ noise sigma)
    %        'corr' - return correlation vector/matrix
    %        'grad' - return gradient
    % 
    % X - (#input dim, #samples)
    % 
    % #hyper parameters: 3 + #input_dim
    
    dim = size(X,1);
    
    K = zeros(size(X,2),size(Y,2));
    sigma_n = hyper_param(1); 
    sigma_f = hyper_param(2);  
    sigmal_l = hyper_param(3); 
    
    if strcmp(mode,'cov') || strcmp(mode,'corr')
        for i = 1:size(X,2)
            for j = 1:size(Y,2)
                K(i,j) = sigma_f * exp(-norm(X(:,i) - Y(:,j))*sigmal_l);
            end
        end
        if strcmp(mode,'cov')
            K = K + sigma_n^2 * eye(size(K));
        end
        output = K;
        
    elseif strcmp(mode,'grad')
        
        grad = zeros(3,1);
        X_mat = zeros(size(K));

        for i = 1:size(X,2)
            for j = 1:size(Y,2)
                K(i,j) = sigma_f * exp(-norm(X(:,i) - Y(:,j))*sigmal_l);
                X_mat(i,j) = norm(X(:,i) - Y(:,j));
            end
        end
           
        
        dKdsigma_f = K / sigma_f;
        dKdsigma_n = 2*sigma_n*eye(size(K));
        dKdsigma_l = -X_mat .* K;
        
        K_inv = pinv(K + sigma_n^2*eye(size(K)));
        alpha = K_inv * y';
        
        grad(1) = -1/2*trace((alpha*alpha'-K_inv)*dKdsigma_n);
        grad(2) = -1/2*trace((alpha*alpha'-K_inv)*dKdsigma_f);
        grad(3) = -1/2*trace((alpha*alpha'-K_inv)*dKdsigma_l);
        
        output = grad;
        
    else
        
        ME = MException('MyComponent:noSuchMode',...
            'Wrong mode: %s \n',mode);
        throw(ME)
        
    end

end