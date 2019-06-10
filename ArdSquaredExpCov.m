function output = ArdSquaredExpCov(X,Y,y,hyper_param,mode)

    % ARD Squared Exponential Kernel
    %
    % mode : 'cov' - return covariance matrix (+ noise sigma)
    %        'corr' - return correlation vector/matrix
    %        'grad' - return gradient
    % 
    % X - (#input dim, #samples)
    % 
    % #hyper parameters: 2 + #input_dim
    
    dim = size(X,1);
    
    K = zeros(size(X,2),size(Y,2));
    sigma_n = hyper_param(1); 
    sigma_f = hyper_param(2);  
    sigma_m_inv = hyper_param(3:2+dim);
    
    if strcmp(mode,'cov') || strcmp(mode,'corr')
        for i = 1:size(X,2)
            for j = 1:size(Y,2)
                K(i,j) = sigma_f * exp(-1/2*sum(((X(:,i) - Y(:,j)).^2).*sigma_m_inv));
            end
        end
        if strcmp(mode,'cov')
            K = K + sigma_n^2 * eye(size(K));
        end
        output = K;
        
    elseif strcmp(mode,'grad')
        
        grad = zeros(2+dim,1);

        for i = 1:size(X,2)
            for j = 1:size(X,2)
                K(i,j) = sigma_f * exp(-1/2*sum(((X(:,i) - X(:,j)).^2).*sigma_m_inv));
            end
        end
        
        dKdsigma_f = K / sigma_f;
        dKdsigma_n = 2*sigma_n*eye(size(K));
        
        K_inv = pinv(K + sigma_n^2*eye(size(K)));
        alpha = K_inv * y';
        
        grad(1) = -1/2*trace((alpha*alpha'-K_inv)*dKdsigma_n);
        grad(2) = -1/2*trace((alpha*alpha'-K_inv)*dKdsigma_f);
        
        for i = 3:2+dim

            dKdsigma_m_inv = -1/2*(X(i-2,:) - X(i-2,:)').^2 .* K;
            grad(i) = -1/2*trace((alpha*alpha'-K_inv)*dKdsigma_m_inv);
        
        end
        
        output = grad;
        
    else
        
        ME = MException('MyComponent:noSuchMode',...
            'Wrong mode: %s \n',mode);
        throw(ME)
        
    end

end