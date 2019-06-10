function grad = find_grad(y,X,hyper_param,mode)

    grad = zeros(2+size(X,1),1);

    hyper_param = exp(hyper_param);

    %X : #states x #samples
    %Y : #states x #samples

    global hyper_param_from_paper;
    global ArdSquaredExp_four_hyper_params;
    global ArdSquaredExp_three_hyper_params;
    global ArdSquaredExp_two_hyper_params;
    global Locally_Periodic_Kernel;
    global Periodic_Kernel;
    global ardsquaredexponential;

    %CHOOSE MODE
    %mode = three_hyper_params;

    if (mode == ardsquaredexponential)

        v0 = hyper_param(1);
        v1 = hyper_param(2);  
        len = hyper_param(3:2+size(X,1));

        C = zeros(size(X,2),size(X,2));

        for i = 1:size(X,2)
            for j = 1:size(X,2)
                C(i,j) = v1 * exp(-1/2*sum(((X(:,i) - X(:,j)).^2).*len));
            end
        end

        dKdv0 = 2*v0*eye(size(C));
        dKdv1 = C / v1;
        
        C_inv = pinv(C+ v0^2*eye(size(C)));
        alpha = C_inv * y';
        
        grad(1) = -1/2*trace((alpha*alpha'-C_inv)*dKdv0);
        grad(2) = -1/2*trace((alpha*alpha'-C_inv)*dKdv1);
        
        for i = 3:2+size(X,1)

            dKdw = -1/2*(X(i-2,:) - X(i-2,:)').^2 .* C;
            grad(i) = -1/2*trace((alpha*alpha'-C_inv)*dKdw);
        
        end
        

    elseif (mode == Periodic_Kernel)
        
        v0 = hyper_param(1);
        v1 = hyper_param(2);
        l = hyper_param(3);
        p = hyper_param(4);
        
        C = zeros(size(X,2),size(X,2));
        dl = zeros(size(X,2),size(X,2));
        dp = zeros(size(X,2),size(X,2));
        
        for i = 1:size(X,2)
            for j = 1:size(X,2)
                C(i,j) = v1*exp(-2*l^2*sin(pi*...
                    sum(abs(X(:,i) - X(:,j)))*p)^2);
                
                dl(i,j) = -4*l*sin(pi*...
                    sum(abs(X(:,i) - X(:,j)))*p)^2;
                
                dp(i,j) = -2*l^2*sin(pi*...
                    sum(abs(X(:,i) - X(:,j)))*p)*...
                cos(pi*...
                    sum(abs(X(:,i) - X(:,j)))*p)*...
                pi*sum(abs(X(:,i) - X(:,j)));
                
                if j == i
                    C(i,j) = C(i,j) + v0;
                end
            end
        end
        
        dKdv0 = eye(size(C));
        dKdv1 = C / v1;
        dKdl = dl .* C;
        dKdp = dp .* C;
        
        C_inv = pinv(C+ v0^2*eye(size(C)));
        alpha = C_inv * y';
        
        grad(1) = -1/2*trace((alpha*alpha'-C_inv)*dKdv0);
        grad(2) = -1/2*trace((alpha*alpha'-C_inv)*dKdv1);
        grad(3) = -1/2*trace((alpha*alpha'-C_inv)*dKdl);
        grad(4) = -1/2*trace((alpha*alpha'-C_inv)*dKdp);

    end

end