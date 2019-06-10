function C = CovFunc(X,Y, hyper_param, mode)

    X = X;
    Y = Y;

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

    if (mode == hyper_param_from_paper)
        w = [0.2948; 0.1323];
        v1 = 6.2618;
        v0 = 0.0045;

        matrix = zeros(size(X,2),size(Y,2));

        for i = 1:size(X,2)
            for j = 1:size(Y,2)
                matrix(i,j) = w' * ((X(:,i) - Y(:,j)).^2);
            end
        end

        C = v1 * exp(-1/2*matrix) + v0 * eye(size(matrix));

    elseif (mode == ArdSquaredExp_four_hyper_params)
        v0 = hyper_param(1);
        v1 = hyper_param(2);
        w = hyper_param(3:end);

        matrix = zeros(size(X,2),size(Y,2));

        for i = 1:size(X,2)
            for j = 1:size(Y,2)
                matrix(i,j) = w' * ((X(:,i) - Y(:,j)).^2);
            end
        end

        C = v1 * exp(-1/2*matrix) + v0 * eye(size(matrix));

    elseif (mode == ArdSquaredExp_three_hyper_params)
        v0 = hyper_param(1);
        v1 = hyper_param(2);
        w = hyper_param(3);

        matrix = zeros(size(X,2),size(Y,2));

        for i = 1:size(X,2)
            for j = 1:size(Y,2)
                matrix(i,j) = w * sum((X(:,i) - Y(:,j)).^2);
            end
        end

        C = v1 * exp(-1/2*matrix) + v0 * eye(size(matrix));


    elseif (mode == ArdSquaredExp_two_hyper_params)
        v1 = hyper_param(1);
        w = hyper_param(2);

        matrix = zeros(size(X,2),size(Y,2));

        for i = 1:size(X,2)
            for j = 1:size(Y,2)
                matrix(i,j) = w * sum((X(:,i) - Y(:,j)).^2);
            end
        end

        C = v1 * exp(-1/2*matrix);


    elseif (mode == Locally_Periodic_Kernel)

        sigma = hyper_param(1);
        l = hyper_param(2);
        p = hyper_param(3);

        C = zeros(size(X,2),size(Y,2));

        for i = 1:size(X,2)
            for j = 1:size(Y,2)
                C(i,j) = sigma^2*exp(-2/l^2*sin(pi*...
                    sum(abs(X(:,i) - Y(:,j)))/p)^2)*...
                    exp(-1/(2*l^2)*sum((X(:,i) - Y(:,j)).^2));
            end
        end

    elseif (mode == Periodic_Kernel)

        v0 = hyper_param(1);
        v1 = hyper_param(2);
        l = hyper_param(3);
        p = hyper_param(4);

        C = zeros(size(X,2),size(Y,2));

        for i = 1:size(X,2)
            for j = 1:size(Y,2)
                C(i,j) = v1*exp(-2*l^2*sin(pi*...
                    sum(abs(X(:,i) - Y(:,j)))*p)^2);
                if j == i
                    C(i,j) = C(i,j) + v0;
                end
            end
        end
        return

    elseif (mode == ardsquaredexponential) 
        
        v0 = hyper_param(1);
        v1 = hyper_param(2);  
        len = hyper_param(3:2+size(X,1));

        C = zeros(size(X,2),size(Y,2));

        for i = 1:size(X,2)
            for j = 1:size(Y,2)
                C(i,j) = v1 * exp(-1/2*sum(((X(:,i) - Y(:,j)).^2).*len));
                if j == i
                    C(i,j) = C(i,j) + v0^2;
                end
            end
        end
    end


end
