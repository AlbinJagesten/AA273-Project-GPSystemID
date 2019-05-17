function C = CovFunc(X,Y)

    %X : #states x #samples
    %Y : #states x #samples

    w = [0.2948;0.1323];
    v0 = 6.2618;
    v1 = 0.0045;
    
    matrix = zeros(size(X,2),size(Y,2));
    
    for i = 1:size(X,2)
        for j = 1:size(Y,2)
            matrix(i,j) = w' * (X(:,i) - Y(:,j)).^2;
        end
    end
    
    C = v1 * exp(-1/2*matrix) + v0;

end