function log_p = LogLikelihood(K,y,Q)
    N = size(y,2);
    %adding elements to diagonal of K to make it invertible and get a
    %  non-zero determinant
    K = K + Q*eye(size(K));
    log_p = -0.5*log(det(K)) -0.5*(y)*(pinv(K))*(y') - (N/2)*log(2*pi);
end