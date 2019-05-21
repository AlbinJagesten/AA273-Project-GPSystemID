function log_p = LogLikelihood(K,y)
    N = size(y,2);
    %adding elements to diagonal of K to make it invertible and get a
    %  non-zero determinant
    %K = K + 0.1 * eye(size(K));
    log_p = -0.5*log(det(K)) -0.5*(y)*(pinv(K))*(y') - (N/2)*log(2*pi);
end