function [grad, val] = find_grad(y,X,hyper_param)

grad = zeros(4,1);

w = [hyper_param(1); hyper_param(2)] ;
v1 = hyper_param(3);
v0 = hyper_param(4);

matrix = zeros(size(X,2),size(X,2));
x1y1 = zeros(size(X,2),size(X,2));
x2y2 = zeros(size(X,2),size(X,2));

for i = 1:size(X,2)
    for j = 1:size(X,2)
        matrix(i,j) = w' * (X(:,i) - X(:,j)).^2;
        x1y1(i,j) = -1/2*(X(1,i) - X(1,j)).^2;
        x2y2(i,j) = -1/2*(X(2,i) - X(2,j)).^2;
    end
end

K = v1 * exp(-1/2*matrix) + v0 * eye(size(matrix));

dKdv0 = eye(size(K));
dKdv1 = exp(-1/2*matrix);
dKdw1 = x1y1 * v1 * exp(-1/2*matrix);
dKdw2 = x2y2(i,j) * v1 * exp(-1/2*matrix);

K_inv = pinv(K);
alpha = K_inv * y';

grad(1) = 1/2*trace((alpha*alpha'-K_inv)*dKdw1);
grad(2) = 1/2*trace((alpha*alpha'-K_inv)*dKdw2);
grad(3) = 1/2*trace((alpha*alpha'-K_inv)*dKdv1);
grad(4) = 1/2*trace((alpha*alpha'-K_inv)*dKdv0);

N = size(y,2);
val = -0.5*log_det(K) -0.5*(y)*(K_inv)*(y') - (N/2)*log(2*pi);

end
