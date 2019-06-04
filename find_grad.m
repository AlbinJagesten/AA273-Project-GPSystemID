function grad = find_grad(y,X,hyper_param)

grad = zeros(5,1);

hyper_param = exp(hyper_param);

len = [hyper_param(1); hyper_param(2); hyper_param(3)] ;
sigma2 = hyper_param(4);
v0 = hyper_param(5);

C = zeros(size(X,2),size(X,2));

for i = 1:size(X,2)
    for j = 1:size(X,2)
        C(i,j) = sigma2 * exp(-1/2*sum(((X(:,i) - X(:,j)).^2).*len));
    end
end

dKdv0 = 2*v0*eye(size(C));
dKdv1 = C / sigma2;
dKdw1 = -1/2*(X(1,:) - X(1,:)').^2 .* C;
dKdw2 = -1/2*(X(2,:) - X(2,:)').^2 .* C;
dKdw3 = -1/2*(X(3,:) - X(3,:)').^2 .* C;

C = C + v0^2*eye(size(C));

C_inv = pinv(C);
alpha = C_inv * y';

grad(1) = -1/2*trace((alpha*alpha'-C_inv)*dKdw1);
grad(2) = -1/2*trace((alpha*alpha'-C_inv)*dKdw2);
grad(3) = -1/2*trace((alpha*alpha'-C_inv)*dKdw3);
grad(4) = -1/2*trace((alpha*alpha'-C_inv)*dKdv1);
grad(5) = -1/2*trace((alpha*alpha'-C_inv)*dKdv0);

end