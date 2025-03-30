clear;clc;close all;


%% equality constrained entropy maximization
n = 20; v1 = floor(n/4); v2 = floor(n/6); v3 = floor(n/3);
 A = [ones(1,n);
     ones(1,v1),zeros(1,n-v1);
     ones(1,v2),zeros(1,n-v2); 
     zeros(1,n-v3), (1:v3)/v3];
b = [1;0.2;0.1;0.02]; 

m = length(b);

obj_f = @(x) sum(x.*log(x));
obj_grad = @(x) log(x)+1;
obj_hess = @(x) diag(1./(x));

x0 = 1*ones(n,1); 
xs = [x0];

eps1 =  1e-6;
eps2 = 1e-6;

r_dual = @(x,lambda) obj_grad(x) + A'*lambda;
r_primal = @(x) A*x-b;
r = @(x,lambda) [r_dual(x,lambda); r_primal(x)];
iter = 1;
lambda = zeros(m,1);
tic
while norm(r_dual(xs(:,end),lambda)) > eps1 || norm(r_primal(xs(:,end))) > eps2
    % Newton step
    Hess = obj_hess(xs(:,end));
    M = [Hess, A' ; A , zeros(m,m)];
    RightHand = -[r_dual(xs(:,end),lambda); r_primal(xs(:,end))];
    
    result = M\RightHand;
    
    delta_x_pd = result(1:n);
    delta_lambda_pd = result(n+1:end);
    %line search
    f = @(alpha) norm(r(xs(:,end)+alpha*delta_x_pd, lambda+alpha*delta_lambda_pd));
    [a,c] = bracket_minimum(f);
    [a,c] = golden_section_search(f,a,c,30);
    alpha = (a+c)/2;
    %ulozeni novych hodnot
    xs(:,end+1) = xs(:,end) + alpha*delta_x_pd;
    lambda = lambda + alpha*delta_lambda_pd;
    iter = iter + 1;
end
toc
%pro n = 1400 to trva 1.05s

% disp(xs(:,1))
disp(xs(:,end))
y1 = obj_f(xs(:,end));
disp(y1)

% pro kontrolu 
[x2,y2] = fmincon(obj_f,x0,[],[],A,b);
disp(x2)
disp(y2)



%% Linear programming in inequality form - barrier

% rng(1,'twister');
% n = 20; m = 60;
% c = randn(n,1);
% A = rand(m,n); x0 = ones(n,1);
% b = A*x0 + 5 + 10*rand(m,1);
% A_ineq = [A;-eye(n)]; b_ineq = [b;zeros(n,1)];
% Phi = @(x) -sum(log(b_ineq-A_ineq*x));
% r = @(x) 1./(b_ineq-A_ineq*x);
% Phi_grad = @(x) A_ineq'*r(x);
% Phi_hess = @(x) A_ineq'*(diag(r(x))^2)*A_ineq;
% 
% xs = [x0]; t = 0.1; beta = 20; vareps = 1e-4;
% 
% [x_lin,fval] = linprog(c,A_ineq,b_ineq) % pro kontrolu


%% Quadratic programming in inequality form, SVM
% rng(1,'twister');
% n = 20;
% x_p = 0.4*randn(n,2) + 1;
% x_m = 0.7*randn(n,2) - 1;
% C = 1e-1; 
% 
% P = zeros(3+2*n,3+2*n); P(1,1) = 1; P(2,2) = 1; f = [zeros(3,1);C*ones(2*n,1)];
% b = -ones(2*n,1); A = [-x_p, -ones(n,1); x_m, ones(n,1)]; A = [A,-eye(2*n)];
% A_ineq = [A; zeros(2*n,3), -eye(2*n)]; b_ineq = [b;zeros(2*n,1)];
% 
% Phi = @(x) -sum(log(b_ineq-A_ineq*x));
% r = @(x) 1./(b_ineq-A_ineq*x);
% Phi_grad = @(x) A_ineq'*r(x);
% Phi_hess = @(x) A_ineq'*(diag(r(x))^2)*A_ineq;
% 
% m = 4*n;
% xs = [zeros(3,1);10*ones(2*n,1)]; 
% t = 0.1; beta = 20; vareps = 1e-3;
% 
% 
% [x_opt, fval] = quadprog(P,f,A_ineq,b_ineq)  % pro kontrolu