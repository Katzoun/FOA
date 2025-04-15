clear;clc;close all; rng(1,'twister');

%% Problem 0, simple example
% c = [-3,-2,0,0]';
% A = [1 1 1 0;2 0.5 0 1];
% [m,n] = size(A);
% b = [5;8];
% B_ids = [3,4]; N_ids = setdiff(1:n,B_ids);
% 
% [x,B_ids,N_ids,zs,flag] = simplex_method(c,A,b,B_ids,N_ids)


%% Problem 1, randomly generated, basic feasible point simple to derive
% min c'x, s.t. A_ineq*x <= b_ineq, x >= 0.
% n = 300; m = 900;
% c = randn(n,1);
% A_ineq = rand(m,n); x0 = ones(n,1);
% b_ineq = A_ineq*x0 + 5 + 10*rand(m,1);
% 
% A_eq = [A_ineq, eye(m)];
% b_eq = b_ineq;
% c_eq = [c;zeros(m,1)];
% B_ids = n+1:n+m;
% N_ids = 1:n;
% 
% tic
% [x,B_ids,N_ids,zs,flag] = simplex_method(c_eq,A_eq,b_eq,B_ids,N_ids);
% zs(end)
% toc
% 
% tic
% [x_lin,fval] = linprog(c,A_ineq,b_ineq,[],[],zeros(n,1)); fval     % pro kontrolu
% toc

%% Problem 2 - two phase method
% c = [3 1 1 0 0]';
% A = [2 1 1 1 0;1 -1 -1 0 -1];
% b = [-2;-1];
% [m,n] = size(A);
% E = eye(m);
% for i=1:m
%     if b(i) < 0
%         E(i,i) = -1;
%     end
% end
% 
% c_p1 = [0*c; ones(m,1)];
% A_p1 = [A,E];
% b_p1 = b;
% B_ids = n+1:n+m; N_ids = 1:n;
% 
% [x,B_ids,N_ids,zs,flag] = simplex_method(c_p1,A_p1,b_p1,B_ids,N_ids)
% 
% any(B_ids >= n+1)
% zs(end)

% N_ids = setdiff(1:n,B_ids)
% [x,B_ids,N_ids,zs,flag] = simplex_method(c,A,b,B_ids,N_ids)


%% Problem 3 - sparse matrix, unknown basic feasible point
% min c'x, s.t. A_ineq*x <= b_ineq, x >= 0.
load problem.mat;                                                  % loading the problem
A_ineq = problem.A; b_ineq = problem.b; c = problem.c;
[m,n] = size(A_ineq);
%plot_problem(problem)                                            % visualized problem

A_eq = [A_ineq, eye(m)]; 
b_eq = b_ineq;
c_eq = [c;zeros(m,1)];

E = eye(m);
for i=1:m
    if b_eq(i) < 0
        E(i,i) = -1;
    end
end
A_p1 = [A_eq,E];
b_p1 = b_eq;
c_p1 = [0*c_eq;ones(m,1)];

B_ids = [n+m+1:n+m+m]; N_ids = [1:n+m];

tic
[x,B_ids,N_ids,zs,flag] = simplex_method(c_p1,A_p1,b_p1,B_ids,N_ids);

any(B_ids >= n+m+1)
zs(end)

N_ids = setdiff(1:n+m,B_ids);

[x,B_ids,N_ids,zs,flag] = simplex_method(c_eq,A_eq,b_eq,B_ids,N_ids);
zs(end)
toc

tic
[x_lin,fval] = linprog(c,A_ineq,b_ineq,[],[],zeros(n,1)); fval   % pro kontrolu   
toc

%plot_problem(problem,x_lin)                                      % visualize solution

function [x,B_ids,N_ids,zs,flag] = simplex_method(c,A,b,B_ids,N_ids)
[m,n] = size(A);
B = A(:,B_ids); N = A(:,N_ids);
vareps = -1e-6;
x_B = B\b;
x = zeros(n,1); x(B_ids) = x_B;
iter = 1; zs(iter) = c'*x; flag = 0;
while true
    lambda = B'\c(B_ids);
    mu_N = c(N_ids) - N'*lambda;
    if all(mu_N >= vareps)
        flag = 1; 
        display('optimum found');
        break;
    end
    [~,q_ind] = min(mu_N);
    q = N_ids(q_ind);
    d = B\A(:,q);
    if all(d <= 0)
        flag = 2; %unbounded
        display('unbounded');
        iter = iter + 1;
        zs(iter) = -Inf;
        break;
    end
    d_i_positive = find(d > 0);
    ratios = x(B_ids(d_i_positive)) ./ d(d_i_positive);
    [x_q_plus, p_idx] = min(ratios);
    p = B_ids(d_i_positive(p_idx));
    x(B_ids) = x(B_ids) - d*x_q_plus;
    x(N_ids) = 0;
    x(q) = x_q_plus;
    B_ids = [B_ids(B_ids ~= p),q];
    N_ids = [N_ids(N_ids ~= q),p];
    B = A(:,B_ids); N = A(:,N_ids);
    iter = iter + 1;
    zs(iter) = c'*x;
end

end
