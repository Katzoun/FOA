clear;clc;close all;

% Problem 1 - Simulator
% load('prob1.mat');
% rng(1,'twister');
% x0 = [5;5;5;5;5];
% f_sim = @(x) simulator(x);
% out = simulator(x0);
% rho_count = 20; rho_quad = 20;

% f = @(x) objetive_func_penalized(x,rho_count,rho_quad);

% %cross entropy method params
% kmax = 100; 

% [xs_ce,iter_ce,mus,Sigmas,samples_all,samples_elite] = cross_entropy_method(f,x0,kmax);
% x_best = xs_ce(:,end);
% % out = simulator(x_best)
% % Hooke-Jeeves params
% par_eps = 0.01;
% [xs_hj,iter_hj,x_ns] = hooke_jeeves(f,x_best,par_eps,kmax);

% x_best_final = xs_hj(:,end);
% out = simulator(x_best_final)

% function val = objetive_func_penalized(x, rho_count, rho_quad)
%     UB = [10;10;10;10;10]; % upper bounds
%     LB = [0;0;0;0;0]; % lower bounds
%     x_c = [7;2;1;4;9]; % 
%     r = 5; % radius of the circle
    
%     val_obj = simulator(x);
%     UBpen = x - UB;
%     LBpen = LB - x;
%     CirclePen = norm(x-x_c) - r;

%     penalty_count = sum(UBpen > 0) + sum(LBpen > 0) + (CirclePen > 0);
%     penalty_quad =  sum(max(UBpen,0).^2) + sum(max(LBpen,0).^2) + (max(CirclePen,0))*CirclePen^2;

%     val = val_obj + rho_count*penalty_count + rho_quad*penalty_quad;
%     end



%% Problem 2 - Algorithm for convex problems
% load('prob2.mat')


% g1 =  @(x) x'*P1*x - 1;
% g2 =  @(x) x'*P2*x - 1;

% Phi = @(x) -log(-g1(x)) - log(-g2(x));
% Phi_grad = @(x) (2*P1*x)/(1 - x'*P1*x) + (2*P2*x)/(1 - x'*P2*x);

% Phi_hess = @(x) (1/(g1(x)^2))*((2*P1*x)/(1 - x'*P1*x))*((2*P1*x)/(1 - x'*P1*x))' + (1/(g2(x)^2))*((2*P2*x)/(1 - x'*P2*x))*((2*P2*x)/(1 - x'*P2*x))' ...
% - ((1/g1(x))*(2*P1) + 1/(g2(x))*(2*P2));

% xs = [0;0];
% m = 2;
% t = 0.1; beta = 20; 
% vareps = 1e-3;
% vareps_inner = 1e-3; 

% opt_alphas = cell(1);
% iter_out = 1; iters_in = 0;
% tic

% while m/t > vareps
%     t = beta*t;
    
%     %obj_f = @(x) t*(x'*eye(2)*x-2*x_0'*x) + Phi(x);
%     obj_f = @(x) t*((x-x_0)'*(x-x_0)) + Phi(x);


%     obj_grad = @(x) t*(2*x-2*x_0) + Phi_grad(x);
%     obj_hess = @(x) t*(2) + Phi_hess(x); 
    
%     g = obj_grad(xs(:,end));
%     H = obj_hess(xs(:,end));
%     iter_in = 1;
%     while 1
%         d = -H\g;
%         if d'*H*d < vareps_inner
%             break
%         end
%         f_red = @(alpha) obj_f(xs(:,end) + d*alpha);
%         [b_left, b_right] = bracket_minimum(f_red);
%         [b_left, b_right] = golden_section_search(f_red,b_left, b_right,20);
%         alpha = (b_left + b_right)/2;
%         opt_alphas{iter_out,1}(iter_in,1) = alpha;
%         xs(:,end+1) = xs(:,end) + alpha*d;
%         g = obj_grad(xs(:,end));
%         H = obj_hess(xs(:,end));
%         iter_in = iter_in + 1;
%     end
%     iters_in(iter_out) = iter_in;
%     iter_out = iter_out + 1;
% end
% toc
% x_best = xs(:,end);



% fvalbest = (x_best-x_0)'*(x_best-x_0);
% disp('fvalbest');
% % norm(x_best-x_0)^2;

% sum(iters_in);


%% Problem 3 - Pwo phase simplex method
% vars: x1, x2+, x2-, x3, x4+, x4-, z1, z2    
c = [7 4 -4 -3 1 -1 0 0 ]';

A = [3 2 -2 0 1 -1 0 0; -1 0 0 3 0 0 1 0; -2 -4 4 1 0 0 0 1];
b = [10;-5;-2];


[m,n] = size(A);
E = eye(m);


for i=1:m
    if b(i) < 0
        E(i,i) = -1;
    end
end

c_p1 = [0*c; ones(m,1)];
A_p1 = [A,E];
b_p1 = b;
B_ids = n+1:n+m; N_ids = 1:n;

[x,B_ids,N_ids,zs,flag] = simplex_method(c_p1,A_p1,b_p1,B_ids,N_ids)


any(B_ids >= n+1)
zs(end)

N_ids = setdiff(1:n,B_ids)
[x,B_ids,N_ids,zs,flag] = simplex_method(c,A,b,B_ids,N_ids)

% disp('Kontrola pomoci linprog')
% tic
% [x_lin,fval] = linprog(c,[],[],A,b,zeros(n,1)); fval     % pro kontrolu
% toc



%% funkce     
function [xs,iter,mus,Sigmas,samples_all,samples_elite] = cross_entropy_method(f,x0,max_iter)
    load('prob1.mat');
    xs = [x0]; n = length(x0);
    m = 40; m_elite = 10;
    mu = x0; Sigma = 20*eye(n);
    mus = x0; Sigmas = Sigma;
    y0 = f(x0); fmin = y0;
    iter = 1; 
    samples_all = [];
    samples_elite = [];
    
    for k=1:max_iter
        samples = mvnrnd(mu,Sigma,m)';
        samples_all(:,:,end+1) = samples;
        for i=1:m
            y(i) = f(samples(:,i));
        end
        [y,s] = sort(y,"ascend"); P = samples(:,s(1:m_elite));
        samples_elite(:,:,end+1) = P;
        if y(1) < fmin
            fmin = y(1);
            xs(:,end+1) = P(:,1);
        end
        mu_new = (sum(P,2))/(m_elite);
        mu = mu_new;
        Sigma = cov(P');
        Sigmas(:,:,end+1) = Sigma;
        mus(:,end+1) = mu;
        iter = iter + 1;
    end
        
    end

function [xs,iter,x_ns] = hooke_jeeves(f,x0,par_eps,max_iter)
    xs = [x0]; iter = 1; n = length(x0);
    D = eye(n); y = f(xs(:,end)); 
    gamma = 0.5; alpha_0 = 1;
    alpha = alpha_0;
    do_restart = false;
    x_ns = [];
    while alpha > par_eps || do_restart
        improved = false; x_best = xs(:,end); y_best = y;
        if alpha < par_eps && do_restart
            alpha = alpha_0;
        end
        cntr = 0;
        for i=1:n
            for sgn = [-1,1]
                x_n = xs(:,end) + sgn*alpha*D(:,i); y_n = f(x_n);
                cntr = cntr + 1;
                x_ns(:,cntr,iter) = x_n;
                if y_n < y_best
                    x_best = x_n; y_best = y_n; improved = true; 
                end
            end
        end
        xs(:,end+1) = x_best; y = y_best; 
        if ~improved
            alpha = alpha*gamma; 
        end
        if iter > max_iter
            break; 
        end
        iter = iter + 1;
    end
    
    
    end

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
                disp('optimum found');
                break;
            end
            [~,q_ind] = min(mu_N);
            q = N_ids(q_ind);
            d = B\A(:,q);
            if all(d <= 0)
                flag = 2; %unbounded
                disp('unbounded');
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
        