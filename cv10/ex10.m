clear;clc;close all;

%% Problem 1 - Simulator
 load('prob1.mat')

x0 = [5;5;5;5;5];
f_sim = @(x) simulator(x);
out = simulator(x0);
rho_count = 20; rho_quad = 20;

f = @(x) objetive_func_penalized(x,rho_count,rho_quad);

%cross entropy method params
kmax = 100; 

[xs_ce,iter_ce,mus,Sigmas,samples_all,samples_elite] = cross_entropy_method(f,x0,kmax);
x_best = xs_ce(:,end);
% out = simulator(x_best)
% Hooke-Jeeves params
par_eps = 0.01;
[xs_hj,iter_hj,x_ns] = hooke_jeeves(f,x_best,par_eps,kmax);

x_best_final = xs_hj(:,end);
out = simulator(x_best_final)

function val = objetive_func_penalized(x, rho_count, rho_quad)
    UB = [10;10;10;10;10]; % upper bounds
    LB = [0;0;0;0;0]; % lower bounds
    x_c = [7;2;1;4;9]; % 
    r = 5; % radius of the circle
    
    val_obj = simulator(x);
    UBpen = x - UB;
    LBpen = LB - x;
    CirclePen = norm(x-x_c) - r;

    penalty_count = sum(UBpen > 0) + sum(LBpen > 0) + (CirclePen > 0);
    penalty_quad =  sum(max(UBpen,0).^2) + sum(max(LBpen,0).^2) + (max(CirclePen,0))*CirclePen^2;

    val = val_obj + rho_count*penalty_count + rho_quad*penalty_quad;
    end










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