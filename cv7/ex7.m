clear;clc;close all;


%% equality constrained entropy maximization
% n = 20; v1 = floor(n/4); v2 = floor(n/6); v3 = floor(n/3);
%  A = [ones(1,n);
%      ones(1,v1),zeros(1,n-v1);
%      ones(1,v2),zeros(1,n-v2); 
%      zeros(1,n-v3), (1:v3)/v3];
% b = [1;0.2;0.1;0.02]; 

% m = length(b);

% obj_f = @(x) sum(x.*log(x));
% obj_grad = @(x) log(x)+1;
% obj_hess = @(x) diag(1./(x));

% x0 = 1*ones(n,1); 
% xs = [x0];

% eps1 =  1e-6;
% eps2 = 1e-6;

% r_primal = @(x) A*x-b;
% r_dual = @(x,lambda) obj_grad(x) + A'*lambda;

% r = @(x,lambda) [r_dual(x,lambda); r_primal(x)];
% iter = 1;
% lambda = zeros(m,1);
% tic
% while norm(r_dual(xs(:,end),lambda)) > eps1 || norm(r_primal(xs(:,end))) > eps2
%     % Newton step
%     Hess = obj_hess(xs(:,end));
%     M = [Hess, A' ; A , zeros(m,m)];
%     RightHand = -r(xs(:,end),lambda);
    
%     result = M\RightHand;
    
%     delta_x_pd = result(1:n);
%     delta_lambda_pd = result(n+1:end);
%     %line search
%     f = @(alpha) norm(r(xs(:,end)+alpha*delta_x_pd, lambda+alpha*delta_lambda_pd));
%     [a_br,c_br] = bracket_minimum(f);
%     [a_br,c_br] = golden_section_search(f,a_br,c_br,30);
%     alpha = (a_br+c_br)/2;
%     %ulozeni novych hodnot
%     xs(:,end+1) = xs(:,end) + alpha*delta_x_pd;
%     lambda = lambda + alpha*delta_lambda_pd;
%     iter = iter + 1;
% end
% toc
% %pro n = 1400 to trva 1.05s

% % disp(xs(:,1))
% disp(xs(:,end))
% y1 = obj_f(xs(:,end));
% disp(y1)

% % pro kontrolu 
% [x2,y2] = fmincon(obj_f,x0,[],[],A,b);
% disp(x2)
% disp(y2)



%% Linear programming in inequality form - barrier

rng(1,'twister');
n = 20; m = 3*n;
c = randn(n,1);
A = rand(m,n); x0 = ones(n,1);
b = A*x0 + 5 + 10*rand(m,1);
A_ineq = [A;-eye(n)]; 
b_ineq = [b;zeros(n,1)];

Phi = @(x) -sum(log(b_ineq-A_ineq*x));
r = @(x) 1./(b_ineq-A_ineq*x);
Phi_grad = @(x) A_ineq'*r(x);
Phi_hess = @(x) A_ineq'*(diag(r(x))^2)*A_ineq;

xs = [x0]; t = 0.1; beta = 20; 
vareps = 1e-4;
iter = 1;

obj_f_nobarrier = @(x) c'*x;
obj_f = @(x,t) c'*x*t + Phi(x);
obj_grad = @(x,t) c*t + Phi_grad(x);
obj_hess = @(x) Phi_hess(x);


tic
while m/t > vareps
    t = t*beta;
    grad = obj_grad(xs(:,end),t);
    hess = obj_hess(xs(:,end));
    
    while true
        
        dir = -hess\grad; % Newton step
        if (dir'*hess*dir)^0.5 <vareps %newton decrement
            break
        end

        % line search
        f = @(alpha) obj_f(xs(:,end)+alpha*dir,t);
        [a_br,c_br] = bracket_minimum(f);
        [a_br,c_br] = golden_section_search(f,a_br,c_br,15);
        alpha = (a_br+c_br)/2;
        %ulozeni novych hodnot
        xs(:,end+1) = xs(:,end) + alpha*dir;
    
        grad = obj_grad(xs(:,end),t);
        hess = obj_hess(xs(:,end));
    end
    iter = iter + 1;
end
toc
disp(xs(:,end));
disp(obj_f_nobarrier(xs(:,end)));

% pro n > 62 to zacne casova narocnost prudce narustat (n=62, t = 1.1s)

[x_lin,fval] = linprog(c,A_ineq,b_ineq) % pro kontrolu


%% Quadratic programming in inequality form, SVM
% rng(1,'twister');
% n = 20;
% x_p = 0.4*randn(n,2) + 1;
% x_m = 0.7*randn(n,2) - 1;
% C = 1e-1; 

% P = zeros(3+2*n,3+2*n); P(1,1) = 1; P(2,2) = 1; 
% f = [zeros(3,1);
% C*ones(2*n,1)];
% b = -ones(2*n,1); A = [-x_p, -ones(n,1); x_m, ones(n,1)]; A = [A,-eye(2*n)];
% A_ineq = [A; zeros(2*n,3), -eye(2*n)]; b_ineq = [b;zeros(2*n,1)];

% Phi = @(x) -sum(log(b_ineq-A_ineq*x));
% r = @(x) 1./(b_ineq-A_ineq*x);
% Phi_grad = @(x) A_ineq'*r(x);
% Phi_hess = @(x) A_ineq'*(diag(r(x))^2)*A_ineq;

% m = 4*n;
% xs = [zeros(3,1);10*ones(2*n,1)]; 
% t = 0.1; beta = 20; 
% iter = 1;
% vareps = 1e-3;

% obj_f_nobarrier = @(x) (0.5*x'*P*x + f'*x);
% obj_f = @(x,t) (0.5*x'*P*x + f'*x)*t + Phi(x);
% obj_grad = @(x,t) (P*x + f)*t + Phi_grad(x);
% obj_hess = @(x,t) (P)*t + Phi_hess(x);

% tic
% while m/t > vareps
%         t = t*beta;
%         grad = obj_grad(xs(:,end),t);
%         hess = obj_hess(xs(:,end),t);

%         while true
%             dir = -hess\grad; % Newton step
%             if (dir'*hess*dir)^0.5 <vareps %newton decrement
%                 break
%             end
    
%             % line search
%             f_s = @(alpha) obj_f(xs(:,end)+alpha*dir,t);
%             [a_br,c_br] = bracket_minimum(f_s);
%             [a_br,c_br] = golden_section_search(f_s,a_br,c_br,30);
%             alpha = (a_br+c_br)/2;
%             %ulozeni novych hodnot
%             xs(:,end+1) = xs(:,end) + alpha*dir;
    
%             grad = obj_grad(xs(:,end),t);
%             hess = obj_hess(xs(:,end),t);
%         end
%         iter = iter + 1;
% end
% toc 

% disp(xs(:,end));
% disp(obj_f_nobarrier(xs(:,end)));

% %t = 1.04s pro n = 170

% [x_opt, fval] = quadprog(P,f,A_ineq,b_ineq)  % pro kontrolu