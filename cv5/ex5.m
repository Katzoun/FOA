clear;clc;

t = linspace(0,2*pi);

%constraint
constraint = @(x) -x(1)^2 -x(2)^2 + 2;
constraint_grad = @(x) [-2*x(1); -2*x(2)];

%penalty functions
pcount = @(x) (constraint(x)>0); %pokud bod lezi v kruhu tak je 1 (penalizace), pokud lezi mimo tak je 0
pquad = @(x) max(constraint(x),0)^2; %pokud bod lezi v kruhu tak je penalizace
pquad_grad = @(x) 2*max(constraint(x),0)*constraint_grad(x);

p_lagrange = @(x,rho,lambda)      0.5*rho*(constraint(x)^2) - lambda*constraint(x);
p_lagrange_grad = @(x, rho,lambda)  rho*constraint(x)*constraint_grad(x) - lambda*constraint_grad(x);

%functions to evaluate
func = @(x) flower(x);
func_grad = @(x) flower_grad(x);

% func = @(x) rosenbrock(x);
% func_grad = @(x) rosenbrock_grad(x);


%% direct method: cyclic coordinate search with penalty rho = [0.5, 1, 5, 10].

% x0 = [-2;-2];
% rho = [0.5,1,5,10];

% for k=1:length(rho)
    
%     figure()
%     f_penalized = @(x) func(x) + rho(k)*pcount(x);
%     plot_function(f_penalized, [-4, 4], [0, 0]);
    
%     plot(sqrt(2)*sin(t),sqrt(2)*cos(t),'b-');

%     [cyclic_points, cyclic_iter, cyclic_fcalls] = cyclic_coordinate_search(f_penalized, x0, 1e-4, 1e2, 1e3);
%     str = strcat('Cyclic coordinate search, f calls = ',num2str(cyclic_fcalls),', best f = ', num2str(func(cyclic_points(:,end))),', iters = ', num2str(cyclic_iter), ', rho = ', num2str(rho(k)));

%     plt1 = plot(cyclic_points(1,:),cyclic_points(2,:),'r','Linewidth',1,'DisplayName',str);
%     l = legend(plt1);
%     l.FontSize = 10;
    
% end


%% stochastic method (MADS) with the count penalty function, rho  = [10].

% x0 = [-2;-2];
% rho = [10];

% f_penalized = @(x) func(x) + rho*pcount(x);
% plot_function(f_penalized, [-4, 4], [0, 0]);
% plot(sqrt(2)*sin(t),sqrt(2)*cos(t),'b-');

% [mads_points, mads_iter, mads_fcalls] = mesh_adaptive_direct_search(f_penalized, x0, 1e-4, 1e2, 1e3);
% str = strcat('Mesh adaptive direct search, f calls = ',num2str(mads_fcalls),', best f = ', ...
%         num2str(func(mads_points(:,end))),', iters = ', num2str(mads_iter));
% plt2 = plot(mads_points(1,:),mads_points(2,:),'r','Linewidth',1,'DisplayName',str);
% l = legend(plt2);
% l.FontSize = 10;

%% The BFGS Quasi-Newton method with the quadratic penalty, rho = [0.01, 0.5, 1, 4, 8, 16]

x0 = [-2;-2];
rho = [0.01, 0.5, 1, 4, 8, 16];

for k=1:length(rho)
    
    figure()
    f_penalized_quad = @(x) func(x) + rho(k)*pquad(x);
    f_penalized_quad_grad = @(x) func_grad(x) + rho(k)*pquad_grad(x);
    
    plot_function(f_penalized_quad, [-4, 4], [0, 0]);
    plot(sqrt(2)*sin(t),sqrt(2)*cos(t),'b-');

    %%newtons method
    [newtons_points, newtons_iter, newtons_fcalls] = newtons_method_BFGS(f_penalized_quad, f_penalized_quad_grad, x0, 1e-3, 1e2, 1e3);
    str = strcat('Newtons method BFGS, f calls = ',num2str(newtons_fcalls),', best f = ', ...
            num2str(func(newtons_points(:,end))),', iters = ', num2str(newtons_iter), ', rho = ', num2str(rho(k)));
    plt3 = plot(newtons_points(1,:),newtons_points(2,:),'r','Linewidth',1,'DisplayName',str);

    l = legend(plt3);
    l.FontSize = 10;

end

%% The BFGS Quasi-Newton method with the Augmented Lagrange Method (inequality constraint is changed into equality)
 
% x0 = [-2;-2];
% rho = [0.01, 0.5, 1, 4, 8, 16];
% lambda = 0;

% for k=1:length(rho)
    
%     clf;

%     f_penalized_lag= @(x) func(x) + p_lagrange(x, rho(k), lambda);
%     f_penalized_lag_grad = @(x) func_grad(x) + p_lagrange_grad(x, rho(k), lambda);

%     plot_function(f_penalized_lag, [-4, 4], [0, 0]);
%     plot(sqrt(2)*sin(t),sqrt(2)*cos(t),'b-');

%     %%newtons method
%     [newtons_points, newtons_iter, newtons_fcalls] = newtons_method_BFGS(f_penalized_lag, f_penalized_lag_grad, x0, 1e-3, 100, 1e4);
%     str = strcat('Newtons method BFGS with Augmented Lagrange, f calls = ',num2str(newtons_fcalls),', best f = ', ...
%             num2str(func(newtons_points(:,end))),', iters = ', num2str(newtons_iter), ', rho = ', num2str(rho(k)));
%     plt3 = plot(newtons_points(1,:),newtons_points(2,:),'r*','Linewidth',1,'DisplayName',str);
%     l = legend(plt3);
%     l.FontSize = 10;

%     x0 = newtons_points(:,end);
%     lambda = lambda - rho(k)*constraint(x0);
%     pause(1);
% end





% for debugging purposes
function [val] = rosenbrock(x)
    a = 1; b = 5;
    val = (a-x(1))^2 + b*(x(2)-x(1)^2)^2;
    end

function [val]= rosenbrock_grad(x)
    a = 1;
    b = 5;
    val = [2*(x(1)-a) - 4*b*x(1)*(x(2)-x(1)^2); 
            2*b*(x(2)-x(1)^2)];
end

function val = flower(x)
     a = 1;
     b = 1;
     c = 4;  
     val = a * norm(x) + b * sin(c * atan2(x(2), x(1)));
end

function val = flower_grad(x)
    a = 1;
    b = 1;
    c = 4;
    val = [(a*x(1)/(abs(x(1))^2 + abs(x(2))^2)^(1/2)) - (b*c*x(2)*cos(c*atan2(x(2), x(1))))/(x(1)^2 + x(2)^2);
           (a*x(2)/(abs(x(1))^2 + abs(x(2))^2)^(1/2)) + (b*c*x(1)*cos(c*atan2(x(2), x(1))))/(x(1)^2 + x(2)^2)];
end

function plot_function(f, range, opt)

    x1 = linspace(range(1),range(2),200);
    x2 = linspace(range(1),range(2),200);
    [x1,x2] = meshgrid(x1,x2);
    Zs = zeros(size(x1));
    for i=1:size(x1,1)
        for j=1:size(x1,2)
            Zs(i,j) = f([x1(i,j),x2(i,j)]);
        end
    end
    contour(x1,x2,Zs,20); 

    hold on; 
    grid on; 
    axis equal;
    xlim(range)
    ylim(range)
    plot3(opt(1),opt(2),0,'g*'); 
end

function [xs,iter, f_calls] = newtons_method_BFGS(f,f_grad,x0,grad_lim_size,max_iter,max_fcalls)

    f_calls = 0; 
    nr_f_calls_line_search = 7;

    xs = [x0]; 
    iter = 1;
    Q = eye(2);

    grad  = f_grad(x0);
    f_calls = f_calls + 1;
    grad_size = norm(grad);

    while (grad_size > grad_lim_size) && (f_calls < max_fcalls) && (iter < max_iter)
        if iter == 1
            dir = -grad;
        else
            gamma = grad - grad_prev;
            delta = xs(:,end) - xs(:,end-1);
            
            % DFP
            %Q = Q - (Q*gamma)*(gamma'*Q)/(gamma'*Q*gamma)+(delta*delta')/(delta'*gamma);
            
            %BFGS
            Q = Q - (delta*gamma'*Q + Q*gamma*delta')/(delta'*gamma)+ (1+ gamma'*Q*gamma/(delta'*gamma))*(delta*delta')/(delta'*gamma);
        
            dir = -Q*grad;
        end

        f_red = @(alpha) f(xs(:,end) + dir*alpha);

        [a,c,~,~,fcalls_new] = bracket_minimum(f_red); 
        f_calls = f_calls + fcalls_new;

        [a,b,fcalls_new] = golden_section_search(f_red,a,c,nr_f_calls_line_search);
        f_calls = f_calls + fcalls_new;

        alpha = (a+b)/2;
        x_new = xs(:,end) + alpha*dir;
        

        xs(:,end+1) = x_new;
        grad_prev = grad;
        grad = f_grad(x_new);
        f_calls = f_calls + 1;
        grad_size = norm(grad);
        iter = iter + 1;
    end
end


function [xs,iter, fcalls] = mesh_adaptive_direct_search(f,x0,step_size_lim_mad,max_iter,max_fcalls)
    fcalls = 0;
    xs = [x0]; 
    alpha = 1; 
    y = f(x0); 
    n = length(x0); 
    x = x0;
    iter = 1;

    while alpha > step_size_lim_mad && iter < max_iter && fcalls < max_fcalls
        improved = false;
        D = rand_positive_spanning_set(alpha,n);
        for i=1:3
            d = D(:,i); 
            x_p = x+alpha*d; 
            y_p = f(x_p);
            fcalls = fcalls + 1;

            if y_p < y
                x = x_p; y = y_p; improved = true;
                x_p = x+3*alpha*d; 
                y_p = f(x_p);
                fcalls = fcalls + 1;

                if y_p < y
                    x = x_p; y = y_p;
                end
                break
            end
        end
        if improved
            alpha = min(4*alpha,1);
        else
            alpha = alpha/4;
        end
        xs(:,end+1) = x;

        iter = iter + 1;
    end
    end
    
    function [D] = rand_positive_spanning_set(alpha,n)
    delta = round(1/sqrt(alpha));
    L = diag(delta*(2*rand(n,1)-1));
    for i=1:n-1
        for j=i+1:n
            L(i,j) = -delta+1 +(2*delta+2)*rand;
        end
    end
    
    D = L; D = D(randperm(n),:); D = D(:,randperm(n));
    D = [D, -sum(D,2)];
    end

 function [xs,iter, fcalls] = cyclic_coordinate_search(f,x0,par_eps,max_iter,max_fcalls)
    fcalls = 0;
    xs = [x0]; 
    iter = 1; 
    Delta = inf; 
    n = length(x0); 
    D = eye(n);
    nr_f_calls_line_search = 7;
    acceleration_step = true;

    while abs(Delta) > par_eps
        x_1 = xs(:,end);
        for i=1:n
            d = D(:,i);
            f_red = @(alpha) f(xs(:,end)+d*alpha);
            [a,c,~,~,fcalls_new] = bracket_minimum(f_red);
            fcalls = fcalls + fcalls_new;

            [a,b,fcalls_new] = golden_section_search(f_red,a,c,nr_f_calls_line_search);
            fcalls = fcalls + fcalls_new;

            alpha = (a+b)/2; 
            x_new = xs(:,end) + alpha*d;
            xs(:,end+1) = x_new; 
        end

        if acceleration_step
            d = xs(:,end) - x_1;
            f_red = @(alpha) f(xs(:,end)+d*alpha);
            [a,c,~,~,fcalls_new] = bracket_minimum(f_red); 
            fcalls = fcalls + fcalls_new;

            [a,b,fcalls_new] = golden_section_search(f_red,a,c,nr_f_calls_line_search);
            fcalls = fcalls + fcalls_new;

            alpha = (a+b)/2; x_new = xs(:,end) + alpha*d; xs(:,end+1) = x_new; 
            Delta = norm(xs(:,end)-x_1);
            iter = iter + 1;
        end
        if iter > max_iter || fcalls> max_fcalls
           break; 
        end
    end
    
    end
    

 function [a,b,c,n,fcalls] = bracket_minimum(f,varargin)
    len_vararg = length(varargin);
    n = 0;
    switch len_vararg
        case 0
            x = 0; s = 1e-2; k = 2;
        case 1
            x = varargin{1}; s = 1e-2; k = 2;
        case 2
            x = varargin{1}; s = varargin{2};  k = 2;
        case 3 
            x = varargin{1}; s = varargin{2};  k = varargin{3};
    end
    a = x; ya = f(x); 
    b = a + s; yb = f(b); 
    n = n + 2;
    fcalls = 2;

    if yb > ya
        temp = a; a = b; b = temp;
        temp = ya; ya = yb; yb = temp;
        s = -s;
    end
    
    while true
        c = b + s; 
        yc = f(c); 
        fcalls = fcalls + 1;
        n = n + 1;

        if yc > yb
            if a < c
                return
            else
                temp = a; a = c; c = temp;
                return
            end
        end
        a = b; ya = yb; b = c; yb = yc;
        s = k*s;
    end
    
    end

 function [a,b,fcalls] = golden_section_search(f,a,b,n)
    phi = (1+sqrt(5))/2;
    rho = phi-1;
    d = rho*b + (1-rho)*a;
    yd = f(d);
    fcalls = 1;
    for i=1:n-1
        c = rho*a + (1-rho)*b;
        yc = f(c);
        fcalls = fcalls + 1;
        if yc < yd
            b = d; d = c; yd = yc;
        else
            a = b; b = c;
        end
    end
    if a < b
        return
    else
        temp = a; a = b; b = temp;
        return
    end
    
    end