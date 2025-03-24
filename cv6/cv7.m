clear;clc;close all;

M1_area = 18; M1_typ = 4; M1_tlak_perm = 5;
M2_area = 4; M2_typ = 4; M2_tlak_perm = 38.22;

M3_area = 4; M3_typ = 4;
C1_tlak = 318.515;
C2_tlak = 146.764;

lower_bounds = [  2, 0.501,   5,   2, 0.501,   5,   2, 0.501, 100, 100]';
upper_bounds = [200, 5.499, 100, 200, 5.499, 100, 200, 5.499, 500, 500]';

output_lb = [0, 6, -100, -100]';
output_ub = [0,30,-95,-97]';

%input limits
M1_area_constr_lb = @(x) lower_bounds(1) - x(1);
M1_area_constr_ub = @(x) x(1) - upper_bounds(1);

M1_typ_constr_lb = @(x) lower_bounds(2) - x(2);
M1_typ_constr_ub = @(x) x(2) - upper_bounds(2);

M1_tlak_perm_constr_lb = @(x) lower_bounds(3) - x(3);
M1_tlak_perm_constr_ub = @(x) x(3) - upper_bounds(3);

M2_area_constr_lb = @(x) lower_bounds(4) - x(4);
M2_area_constr_ub = @(x) x(4) - upper_bounds(4);

M2_typ_constr_lb = @(x) lower_bounds(5) - x(5);
M2_typ_constr_ub = @(x) x(5) - upper_bounds(5);

M2_tlak_perm_constr_lb = @(x) lower_bounds(6) - x(6);
M2_tlak_perm_constr_ub = @(x) x(6) - upper_bounds(6);

M3_area_constr_lb = @(x) lower_bounds(7) - x(7);
M3_area_constr_ub = @(x) x(7) - upper_bounds(7);

M3_typ_constr_lb = @(x) lower_bounds(8) - x(8);
M3_typ_constr_ub = @(x) x(8) - upper_bounds(8);

C1_tlak_constr_lb = @(x) lower_bounds(9) - x(9);
C1_tlak_constr_ub = @(x) x(9) - upper_bounds(9);

C2_tlak_constr_lb = @(x) lower_bounds(10) - x(10);
C2_tlak_constr_ub = @(x) x(10) - upper_bounds(10);

%output limits


%area - second output parameter
area_constr_lb = @(out) output_lb(2) - out(2);
area_constr_ub = @(out) out(2) - output_ub(2);
%recovery - third output parameter
recovery_constr_lb = @(out) output_lb(3) - out(3);
recovery_constr_ub = @(out) out(3) - output_ub(3);
%purity - fourth output parameter
purity_constr_lb = @(out) output_lb(4) - out(4);
purity_constr_ub = @(out) out(4) - output_ub(4);


pcount_input = @(x) sum([ M1_area_constr_lb(x) > 0, M1_area_constr_ub(x) > 0, M1_typ_constr_lb(x) > 0, ...
    M1_typ_constr_ub(x) > 0, M1_tlak_perm_constr_lb(x) > 0, M1_tlak_perm_constr_ub(x) > 0, ...
    M2_area_constr_lb(x) > 0, M2_area_constr_ub(x) > 0, M2_typ_constr_lb(x) > 0, M2_typ_constr_ub(x) > 0,...
    M2_tlak_perm_constr_lb(x) > 0, M2_tlak_perm_constr_ub(x) > 0, M3_area_constr_lb(x) > 0, ...
    M3_area_constr_ub(x) > 0,  M3_typ_constr_lb(x) > 0, M3_typ_constr_ub(x) > 0, C1_tlak_constr_lb(x) > 0,...
    C1_tlak_constr_ub(x) > 0, C2_tlak_constr_lb(x) > 0, C2_tlak_constr_ub(x) > 0, ...
    ]);

pquadratic_input = @(x) sum([max(M1_area_constr_lb(x),0)^2, max(M1_area_constr_ub(x),0)^2, max(M1_typ_constr_lb(x),0)^2, ...
    max(M1_typ_constr_ub(x),0)^2, max(M1_tlak_perm_constr_lb(x),0)^2, max(M1_tlak_perm_constr_ub(x),0)^2, ...
    max(M2_area_constr_lb(x),0)^2, max(M2_area_constr_ub(x),0)^2, max(M2_typ_constr_lb(x),0)^2, max(M2_typ_constr_ub(x),0)^2,...
    max(M2_tlak_perm_constr_lb(x),0)^2, max(M2_tlak_perm_constr_ub(x),0)^2, max(M3_area_constr_lb(x),0)^2, ...
    max(M3_area_constr_ub(x),0)^2,  max(M3_typ_constr_lb(x),0)^2, max(M3_typ_constr_ub(x),0)^2, max(C1_tlak_constr_lb(x),0)^2,...
    max(C1_tlak_constr_ub(x),0)^2, max(C2_tlak_constr_lb(x),0)^2, max(C2_tlak_constr_ub(x),0)^2, ...
    ]);

pcount_output = @(out)  sum([area_constr_lb(out) > 0, area_constr_ub(out) > 0, ...
    recovery_constr_lb(out) > 0, recovery_constr_ub(out) > 0,...
    purity_constr_lb(out) > 0, purity_constr_ub(out) > 0]);

pquadratic_output = @(out)  sum([max(area_constr_lb(out),0)^2, max(area_constr_ub(out),0)^2, ...
    max(recovery_constr_lb(out),0)^2, max(recovery_constr_ub(out),0)^2,...
    max(purity_constr_lb(out),0)^2, max(purity_constr_ub(out),0)^2]);


x0 = [M1_area; M1_typ; M1_tlak_perm; M2_area; M2_typ; M2_tlak_perm; M3_area; M3_typ; C1_tlak; C2_tlak];

% x0 = [10; 2; 5; 10; 2; 5; 10; 2; 200; 200];
% out = simulator(x0(1),x0(2),x0(3),x0(4),x0(5),x0(6),x0(7),x0(8),x0(9),x0(10));

% disp(out)
% rho1 = 10; %tyto penalty taky funguji dobre
% rho2 = 20;

disp("initial point")
disp(x0')
disp(" ")

rho1 = 30;
rho2 = 20;

func = @(x) simulator_penalized(x, pcount_input, pcount_output, pquadratic_input, pquadratic_output ,rho1, rho2);
disp("starting optimization")

tic
% [points, mads_iter, mads_fcalls] = mesh_adaptive_direct_search(func, x0, 1e-4, 200, 1e4);
[points, de_iter, de_pops, de_fcalls] = differential_evolution(func, x0, 600, 1e4);
% [points, sa_iter, sa_fcalls] = simulated_annealing(func, x0, 2000, 1e5);

x_last = points(:,end);
out = simulator(x_last(1),x_last(2),x_last(3),x_last(4),x_last(5),x_last(6),x_last(7),x_last(8),x_last(9),x_last(10));
disp(" ")
toc
disp(" ")
disp("optimal solution")
disp(x_last')
disp("outputs for the optimal solution")
disp(out)


function y = simulator_penalized(x, pcount_input, pcount_output, pquadratic_input, pquadratic_output, rho1, rho2)

out = simulator(x(1),x(2),x(3),x(4),x(5),x(6),x(7),x(8),x(9),x(10));

y = out(1) + rho1*(pcount_input(x) +pcount_output(out)) + rho2*(pquadratic_input(x) + pquadratic_output(out));

end


function [xs,iter, fcalls] = mesh_adaptive_direct_search(f,x0,step_size_lim_mad,max_iter,max_fcalls)
fcalls = 0;
xs = [x0];
alpha = 1;

try
    y = f(x0);
catch
    y = 1e10;
end

n = length(x0);
x = x0;
iter = 1;

while alpha > step_size_lim_mad && iter < max_iter && fcalls < max_fcalls
    improved = false;
    D = rand_positive_spanning_set(alpha,n);
    for i=1:n
        d = D(:,i);
        x_p = x+alpha*d;
        
        try
            y_p = f(x_p);
        catch
            y_p = 1e10;
        end
        
        fcalls = fcalls + 1;
        
        if y_p < y
            x = x_p; y = y_p; improved = true;
            x_p = x+3*alpha*d;
            try
                y_p = f(x_p);
            catch
                y_p = 1e10;
            end
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



function [xs,iter,pops,fcalls] = differential_evolution(f,x0,max_iter,max_fcalls)
fcalls = 0;
n = length(x0);
m = 10;
w = 0.8; p = 0.9;
iter = 1;
xs = x0;
try
    fmin = f(x0);
catch
    fmin = 1e10;
end
fcalls = fcalls + 1;

pop = [x0+(randn(n,m))];
pops = pop;

for i=1:m
    try
        val(i) = f(pop(:,i));
        fcalls = fcalls + 1;
    catch
        val(i) = 1e10;
    end
end
for k=1:max_iter
    children = zeros(n,m);
    for i=1:m
        perm = randperm(m,3);
        z = pop(:,perm(1)) + w*(pop(:,perm(2)) - pop(:,perm(3)));
        j = randperm(n,1);
        children(:,i) = pop(:,i);
        for d = 1:n
            if d == j || p < rand(1)
                children(d,i) = z(d);
            end
        end
    end
    for i=1:m
        try
            val_child = f(children(:,i));
            fcalls = fcalls + 1;
        catch
            val_child = 1e10;
        end
        
        if val_child < val(i)
            pop(:,i) = children(:,i);
            val(i) = val_child;
            if val_child < fmin
                fmin = val_child;
                xs(:,end+1) = children(:,i);
            end
        end
    end
    pops(:,:,end+1) = pop;
    if fcalls > max_fcalls
        break;
    end
    iter = iter + 1;
end

end


function [xs,iter,x_ps,x_ps_selected,deltas,probs] = simulated_annealing(f,x0,max_iter,max_fcalls)
global function_calls;
xs = [x0]; iter = 1; n = length(x0); x_ps = x0;
deltas = []; probs = []; x_ps_selected = 1;
t0 = 800;
t = t0;

try
    y = f(x0);
catch
    y = 1e10;
end

x_best = x0; y_best = y; sigma0 = 1; sigma = sigma0;
for k=1:max_iter
    x = xs(:,end);
    x_p = x + sigma*randn(n,1);
    try
        y_p = f(x_p);
    catch
        y_p = 1e10;
    end
    
    x_ps(:,end+1) = x_p;
    delta = y_p - y;
    deltas(end+1) = delta;
    probs(end+1) = exp(-delta/t);
    
    if delta <= 0 || rand() < exp(-delta/t)
        x = x_p; y = y_p;
        x_ps_selected(end+1) = 1;
    else
        x_ps_selected(end+1) = 0;
    end
    if y_p < y_best
        x_best = x_p; y_best = y_p;
    end
    xs(:,end+1) = x_best;
    t = t0/k; %sigma = sigma0/k;
    %t = t0*(0.95^k); %sigma = sigma0/(k);
    if function_calls > max_fcalls
        break;
    end
    iter = iter + 1;
end


end
