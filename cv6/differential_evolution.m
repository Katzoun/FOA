function [xs,iter,pops] = differential_evolution(f,x0,max_iter,max_fcalls)
global function_calls;
n = length(x0); 
m = 10; 
w = 0.8; p = 0.9;
iter = 1;
xs = x0; fmin = f(x0);
pop = [x0+randn(n,m)]; 
pops = pop;
for i=1:m
    val(i) = f(pop(:,i));
end
for k=1:max_iter
    children = zeros(2,m);
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
        val_child = f(children(:,i));
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
    if function_calls > max_fcalls
        break;
    end
    iter = iter + 1;
end

end

