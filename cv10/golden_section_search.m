function [a,b] = golden_section_search(f,a,b,n)
phi = (1+sqrt(5))/2;
rho = phi-1;
d = rho*b + (1-rho)*a;
yd = f(d); iter = 1;
while ~isreal(yd)
    b = d;
    d = rho*b + (1-rho)*a;
    yd = f(d);
    iter = iter + 1;
    if iter > n
        a = 0; b = 0;
        return
    end
end
for i=1:n-1
    c = rho*a + (1-rho)*b;
    yc = f(c);
    if ~isreal(yc)
        yc = inf;
    end
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

