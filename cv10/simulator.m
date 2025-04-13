function [val] = simulator(x)
a = 20; b = 0.2; c = 2*pi;
d = length(x);
val = -a*exp(-b*sqrt(sum(x.^2))/d) - exp(sum(cos(c*x))/d) + a + exp(1);
end
