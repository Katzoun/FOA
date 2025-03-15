function val = flower_grad(x)
    a = 1;
    b = 1;
    c = 4;
    val = [(a*x(1)/(abs(x(1))^2 + abs(x(2))^2)^(1/2)) - (b*c*x(2)*cos(c*atan2(x(2), x(1))))/(x(1)^2 + x(2)^2);
           (a*x(2)/(abs(x(1))^2 + abs(x(2))^2)^(1/2)) + (b*c*x(1)*cos(c*atan2(x(2), x(1))))/(x(1)^2 + x(2)^2)];
end