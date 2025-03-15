function val = flower(x)
    a = 1;
    b = 1;
    c = 4;  
    val = a * norm(x) + b * sin(c * atan2(x(2), x(1)));
end