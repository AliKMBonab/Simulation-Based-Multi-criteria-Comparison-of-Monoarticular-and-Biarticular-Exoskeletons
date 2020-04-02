clear
clc
p = [1 1 1; 2 0 1; 2 -1 1; 1, 1, 0];
x =  bsxfun( @le, p(1,:), p );
[f, idxs] = ParetoFront(p)
