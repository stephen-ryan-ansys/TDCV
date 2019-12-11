function q = rotation_matrix_to_quaternion(R)
% from https://www.eecis.udel.edu/~cer/arv/readings/old_mkss.pdf
t = acos((trace(R) - 1)/2);
if t ~= 0
    w = (1/(2*sin(t)))*[R(3, 2) - R(2, 3);
                        R(1, 3) - R(3, 1);
                        R(2, 1) - R(1, 2)];
else
    w = [0; 0; 0];
end
q = [cos(t/2); w*sin(t/2)];
end
