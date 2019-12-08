function J = energy_jacobian(camera_params, R, t, M)
N = size(M, 1);
J = zeros(2*N, 6);

% R was returned so that [x, y, z] = [X, Y, Z]*R + [t_X, t_Y, t_Z];
% so need to use R' later when multiplying from left
P = [R; t]*camera_params.IntrinsicMatrix;
proj = [M ones(N, 1)]*P;

v = rotationMatrixToVector(R);
vx = skew(v);
dR1 = ((v(1)*vx + skew(cross(v', (eye(3) - R')*[1; 0; 0])))./norm(v)^2)*R';
dR2 = ((v(2)*vx + skew(cross(v', (eye(3) - R')*[0; 1; 0])))./norm(v)^2)*R';
dR3 = ((v(3)*vx + skew(cross(v', (eye(3) - R')*[0; 0; 1])))./norm(v)^2)*R';

for k = 1:N
    dM = [zeros(3), eye(3)];
    dM(:, 1) = dR1*M(k, :)';
    dM(:, 2) = dR2*M(k, :)';
    dM(:, 3) = dR3*M(k, :)';
    U = proj(k, 1);
    V = proj(k, 2);
    W = proj(k, 3);
    dm = [1/W,   0, -U/W^2;
            0, 1/W, -V/W^2];
    J((2*k-1):(2*k), :) = dm*camera_params.IntrinsicMatrix'*dM;
end
end

