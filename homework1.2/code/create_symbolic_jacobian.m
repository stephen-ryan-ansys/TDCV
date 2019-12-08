function J = create_symbolic_jacobian(K)
syms X Y Z tx ty tz vx vy vz real

% % focal lengths (square pixels since fx = fy):
% fx = 2960.37845;
% fy = fx;
% % optical center:
% cx = 1841.68855;
% cy = 1235.23369;
% % axis skew:
% s = 0;
% % intrinsic matrix
% K = [ fx,  s, cx;
%        0, fy, cy;
%        0,  0,  1];

% Rodrigues
th = sqrt(vx^2 + vy^2 + vz^2);
omega = [  0, -vz,  vy;
          vz,   0, -vx;
         -vy,  vx,   0];
R = eye(3) + (sin(th)/th)*omega + ((1-cos(th))/th^2)*(omega^2);

t = [tx; ty; tz];

proj = K*[R, t]*[X; Y; Z; 1];
u = proj(1)/proj(3);
v = proj(2)/proj(3);

J_uv = jacobian([u; v], [vx, vy, vz, tx, ty, tz]);
J = matlabFunction(J_uv, 'File', 'symbolic_jacobian');
end