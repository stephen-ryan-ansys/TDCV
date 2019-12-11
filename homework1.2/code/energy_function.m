function [energy, W, e] = energy_function(A, R, t, M, m)

bisquare_weights = @(e, c) (abs(e)<c).*(1 - e.^2/(c^2)).^2;
bisquare_func = @(e, c) (c^2)/6 - (abs(e)<=c).*((c^2)/6).*(1-(e./c).^2).^3;
% bisquare_weights = @(e, c) (e<c).*(1 - e.^2/(c^2)).^2;
% bisquare_func = @(e, c) (c^2)/6 - (e<=c).*((c^2)/6).*(1-(e./c).^2).^3;

% Project 3D matches to image i+1 using current guess
proj = worldToImage(A, R, t, M);
% Compute the reprojection error
dist = proj - m;
e = [dist(:, 1)'; dist(:, 2)'];
% interleave the vectors [u1; v1; u2; v2; ...]:
e = e(:);
sigma = 1.48257968*median(abs(e));
c = 4.685;
w = bisquare_weights(e/sigma, c);
W = diag(w);
rho = bisquare_func(e, c);
energy = sum(rho);
end