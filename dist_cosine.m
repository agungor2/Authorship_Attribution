function [dist] = dist_cosine(A,B)
%dist = 1- dot(u,v)/(norm(u)*norm(v));
dist = 1-dot(A,B,2)./(sqrt(sum(A.^2, 2)) .* sqrt(sum(B.^2, 2)));
end

%Short version

