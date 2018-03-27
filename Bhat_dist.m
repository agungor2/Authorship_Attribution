function [dist] = Bhat_dist(sigma1,sigma2,mean1,mean2)
dist = 1/4*log(1/4*(sigma1/sigma2 + sigma2/sigma1 + 2))+1/4*((mean1-mean2)^2/(sigma1+sigma2));

end