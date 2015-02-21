function s = softmaxnn(X)

t = sum(exp(Y_class'));
ind=ones(1,size(Y_class,2));
pt = t(ind,:)';

% extend matrix
pc = exp(Y_class);

s = pc ./ pt;
