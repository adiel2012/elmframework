%load iris.dat
%cv = cvpartition(iris(:, 1), 'holdout', 0.25);
%text_mat = iris(cv.test, :);
%cv
%find(cv.test==1)


load fisheriris;
y = species;
c = cvpartition(y,'k',5)
[c.test(1) c.test(2) c.test(3) c.test(4) c.test(5)]