load iris.dat
cv = cvpartition(iris(:, 1), 'holdout', 0.25);
text_mat = iris(cv.test, :);

find(cv.test==1)