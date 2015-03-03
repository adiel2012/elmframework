function [chi, df] = chi2feature(feature, label)
% chi squared feature measure for binary class
% find class name
class = unique(label);
rows = size(feature, 1);
dim = size(feature, 2);
ct = cell(dim, 1);

for j = 1:dim
    member = unique(feature(:, j));
    % find number of each member
    for k = 1:size(member, 1)
        n_member(k, 1) = length(find(feature(:, j) == member(k, 1)));
    end
    % find partioned class of each member
    for p = 1:size(member, 1)
        % find index
        [xInd, yInd] = find(feature(:, j) == member(p, 1));
        % partioned class
        p1_member = length(find(label(xInd) == class(1, 1)));
        p2_member = length(find(label(xInd) == class(2, 1)));
        % negative class is 1st column, positive is 2nd
        ct{j, 1}(p, 1) = p1_member;
        ct{j, 1}(p, 2) = p2_member;
    end
    clear n_member;
end

% compute chi squared 
for i = 1:dim
    member = unique(feature(:, i));
    
    for j = 1:size(member, 1)
        n_fstar = ct{i, 1}(j, 1) + ct{i, 1}(j, 2);
        n = rows;
        % negative class
        n_in = ct{i, 1}(j, 1);
        n_starn = sum(ct{i, 1}(:, 1));
        mu_in = (n_starn*n_fstar)/n;

        negPart = (n_in-mu_in)^2/mu_in;
        % positive class
        n_ip = ct{i, 1}(j, 2);
        n_starp = sum(ct{i, 1}(:, 2));
        mu_ip = (n_starp*n_fstar)/n;

        posPart = (n_ip-mu_ip)^2/mu_ip;
        
        negPos(j, 1) = negPart + posPart;
    end
    % degree of freedom
    df(i, 1) = size(member, 1) - 1;
    % chi squared value, needed to compare with the threshold of
    % corresponding significance value
    chi(i, 1) = sum(negPos(:, 1));
    clear negPos
end


% http://www.mathworks.com/matlabcentral/fileexchange/28012-chi-square-feature-analysis/content/chi2feature.m


    
    
    
    
    
    
    
    
    
    
    