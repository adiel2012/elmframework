dirs = dir(fullfile(pwd, 'experiments'));
%pwd

numsubfolders = size(dirs,1);

for i = 3 : numsubfolders
    %dirs(i).name
    files = dir(fullfile(pwd, 'test', dirs(i).name));
    numfiles = size(files,1);
    for j = 3 : numfiles
        %files(j).name
        strcat(pwd,'\test\',dirs(i).name,'\',files(j).name);
        fid = fopen(strcat(pwd,'\test\',dirs(i).name,'\',files(j).name));      
        
        while ~feof(fid)
          tline = fgets(fid)
        end
        
        fclose(fid);
        
    end
    
end