dirs = dir(fullfile(pwd, 'experiments'));
%pwd

numsubfolders = size(dirs,1);

encabezados = ['EE.elmAlgorithm';'EE.dbName      ';'dbPreffixTrain ';'dbPreffixTest  ';'EE.XVMin       ';'EE.XVMax       ';'EE.wMin        ';'EE.wMax        ';'preprocessData ';'EE.nnbase      ';'Population     ';'Hidden nodes   ';'Iterations     ';'EE.refresh     ';'EE.CR          ';'EE.F           ';'EE.CR          ';'EE.strategy    ';'EE.tolerance   ';'TrainTimemean  ';'TrainTime_std  ';'TotalTime      ';'CCRTrain_mean  ';'CCRTrain_std   ';'MSTrain_mean   ';'MSTrain_std    ';'CCRTest_mean   ';'CCRTest_std    ';'MSTest_mean    ';'MSTest_std     '];   % faltan

[numcols f] = size(encabezados);



fidw = fopen(strcat(pwd,'\resumen.csv'),'w+'); 

for i=1 : numcols
    fprintf(fidw,strcat(encabezados(i,:),';'));
    %encabezados(i,:)
end
fprintf(fidw,'\n');

for i = 3 : numsubfolders
    %dirs(i).name
    files = dir(fullfile(pwd, 'experiments', dirs(i).name));
    numfiles = size(files,1);
    
    for j = 3 : numfiles
        %files(j).name
        strcat(pwd,'\experiments\',dirs(i).name,'\',files(j).name);
        fid = fopen(strcat(pwd,'\experiments\',dirs(i).name,'\',files(j).name));      
        
       [encontro b] = size(findstr('result',files(j).name));
        if (encontro ~= 0)
            name = files(j).name;
           pos =  findstr('_',files(j).name);
           %ff= cellstr(name);
            datas = strsplit(name,'_');
            dataset=datas(1);
            
             while ~feof(fid)
                tline = fgets(fid);
                temp = strsplit(tline,':');
                if size(temp,2)== 2
                    val=temp(2);
                    %disp(val)
                    %disp(',');
                    fprintf(fidw,strcat(val{1},';'));
                    %fwrite(fidw,val{1});
                end
             end
            fprintf(fidw,'\n');
        end
        
       
        
        fclose(fid);
        
    end
    
end


fclose(fidw);