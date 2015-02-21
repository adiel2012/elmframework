function printExperimentgenerationsLog(generationsLog_file,itResultsTotal,nhidden) 

    if ~exist(generationsLog_file,'file')
        dlmwrite(generationsLog_file, '"nhidden"	"iter"	"itermax"	"CCRTrain Mean"	"CCRTrain StdDv"	"MSTrain Mean"	"MSTrain StdDv"	"CCRTest Mean"	"CCRTest StdDv"	"MSTest Mean"	"MSTest StdDv"	"Time Mean"	"Time StdDv"',...
        'delimiter', '');
    end

  
    % Foreach logged iteration
    for k = 1:size(itResultsTotal{1,1},1)
 
        % Foreach experiment repeitition
        for j = 1:size(itResultsTotal,1)
            %itResults = itResultsTotal{j,1};        
            tmp(:,j)=itResultsTotal{j,1}{k,1};
        end
        tmp = tmp';
        
        tmpMean = mean(tmp);
        tmpStd = std(tmp);
        tmpMeanFoo = [tmpMean 0];
        tmpStdFoo = [0 tmpStd];
        resultado = tmpMeanFoo + tmpStdFoo;
        
        resultado1 = resultado(1:2);
        resultado2 = resultado(3:11)*100;
        
        %resultado(,:)

        dlmwrite(generationsLog_file, [nhidden resultado1 resultado2],'-append', 'delimiter', '\t', ...
             'precision', 8);
         
        clear tmp tmpMean tmpStd tmpMeanFoo tmpStdFoo resultado;

    end
