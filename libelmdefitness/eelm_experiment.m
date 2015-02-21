function EEStats = eelm_experiment(EE,train_data,test_data,logLevel)

TrainingTime = zeros(EE.repeatfold,1);
CCRTrain = zeros(EE.repeatfold,1);
MSTrain = zeros(EE.repeatfold,1);
CCRTest = zeros(EE.repeatfold,1);
MSTest = zeros(EE.repeatfold,1);
% The number of Neurons is a vector because of parfor issues
NumberofInputNeurons = zeros(EE.repeatfold,1);
NumberofHiddenNeurons = zeros(EE.repeatfold,1);
NumberofHiddenNeuronsFinal = zeros(EE.repeatfold,1);
NumberofOutputNeurons = zeros(EE.repeatfold,1);
ConfusionMatrixTrain = cell(EE.repeatfold,1);
ConfusionMatrixTest = cell(EE.repeatfold,1);
InputWeight = cell(EE.repeatfold,1);
OutputWeight = cell(EE.repeatfold,1);

if strcmp(EE.elmAlgorithm, 'eelm')
    itResultsTotal = cell(EE.repeatfold,1);
end

%%% Log files setup
if logLevel > 0
    % Create directory for logs if it doesn't exists
    if ~exist(EE.outputDir,'dir')
        mkdir(EE.outputDir);
    end
    
    performanceTableOutputPreffix = [EE.dbName '_' EE.nnbase ];
    if EE.opelm == 1 || EE.elm == 1
        bestModelOutputPreffix = [EE.dbName '_' EE.elmAlgorithm '_' EE.nnbase '_n' num2str(EE.nhidden)];
    else
        bestModelOutputPreffix = [EE.dbName '_' EE.elmAlgorithm '_' EE.nnbase '_n' num2str(EE.nhidden) '_p' num2str(EE.npop) '_i' num2str(EE.itermax) ];
    end
    
    if logLevel > 1
        %performanceTableOutputPreffix = [EE.dbName '_' EE.nnbase '_n'
        %num2str(EE.nhidden) '_p' num2str(EE.npop) '_i' num2str(EE.itermax) ];
        experimentResultOutput = [EE.outputDir '/' bestModelOutputPreffix '_results.txt'];
        bestOuput_file = [EE.outputDir '/' bestModelOutputPreffix '_best.txt'];
        bestConfusionMatrix_file = [EE.outputDir '/'  bestModelOutputPreffix '_ConfusionMatrix.txt'];
        experimentResultOutputDat = [EE.outputDir '/'  bestModelOutputPreffix '_table.dat'];
        experimentResultOutputXLS = [EE.outputDir '/'  bestModelOutputPreffix '_table.xls'];
        
        performanceTable = [EE.outputDir '/'  performanceTableOutputPreffix '_fitness_table.dat'];
        generationsLog_file = [EE.outputDir '/'  performanceTableOutputPreffix '_fitness_table_generations.dat'];
        
        % Avoid overwriting a file
        while exist(experimentResultOutput,'file') || exist(bestOuput_file,'file') ...
                || exist(bestConfusionMatrix_file,'file') || exist(experimentResultOutputDat,'file') ...
                || exist(experimentResultOutputXLS,'file')
            experimentResultOutput = [experimentResultOutput '-1'];
            bestOuput_file = [bestOuput_file '-1'];
            bestConfusionMatrix_file = [bestConfusionMatrix_file '-1'];
            experimentResultOutputDat  = [experimentResultOutputDat '-1'];
            experimentResultOutputXLS  = [experimentResultOutputXLS '-1'];
        end
    else
        
        crossVal_performanceTableOutputPreffix = [performanceTableOutputPreffix '_crossval'];
        crossVal_performanceTable = [EE.outputDir '/'  crossVal_performanceTableOutputPreffix  '_fitness_table.dat'];
        crossVal_generationsLog_file = [EE.outputDir '/'  crossVal_performanceTableOutputPreffix '_fitness_table_generations.dat'];
    end
end


tic;
%%% EXPERIMENT

if strcmp(EE.db, 'holdout')
    switch lower(EE.elmAlgorithm)
        case {'eelm'}
            if EE.multicore
                parfor it = 1:EE.repeatfold
                    [TrainingTime(it), ConfusionMatrixTrain{it,1}, ConfusionMatrixTest{it,1},...
                        CCRTrain(it), MSTrain(it), CCRTest(it), MSTest(it),...
                        NumberofInputNeurons(it),NumberofHiddenNeurons(it),NumberofOutputNeurons(it),InputWeight{it,1},OutputWeight{it,1},itResultsTotal{it,1}] = ...
                        eelm(train_data, test_data, 1, EE.nhidden, EE.nnbase, EE.wMin, EE.wMax, EE.CR, EE.F, EE.npop, EE.itermax,EE.refresh, EE.strategy,EE.tolerance);
                end
            else
                for it = 1:EE.repeatfold
                    [TrainingTime(it), ConfusionMatrixTrain{it,1}, ConfusionMatrixTest{it,1},...
                        CCRTrain(it), MSTrain(it), CCRTest(it), MSTest(it),...
                        NumberofInputNeurons(it),NumberofHiddenNeurons(it),NumberofOutputNeurons(it),InputWeight{it,1},OutputWeight{it,1},itResultsTotal{it,1}] = ...
                        eelm(train_data, test_data, 1, EE.nhidden, EE.nnbase, EE.wMin, EE.wMax, EE.CR, EE.F, EE.npop, EE.itermax,EE.refresh, EE.strategy,EE.tolerance);
                end
            end
        case {'elm'}
            if EE.multicore
                parfor it = 1:EE.repeatfold
                    [TrainingTime(it), ConfusionMatrixTrain{it,1}, ConfusionMatrixTest{it,1},...
                        CCRTrain(it), MSTrain(it), CCRTest(it), MSTest(it),...
                        NumberofInputNeurons(it),NumberofHiddenNeurons(it),NumberofOutputNeurons(it),InputWeight{it,1},OutputWeight{it,1}] = ...
                        elm(train_data, test_data, 1, EE.nhidden, EE.nnbase, EE.wMin, EE.wMax);
                end
            else
                for it = 1:EE.repeatfold
                    [TrainingTime(it), ConfusionMatrixTrain{it,1}, ConfusionMatrixTest{it,1},...
                        CCRTrain(it), MSTrain(it), CCRTest(it), MSTest(it),...
                        NumberofInputNeurons(it),NumberofHiddenNeurons(it),NumberofOutputNeurons(it),InputWeight{it,1},OutputWeight{it,1}] = ...
                        elm(train_data, test_data, 1, EE.nhidden, EE.nnbase, EE.wMin, EE.wMax);
                end
            end
        case {'pcaelm'}
            if EE.multicore
                parfor it = 1:EE.repeatfold
                    [TrainingTime(it), ConfusionMatrixTrain{it,1}, ConfusionMatrixTest{it,1},...
                        CCRTrain(it), MSTrain(it), CCRTest(it), MSTest(it),...
                        NumberofInputNeurons(it),NumberofHiddenNeurons(it),NumberofHiddenNeuronsFinal(it),NumberofOutputNeurons(it),InputWeight{it,1},OutputWeight{it,1}] = ...
                        pcaelm(train_data, test_data, 1, EE.nhidden, EE.nnbase, EE.wMin, EE.wMax);
                end
            else
                for it = 1:EE.repeatfold
                    [TrainingTime(it), ConfusionMatrixTrain{it,1}, ConfusionMatrixTest{it,1},...
                        CCRTrain(it), MSTrain(it), CCRTest(it), MSTest(it),...
                        NumberofInputNeurons(it),NumberofHiddenNeurons(it),NumberofHiddenNeuronsFinal(it),NumberofOutputNeurons(it),InputWeight{it,1},OutputWeight{it,1}] = ...
                        pcaelm(train_data, test_data, 1, EE.nhidden, EE.nnbase, EE.wMin, EE.wMax);
                end
            end
        case {'ldaelm'}
            if EE.multicore
                parfor it = 1:EE.repeatfold
                    [TrainingTime(it), ConfusionMatrixTrain{it,1}, ConfusionMatrixTest{it,1},...
                        CCRTrain(it), MSTrain(it), CCRTest(it), MSTest(it),...
                        NumberofInputNeurons(it),NumberofHiddenNeurons(it),NumberofHiddenNeuronsFinal(it),NumberofOutputNeurons(it),InputWeight{it,1},OutputWeight{it,1}] = ...
                        ldaelm(train_data, test_data, 1, EE.nhidden, EE.nnbase, EE.wMin, EE.wMax);
                end
            else
                for it = 1:EE.repeatfold
                    [TrainingTime(it), ConfusionMatrixTrain{it,1}, ConfusionMatrixTest{it,1},...
                        CCRTrain(it), MSTrain(it), CCRTest(it), MSTest(it),...
                        NumberofInputNeurons(it),NumberofHiddenNeurons(it),NumberofHiddenNeuronsFinal(it),NumberofOutputNeurons(it),InputWeight{it,1},OutputWeight{it,1}] = ...
                        ldaelm(train_data, test_data, 1, EE.nhidden, EE.nnbase, EE.wMin, EE.wMax);
                end
            end
        case {'pcaldaelm'}
            if EE.multicore
                parfor it = 1:EE.repeatfold
                    [TrainingTime(it), ConfusionMatrixTrain{it,1}, ConfusionMatrixTest{it,1},...
                        CCRTrain(it), MSTrain(it), CCRTest(it), MSTest(it),...
                        NumberofInputNeurons(it),NumberofHiddenNeurons(it),NumberofHiddenNeuronsFinal(it),NumberofOutputNeurons(it),InputWeight{it,1},OutputWeight{it,1}] = ...
                        pcaldaelm(train_data, test_data, 1, EE.nhidden, EE.nnbase, EE.wMin, EE.wMax);
                end
            else
                for it = 1:EE.repeatfold
                    [TrainingTime(it), ConfusionMatrixTrain{it,1}, ConfusionMatrixTest{it,1},...
                        CCRTrain(it), MSTrain(it), CCRTest(it), MSTest(it),...
                        NumberofInputNeurons(it),NumberofHiddenNeurons(it),NumberofHiddenNeuronsFinal(it),NumberofOutputNeurons(it),InputWeight{it,1},OutputWeight{it,1}] = ...
                        pcaldaelm(train_data, test_data, 1, EE.nhidden, EE.nnbase, EE.wMin, EE.wMax);
                end
            end
       
        case {'ielm'}
            if EE.multicore
                parfor it = 1:EE.repeatfold
                    [TrainingTime(it), ConfusionMatrixTrain{it,1}, ConfusionMatrixTest{it,1},...
                        CCRTrain(it), MSTrain(it), CCRTest(it), MSTest(it),...
                        NumberofInputNeurons(it),NumberofHiddenNeurons(it),NumberofHiddenNeuronsFinal(it),NumberofOutputNeurons(it),InputWeight{it,1},OutputWeight{it,1}] = ...
                        ielm(train_data, test_data, 1, EE.nhidden, EE.nnbase, EE.wMin, EE.wMax);
                end
            else
                for it = 1:EE.repeatfold
                    [TrainingTime(it), ConfusionMatrixTrain{it,1}, ConfusionMatrixTest{it,1},...
                        CCRTrain(it), MSTrain(it), CCRTest(it), MSTest(it),...
                        NumberofInputNeurons(it),NumberofHiddenNeurons(it),NumberofHiddenNeuronsFinal(it),NumberofOutputNeurons(it),InputWeight{it,1},OutputWeight{it,1}] = ...
                        ielm(train_data, test_data, 1, EE.nhidden, EE.nnbase, EE.wMin, EE.wMax);
                end
            end
       
        case {'opelm'}
            if EE.multicore
                parfor it = 1:EE.repeatfold
                    [TrainingTime(it), ConfusionMatrixTrain{it,1}, ConfusionMatrixTest{it,1},...
                        CCRTrain(it), MSTrain(it), CCRTest(it), MSTest(it),...
                        NumberofInputNeurons(it),NumberofHiddenNeurons(it),NumberofHiddenNeuronsFinal(it),NumberofOutputNeurons(it),InputWeight{it,1},OutputWeight{it,1}] = ...
                        opelm(train_data, test_data, 1, EE.nhidden, EE.nnbase);
                end
            else
                for it = 1:EE.repeatfold
                    [TrainingTime(it), ConfusionMatrixTrain{it,1}, ConfusionMatrixTest{it,1},...
                        CCRTrain(it), MSTrain(it), CCRTest(it), MSTest(it),...
                        NumberofInputNeurons(it),NumberofHiddenNeurons(it),NumberofHiddenNeuronsFinal(it),NumberofOutputNeurons(it),InputWeight{it,1},OutputWeight{it,1}] = ...
                        opelm(train_data, test_data, 1, EE.nhidden, EE.nnbase);
                end
            end
    end
else
    % TODO: Update from old versions
end

cpu_time=toc;

EEStats.TrainingTime_mean = mean(TrainingTime);
EEStats.TrainingTime_std = std(TrainingTime);

EEStats.CCRTrain_mean = mean(CCRTrain);
EEStats.CCRTrain_std = std(CCRTrain);
EEStats.MSTrain_mean = mean(MSTrain);
EEStats.MSTrain_std = std(MSTrain);

EEStats.CCRTest_mean = mean(CCRTest);
EEStats.CCRTest_std = std(CCRTest);
EEStats.MSTest_mean = mean(MSTest);
EEStats.MSTest_std = std(MSTest);

EEStats.npop = EE.npop;

switch lower(EE.elmAlgorithm)
    case {'eelm','elm'}
        EEStats.nhidden = EE.nhidden;
        EEStats.nhiddenFinal = EE.nhidden
    case{'opelm','pcaelm','ldaelm','pcaldaelm','ielm'}
        EEStats.nhidden = EE.nhidden;
        EEStats.nhiddenFinal = mean(NumberofHiddenNeuronsFinal);
end

if logLevel > 0
    if logLevel > 1
        disp(' ');
        disp('-------------------------------------------------------------------------');
        disp('EXPERIMENT PARAMETERS');
        disp('-------------------------------------------------------------------------');
        disp(['EE.elmAlgorithm: ' EE.elmAlgorithm]);
        disp(['EE.dbName: ' EE.dbName]);
        disp(['dbPreffixTrain: ' EE.dbPreffixTrain]);
        disp(['dbPreffixTest: ' EE.dbPreffixTest]);
        disp(['EE.XVMin: ' num2str(EE.XVMin)]);
        disp(['EE.XVMax: ' num2str(EE.XVMax)]);
        disp(['EE.wMin: ' num2str(EE.wMin)]);
        disp(['EE.wMax: ' num2str(EE.wMax)]);
        disp(['EE.preprocessData: ' EE.preprocessData]);
        disp(['EE.nnbase: ' num2str(EE.nnbase)]);
        disp(['Population: ' num2str(EE.npop)]);
        disp(['Hidden nodes: ' num2str(EEStats.nhiddenFinal)]);
        
        disp(['Iterations: ' num2str(EE.itermax)]);
        disp(['EE.refresh: ' num2str(EE.refresh)]);
        disp(['EE.CR: ' num2str(EE.CR)]);
        disp(['EE.F: ' num2str(EE.F)]);
        disp(['EE.CR: ' num2str(EE.CR)]);
        disp(['EE.strategy: ' num2str(EE.strategy)]);
        disp(['EE.tolerance: ' num2str(EE.tolerance)]);
        
        disp(' ');
        disp('-------------------------------------------------------------------------');
        disp('EXPERIMENT RESULTS');
        disp('-------------------------------------------------------------------------');
        disp(['TrainingTime_mean: ' num2str(EEStats.TrainingTime_mean)]);
        disp(['TrainingTime_std: ' num2str(EEStats.TrainingTime_std)]);
        disp(['TotalTime: ' num2str(cpu_time)]);
        disp(' ');
        disp(['CCRTrain_mean: ' num2str(EEStats.CCRTrain_mean)]);
        disp(['CCRTrain_std: ' num2str(EEStats.CCRTrain_std)]);
        disp(['MSTrain_mean: ' num2str(EEStats.MSTrain_mean)]);
        disp(['MSTrain_std: ' num2str(EEStats.MSTrain_std)]);
        disp(' ');
        disp(['CCRTest_mean: ' num2str(EEStats.CCRTest_mean)]);
        disp(['CCRTest_std: ' num2str(EEStats.CCRTest_std)]);
        disp(['MSTest_mean: ' num2str(EEStats.MSTest_mean)]);
        disp(['MSTest_std: ' num2str(EEStats.MSTest_std)]);
        
        
        dlmwrite(experimentResultOutput, '' );
        
        dlmwrite(experimentResultOutput,'-------------------------------------------------------------------------','-append','delimiter','');
        dlmwrite(experimentResultOutput,'EXPERIMENT PARAMETERS','-append','delimiter','');
        dlmwrite(experimentResultOutput,'-------------------------------------------------------------------------','-append','delimiter','');
        dlmwrite(experimentResultOutput,['EE.elmAlgorithm: ' EE.elmAlgorithm],'-append','delimiter','');
        dlmwrite(experimentResultOutput,['EE.dbName: ' EE.dbName],'-append','delimiter','');
        dlmwrite(experimentResultOutput,['dbPreffixTrain: ' EE.dbPreffixTrain],'-append','delimiter','');
        dlmwrite(experimentResultOutput,['dbPreffixTest: ' EE.dbPreffixTest],'-append','delimiter','');
        dlmwrite(experimentResultOutput,['EE.XVMin : ' num2str(EE.XVMin) ],'-append','delimiter','');
        dlmwrite(experimentResultOutput,['EE.XVMax: ' num2str(EE.XVMax)],'-append','delimiter','');
        dlmwrite(experimentResultOutput,['EE.wMin: ' num2str(EE.wMin)],'-append','delimiter','');
        dlmwrite(experimentResultOutput,['EE.wMax: ' num2str(EE.wMax)],'-append','delimiter','');
        dlmwrite(experimentResultOutput,['EE.preprocessData: ' EE.preprocessData],'-append','delimiter','');
        
        dlmwrite(experimentResultOutput,['EE.nnbase: ' num2str(EE.nnbase)],'-append','delimiter','');
        dlmwrite(experimentResultOutput,['Population: ' num2str(EE.npop)],'-append','delimiter','');
        dlmwrite(experimentResultOutput,['Hidden nodes: ' num2str(EE.nhidden)],'-append','delimiter','');
        dlmwrite(experimentResultOutput,['Iterations: ' num2str(EE.itermax)],'-append','delimiter','');
        dlmwrite(experimentResultOutput,['EE.refresh: ' num2str(EE.refresh)],'-append','delimiter','');
        dlmwrite(experimentResultOutput,['EE.CR: ' num2str(EE.CR)],'-append','delimiter','');
        dlmwrite(experimentResultOutput,['EE.F: ' num2str(EE.F)],'-append','delimiter','');
        dlmwrite(experimentResultOutput,['EE.CR: ' num2str(EE.CR)],'-append','delimiter','');
        dlmwrite(experimentResultOutput,['EE.strategy: ' num2str(EE.strategy)],'-append','delimiter','');
        dlmwrite(experimentResultOutput,['EE.tolerance: ' num2str(EE.tolerance)],'-append','delimiter','');
        
        dlmwrite(experimentResultOutput,' ','-append','delimiter','');
        dlmwrite(experimentResultOutput,'-------------------------------------------------------------------------','-append','delimiter','');
        dlmwrite(experimentResultOutput,'EXPERIMENT RESULTS','-append','delimiter','');
        dlmwrite(experimentResultOutput,'-------------------------------------------------------------------------','-append','delimiter','');
        dlmwrite(experimentResultOutput,['TrainingTime_mean: ' num2str(EEStats.TrainingTime_mean)],'-append','delimiter','');
        dlmwrite(experimentResultOutput,['TrainingTime_std: ' num2str(EEStats.TrainingTime_std)],'-append','delimiter','');
        dlmwrite(experimentResultOutput,['TotalTime: ' num2str(cpu_time)],'-append','delimiter','');
        dlmwrite(experimentResultOutput,' ','-append','delimiter','');
        dlmwrite(experimentResultOutput,['CCRTrain_mean: ' num2str(EEStats.CCRTrain_mean)],'-append','delimiter','');
        dlmwrite(experimentResultOutput,['CCRTrain_std: ' num2str(EEStats.CCRTrain_std)],'-append','delimiter','');
        dlmwrite(experimentResultOutput,['MSTrain_mean: ' num2str(EEStats.MSTrain_mean)],'-append','delimiter','');
        dlmwrite(experimentResultOutput,['MSTrain_std: ' num2str(EEStats.MSTrain_std)],'-append','delimiter','');
        dlmwrite(experimentResultOutput,' ','-append','delimiter','');
        dlmwrite(experimentResultOutput,['CCRTest_mean: ' num2str(EEStats.CCRTest_mean)],'-append','delimiter','');
        dlmwrite(experimentResultOutput,['CCRTest_std: ' num2str(EEStats.CCRTest_std)],'-append','delimiter','');
        dlmwrite(experimentResultOutput,['MSTest_mean: ' num2str(EEStats.MSTest_mean)],'-append','delimiter','');
        dlmwrite(experimentResultOutput,['MSTest_std: ' num2str(EEStats.MSTest_std)],'-append','delimiter','');
        
        
        printExperimentData(bestOuput_file,bestConfusionMatrix_file,experimentResultOutputDat,experimentResultOutputXLS,...
            ConfusionMatrixTrain, ConfusionMatrixTest, CCRTrain, MSTrain, CCRTest, MSTest,EEStats, ...
            NumberofInputNeurons,NumberofHiddenNeurons,NumberofOutputNeurons,...
            InputWeight, OutputWeight, EE.repeatfold);
        
        if strcmp(EE.elmAlgorithm, 'eelm')
            printExperimentgenerationsLog(generationsLog_file, itResultsTotal, EE.nhidden);
        end
        
        printPerformanceTable(performanceTable,EEStats);
        
    else
        printPerformanceTable(crossVal_performanceTable,EEStats);
    end
end
