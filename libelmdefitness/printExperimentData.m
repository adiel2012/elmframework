function printExperimentData(bestOuput_file,bestConfusionMatrix_file,experimentResultOutputDat,experimentResultOutputXLS,...
    ConfusionMatrixTrain, ConfusionMatrixTest, CCRTrain, MSTrain, CCRTest, MSTest,EEStats, ...
    NumberInputNeurons,NumberofHiddenNeurons,NumberofOutputNeurons,...
    InputWeight, OutputWeight, repeatfold)

% for it = 1:repeatfold
%     printRun(bestOuput_file,bestConfusionMatrix_file,it, ConfusionMatrixTrain{it,1}, ConfusionMatrixTest{it,1},...
%         CCRTrain(it), MSTrain(it), CCRTest(it), MSTest(it));
% 
%     printBestMemberStructure(bestModelStructure,it,NumberInputNeurons(it),...
%            NumberofHiddenNeurons(it),NumberofOutputNeurons(it),...
%            InputWeight{it,1},OutputWeight{it,1});
% end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% VOLCADO DATOS MEJORES
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

si = (size(CCRTrain,1));
itNumbers = (1:si)';

results_tmp = [itNumbers CCRTrain MSTrain CCRTest MSTest];
resultsMean_tmp = [EEStats.CCRTrain_mean EEStats.MSTrain_mean EEStats.CCRTest_mean EEStats.MSTest_mean];
resultsStd_tmp = [EEStats.CCRTrain_std EEStats.MSTrain_std EEStats.CCRTest_std EEStats.MSTest_std];


dlmwrite([experimentResultOutputDat ], '' );
dlmwrite(experimentResultOutputDat, '% Iteration, CCRTrain, MSTrain, CCRTest, MSTest','-append', 'delimiter', '');
dlmwrite(experimentResultOutputDat, results_tmp,'-append', 'delimiter', '\t', ...
         'precision', 8);
dlmwrite(experimentResultOutputDat, '%  CCRTrain Mean/StdDv, MSTrain Mean/StdDv, CCRTest Mean/StdDv, MSTest Mean/StdDv',...
    '-append', 'delimiter', '');
dlmwrite(experimentResultOutputDat, resultsMean_tmp,'-append', 'delimiter', '\t', ...
         'precision', 8);
%dlmwrite(experimentResultOutputDat, '% Iteration, CCRTrain StdDv, MSTrain StdDv, CCRTest StdDv, MSTest StdDv',...
%    '-append', 'delimiter', '');
dlmwrite(experimentResultOutputDat, resultsStd_tmp,'-append', 'delimiter', '\t', ...
         'precision', 8);
     
dlmwrite([experimentResultOutputXLS ], '' );
dlmwrite(experimentResultOutputXLS, 'Iteration,CCRTrain,MSTrain,CCRTest,MSTest','-append', 'delimiter', '');
dlmwrite(experimentResultOutputXLS, results_tmp,'-append', 'delimiter', ',', ...
         'precision', 8);
dlmwrite(experimentResultOutputXLS, ' ,CCRTrain Mean/StdDv,MSTrain Mean/StdDv,CCRTest Mean/StdDv,MSTest Mean/StdDv',...
    '-append', 'delimiter', '');
dlmwrite(experimentResultOutputXLS, resultsMean_tmp,'-append', 'delimiter', ',', ...
         'precision', 8,'coffset',1);
%dlmwrite(experimentResultOutputXLS, ' ,CCRTrain StdDv,MSTrain StdDv,CCRTest StdDv,MSTest StdDv','-append', 'delimiter', '');
dlmwrite(experimentResultOutputXLS, resultsStd_tmp,'-append', 'delimiter', ',', ...
         'precision', 8,'coffset',1);
     
clear itNumbers;
clear results_tmp;
clear resultsMean_tmp;
clear resultsStd_tmp;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% MEJOR INDIVIDUO DE CADA EJECUCIÓN
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
dlmwrite(bestOuput_file, '' );
dlmwrite(bestConfusionMatrix_file, '' );
dlmwrite(bestConfusionMatrix_file, '% 2NxN = [ConfusionTrainMatrix ConfusionTestMatrix]','-append', 'delimiter', '');

dlmwrite(experimentResultOutputXLS, ' ','-append', 'delimiter', '');
dlmwrite(experimentResultOutputXLS, 'ConfusionTrainMatrix, ConfusionTestMatrix','-append', 'delimiter', '');
    
for it = 1:repeatfold
    dlmwrite(bestOuput_file, '-------------------------------------------------------------------------' ,'-append','roffset',1, 'delimiter', '');
    dlmwrite(bestOuput_file, ['Best model for execution ' num2str(it)] ,'-append', 'delimiter', '');
    dlmwrite(bestOuput_file, '-------------------------------------------------------------------------' ,'-append', 'delimiter', '');
    dlmwrite(bestOuput_file, 'NumberInputNeurons ','-append','roffset',1,'delimiter',''); 
    dlmwrite(bestOuput_file, NumberInputNeurons(it),'-append', 'delimiter', '\t', ...
         'precision', 8);
    dlmwrite(bestOuput_file, 'NumberofHiddenNeurons ','-append','roffset',1,'delimiter',''); 
    dlmwrite(bestOuput_file, NumberofHiddenNeurons(it),'-append', 'delimiter', '\t', ...
         'precision', 8);
    dlmwrite(bestOuput_file, 'NumberofOutputNeurons ','-append','roffset',1,'delimiter',''); 
    dlmwrite(bestOuput_file, NumberofOutputNeurons(it),'-append', 'delimiter', '\t', ...
         'precision', 8);
    dlmwrite(bestOuput_file, 'InputWeight ','-append','roffset',1,'delimiter',''); 
    dlmwrite(bestOuput_file, InputWeight{it,1},'-append', 'delimiter', '\t', ...
         'precision', 8);
    dlmwrite(bestOuput_file, 'OutputWeight ','-append','roffset',1,'delimiter',''); 
    dlmwrite(bestOuput_file, OutputWeight{it,1},'-append', 'delimiter', '\t', ...
         'precision', 8);
     
    dlmwrite(bestOuput_file, 'Train confusion matrix','-append','roffset',1,'delimiter',''); 
    dlmwrite(bestOuput_file, ConfusionMatrixTrain{it,1},'-append', 'delimiter', '\t', ...
            'precision', 4);
    dlmwrite(bestOuput_file, 'Test confusion matrix','-append','roffset',1,'delimiter',''); 
    dlmwrite(bestOuput_file, ConfusionMatrixTest{it,1},'-append', 'delimiter', '\t', ...
            'precision', 4);

    dlmwrite(bestOuput_file, 'CCRTrain','-append','roffset',1,'delimiter',''); 
    dlmwrite(bestOuput_file, CCRTrain(it),'-append', 'delimiter', '\t', ...
         'precision', 8);

    dlmwrite(bestOuput_file, 'MSTrain','-append','roffset',1,'delimiter',''); 
    dlmwrite(bestOuput_file, MSTrain(it),'-append', 'delimiter', '\t', ...
         'precision', 8);

    dlmwrite(bestOuput_file, 'CCRTest ','-append','roffset',1,'delimiter',''); 
    dlmwrite(bestOuput_file, CCRTest(it),'-append', 'delimiter', '\t', ...
         'precision', 8);

    dlmwrite(bestOuput_file, 'MSTest','-append','roffset',1,'delimiter',''); 
    dlmwrite(bestOuput_file, MSTest(it),'-append', 'delimiter', '\t', ...
         'precision', 8);
     
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % VOLCADO MATRIZ CONFUSIÓN
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    ConfusionMatrix_temp = [ConfusionMatrixTrain{it,1} ConfusionMatrixTest{it,1}];

    dlmwrite(bestConfusionMatrix_file, ConfusionMatrix_temp,'-append', 'delimiter', '\t', ...
             'precision', 8);

    dlmwrite(experimentResultOutputXLS, ConfusionMatrix_temp,'-append', 'delimiter', ',', ...
         'precision', 8);
end

clear ConfusionMatrix_temp;
