function printPerformanceTable(performanceTable,EEStats) 

results = [EEStats.nhidden EEStats.nhiddenFinal EEStats.CCRTrain_mean EEStats.CCRTrain_std EEStats.MSTrain_mean EEStats.MSTrain_std ...
                        EEStats.CCRTest_mean EEStats.CCRTest_std EEStats.MSTest_mean EEStats.MSTest_std ...
                        EEStats.TrainingTime_mean EEStats.TrainingTime_std];

dlmwrite(performanceTable, results,'-append', 'delimiter', '\t', ...
         'precision', 16);
     
clear results FitnessId;


