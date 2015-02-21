
% Base function
EE.nnbase = 'sig';

switch lower(EE.nnbase)
    case {'sig','sigmoid'}
        EE.XVMin = -1;
        EE.XVMax = 1;
        EE.wMin = -1;
        EE.wMax = 1;
        EE.preprocessData = 'scale'; % 'standarize', 'scale'
    case {'rbf','grbf','krbf','rbf2'}
        EE.XVMin = -1;
        EE.XVMax = 1;
        EE.wMin = -1;
        EE.wMax = 1;
        EE.preprocessData = 'scale';   
    case {'up'}
        EE.XVMin = 1;
        EE.XVMax = 2;
        EE.wMin = -5;
        EE.wMax = 5;
        EE.preprocessData = 'scale'; 
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Differential evolution parameters
% DE crossover Strategy
EE.CR = 0.8;
EE.F = 1;
EE.strategy = 3;
EE.tolerance = 0.02;
%%%%%%%%%%%%%%%%%%%% End Setup

% Autovalues for files and dirs

c = clock;
dir_suffix = [num2str(c(1)) '-' num2str(c(2)) '-'  num2str(c(3)) ...
            '-' num2str(c(4)) '-' num2str(c(5)) '-' num2str(uint8(c(6)))];
EE.dbPreffixTrain = ['../datasets/' EE.dbName '/train_' EE.dbName '.elm'];
EE.dbPreffixTest = ['../datasets/' EE.dbName '/test_' EE.dbName '.elm'];
EE.outputDir = ['experiments/' EE.dbName '_' dir_suffix];
clear c dir_suffix;

runexperiment(EE);
