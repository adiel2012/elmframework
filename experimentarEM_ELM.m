clear;
addpath libelmdefitness/;
addpath opelm/;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
EE.nhidden = 100;
EE.hNodesSet = 10:10:100;

% Evolutionary parameters
EE.npop = 50;
EE.itermax = 50;
EE.refresh = 5;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Data base and log files setup

EE.db = 'holdout';
EE.repeatfold = 30;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
EE.multicore = 0; % 0 o 1
EE.cores = 1;

%%% Selection of the algorithm
EE.eelm = 0; % E-ELM with C and S
EE.opelm = 0;
EE.elm = 0;
EE.pcaelm = 0;
EE.ldaelm = 0;
EE.pcaldaelm = 0;

EE.ielm = 0;
EE.pelm = 0;
EE.em_elm = 1;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

EE.dbName = 'anneal';experimentarutil;
EE.dbName = 'balance';experimentarutil;
EE.dbName = 'breast';experimentarutil;
EE.dbName = 'breastw';experimentarutil;
EE.dbName = 'btx';experimentarutil;
EE.dbName = 'card';experimentarutil;
EE.dbName = 'diabetes';experimentarutil;
EE.dbName = 'ecoli';experimentarutil;
EE.dbName = 'gcm_BA_190x311';experimentarutil;
EE.dbName = 'gcm_BI_190x288';experimentarutil;
EE.dbName = 'gcm_FC_190x264';experimentarutil;
EE.dbName = 'gene';experimentarutil;
EE.dbName = 'german';experimentarutil;
EE.dbName = 'glass';experimentarutil;
EE.dbName = 'haberman';experimentarutil;
EE.dbName = 'heartstatlog';experimentarutil;
EE.dbName = 'hepatitis';experimentarutil;
EE.dbName = 'ionos';experimentarutil;
EE.dbName = 'iris';experimentarutil;
EE.dbName = 'lymph';experimentarutil;
EE.dbName = 'newthyroid';experimentarutil;
EE.dbName = 'pima';experimentarutil;
EE.dbName = 'postop';experimentarutil;
EE.dbName = 'saureus4';experimentarutil;
%EE.dbName = 'seguros';experimentarutil;
EE.dbName = 'vote';experimentarutil;
EE.dbName = 'vowel';experimentarutil;
EE.dbName = 'zoo';experimentarutil;
resumen;