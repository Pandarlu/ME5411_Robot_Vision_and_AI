clc
clear

ims = imageDatastore('p_dataset_26',...
    'IncludeSubfolders', true, ...
    'LabelSource', 'foldernames');

targetSize = [128, 128];
[imsTrain, imsTest] = splitEachLabel(ims,0.75,'randomized');
numClasses = numel(categories(imsTrain.Labels));

layers = [
    imageInputLayer(targetSize)
   
    fullyConnectedLayer(256)
    batchNormalizationLayer
    reluLayer

    fullyConnectedLayer(128)
    batchNormalizationLayer
    reluLayer

    fullyConnectedLayer(64)
    batchNormalizationLayer
   reluLayer

    fullyConnectedLayer(32) 
    batchNormalizationLayer
    sigmoidLayer

    fullyConnectedLayer(7)
    softmaxLayer
    classificationLayer
    ];

options = trainingOptions("sgdm",...
   "InitialLearnRate",0.008,...
   "MaxEpochs",49,...
   "Shuffle","every-epoch",...
   "ValidationData",imsTest, ...
   "validationfrequency",20,...
   "MiniBatchSize",49,...
   "Verbose",true,...
   "Plots","training-progress");

net=trainNetwork(imsTrain,layers,options);

 YPred = classify(net,imsTest);
  YTest = imsTest.Labels;

   accuracy = sum(YPred == YTest)/numel(YTest)

   confmat=confusionmat(YTest,YPred);
   %MhelperDisplayConfusionMatrix(confmat);
   plotconfusion(YTest,YPred);

   % Save the trained MLP model
   % Specify the directory for saving the model
   nn_directory = 'TrainedMLP';
   % Check if the directory exists
    if ~isfolder(nn_directory)
   % Specify the name for saving the model
    mkdir(nn_directory);
    end
   % Form the full path for saving the model by combining the directory and name
   name_nn = "MLP_1";
   nn_fullpath = fullfile(nn_directory, name_nn);
   save(nn_fullpath,'net')
