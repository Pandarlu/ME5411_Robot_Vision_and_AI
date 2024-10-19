clc
clear

ims = imageDatastore('p_dataset_26',...
    'IncludeSubfolders', true, ...
    'LabelSource', 'foldernames');

targetSize = [128, 128];
[imsTrain, imsTest] = splitEachLabel(ims,0.75,'randomized');
numClasses = numel(categories(imsTrain.Labels));

augmenter = imageDataAugmenter(...
 'RandRotation', [-12, 12], ...
    'RandXTranslation', [-4, 4] , ...
    'RandYTranslation', [-4, 4], ...
    'RandXScale', [0.8, 1.2], ...
    'RandYScale', [0.8, 1.2]);

augmentedtrainingSet = augmentedImageDatastore(targetSize, imsTrain, 'DataAugmentation', augmenter);
augmentedtestSet = augmentedImageDatastore(targetSize, imsTest, 'DataAugmentation', augmenter);

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

net=trainNetwork(augmentedtrainingSet,layers,options);

YPred=classify(net,augmentedtestSet);
YTest=imsTest.Labels;

accuracy = sum(YPred == YTest)/numel(YTest)

confmat=confusionmat(YTest,YPred);
%MhelperDisplayConfusionMatrix(confmat);
plotconfusion(YTest,YPred);

nn_directory = 'TrainedMLP_enhance';
if ~isfolder(nn_directory)
    mkdir(nn_directory);
end
name_nn = "MLP_2";
nn_fullpath = fullfile(nn_directory, name_nn);
save(nn_fullpath,'net');

