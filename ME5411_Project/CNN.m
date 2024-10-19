clc
clear

ims = imageDatastore('p_dataset_26',...
    'IncludeSubfolders', true, ...
    'LabelSource', 'foldernames');

targetSize = [128, 128];
[imsTrain, imsTest] = splitEachLabel(ims,0.75,'randomized');
numClasses = numel(categories(imsTrain.Labels));

layers = [
    imageInputLayer([targetSize])

    convolution2dLayer(3,8,'Padding','same')
    batchNormalizationLayer
    reluLayer

    maxPooling2dLayer(2,'Stride',2)

    convolution2dLayer(3,16,'Padding','same')
    batchNormalizationLayer
    reluLayer

    maxPooling2dLayer(2,'Stride',2)

    convolution2dLayer(3,32,'Padding','same')
    batchNormalizationLayer
    reluLayer

    maxPooling2dLayer(2,'Stride',2)

    convolution2dLayer(3,64,'Padding','same')
    batchNormalizationLayer
    reluLayer

    convolution2dLayer(3,128,'Padding','same')
    batchNormalizationLayer
    reluLayer

    maxPooling2dLayer(2,'Stride',2)
    
    convolution2dLayer(3,256,'Padding','same')
    batchNormalizationLayer
    reluLayer

    maxPooling2dLayer(2,'Stride',2)

    dropoutLayer

    fullyConnectedLayer(7)
    softmaxLayer
    classificationLayer
    ];

options = trainingOptions("adam",...
   "InitialLearnRate",0.002,...
   "MaxEpochs",22,...
   "Shuffle","every-epoch",...
   "ValidationData",imsTest, ...
   "validationfrequency",20,...
   "MiniBatchSize",55,...
   "Verbose",false,...
   "ExecutionEnvironment","auto",...
   "Plots","training-progress");

net=trainNetwork(imsTrain,layers,options);

 YPred = classify(net,imsTest);
  YTest = imsTest.Labels;

   accuracy = sum(YPred == YTest)/numel(YTest)

   confmat=confusionmat(YTest,YPred);
   %MhelperDisplayConfusionMatrix(confmat);
   plotconfusion(YTest,YPred);

   % Save the trained CNN model
   % Specify the directory for saving the model
   nn_directory = 'Trainedcnn';
   % Check if the directory exists
    if ~isfolder(nn_directory)
    mkdir(nn_directory);
    end
   % Specify the name for saving the model
   name_nn = "CNN_1";
   % Form the full path for saving the model by combining the directory and name
   nn_fullpath = fullfile(nn_directory, name_nn);
   save(nn_fullpath,'net')
 