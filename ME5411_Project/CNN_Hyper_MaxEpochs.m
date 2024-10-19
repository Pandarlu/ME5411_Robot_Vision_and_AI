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
   "InitialLearnRate",0.001,...
   "MaxEpochs",9,...
   "Shuffle","every-epoch",...
   "ValidationData",imsTest, ...
   "validationfrequency",20,...
   "MiniBatchSize",64,...
   "Verbose",false,...
   "ExecutionEnvironment","auto");
   %"ExecutionEnvironment","auto",...
   %"Plots","training-progress");

net=trainNetwork(imsTrain,layers,options);

 YPred = classify(net,imsTest);
  YTest = imsTest.Labels;

   accuracy = sum(YPred == YTest)/numel(YTest)

   confmat=confusionmat(YTest,YPred);
   %MhelperDisplayConfusionMatrix(confmat);
   plotconfusion(YTest,YPred);

maxEpochsRange = [9, 38];
maxEpochsValues = linspace(maxEpochsRange(1), maxEpochsRange(2), 10);
results = struct('LearnRate', {}, 'MaxEpochs', {}, 'MiniBatchSize', {}, 'Accuracy', {});

for j = 1:length(maxEpochsValues)
    options.MaxEpochs = round(maxEpochsValues(j));
    net = trainNetwork(imsTrain, layers, options);
    YPred = classify(net, imsTest);
    YTest = imsTest.Labels;
    accuracy = sum(YPred == YTest) / numel(YTest);

    results(end + 1).LearnRate = options.InitialLearnRate;
    results(end).MaxEpochs = round(maxEpochsValues(j));
    results(end).MiniBatchSize = options.MiniBatchSize;
    results(end).Accuracy = accuracy;
end


disp('Results:');
disp(results);

maxEpochsResults = [results(:).MaxEpochs];
accuracyResults = [results(:).Accuracy];
[bestAccuracyMaxEpochs, idxMaxEpochs] = max(accuracyResults(1:10));
bestMaxEpochs = maxEpochsResults(idxMaxEpochs);

figure;

subplot(3, 1, 2);
plot(maxEpochsResults, accuracyResults, 'o-');
hold on;
scatter(bestMaxEpochs, bestAccuracyMaxEpochs, 100, 'r', 'filled');
title('Accuracy vs MaxEpochs');
xlabel('MaxEpochs');
ylabel('Accuracy');