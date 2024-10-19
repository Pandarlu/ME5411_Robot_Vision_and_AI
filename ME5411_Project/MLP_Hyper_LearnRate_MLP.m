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
   "InitialLearnRate",0.01,...
   "MaxEpochs",52,...
   "Shuffle","every-epoch",...
   "ValidationData",imsTest, ...
   "validationfrequency",20,...
   "MiniBatchSize",64,...
   "Verbose",true,...
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

learnRateRange = [0.001, 0.01];

learnRateValues = linspace(learnRateRange(1), learnRateRange(2), 10);

results = struct('LearnRate', {}, 'MaxEpochs', {}, 'MiniBatchSize', {}, 'Accuracy', {});


for i = 1:length(learnRateValues)
    options.InitialLearnRate = learnRateValues(i);
    net = trainNetwork(imsTrain, layers, options);
    YPred = classify(net, imsTest);
    YTest = imsTest.Labels;
    accuracy = sum(YPred == YTest) / numel(YTest);

    results(end + 1).LearnRate = learnRateValues(i);
    results(end).MaxEpochs = options.MaxEpochs;
    results(end).MiniBatchSize = options.MiniBatchSize;
    results(end).Accuracy = accuracy;
end

disp('Results:');
disp(results);

learnRateResults = [results(:).LearnRate];
accuracyResults = [results(:).Accuracy];
[bestAccuracyLearnRate, idxLearnRate] = max(accuracyResults(1:10));
bestLearnRate = learnRateResults(idxLearnRate);

figure;

subplot(3, 1, 1);
plot(learnRateResults, accuracyResults, 'o-');
hold on;
scatter(bestLearnRate, bestAccuracyLearnRate, 100, 'r', 'filled');
title('Accuracy vs InitialLearnRate');
xlabel('InitialLearnRate');
ylabel('Accuracy');