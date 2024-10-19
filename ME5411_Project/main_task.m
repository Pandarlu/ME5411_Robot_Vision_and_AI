clc; clear; close all;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Q1: Display the original image on the screen
fprintf('Showing original image...\n');
originalImage = imread('img_input\charact2.bmp');
myImage = im2gray(originalImage);
imwrite(myImage,'img_output\myImage.png');
figure;
imshow(myImage);
title('Original Image');



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Q2: Implement and apply different masks for image smoothing
fprintf('Applying different masks for image smoothing...\n');
filterSizes = [3, 5]; % Different filter sizes
methods = {'Averaging', 'Rotating'}; % Two methods: Averaging and Rotating

figure;
index = 1;

for i = 1:length(filterSizes)
    filterSize = filterSizes(i);

    for j = 1:length(methods)
        method = methods{j};

        if strcmp(method, 'Averaging')
            smoothedImage = uint8(averageFilter(myImage, filterSize));
        elseif strcmp(method, 'Rotating')
            smoothedImage = uint8(rotateFilter(myImage, filterSize));
        else
            error('Invalid smoothing method');
        end

        % Display the smoothed image
        subplot(2, 2, index);
        index = index + 1;
        imshow(smoothedImage);
        title(sprintf('%dx%d %s mask result', filterSize, filterSize, method));

        % Save the smoothed image to the specified output directory
        outputFilename = sprintf('%dx%d %s mask result.png', filterSize, filterSize, method);
        outputPath = fullfile('img_output\MaskedImages', outputFilename);
        imwrite(smoothedImage, outputPath);
    end
end

%%%% Compare results of different image smoothing methods using histograms
fprintf('Comparing results of different image smoothing methods using histograms...\n');

% Display the histogram of the original image
figure;
plotImageHistogram('img_output\myImage.png', 'Histogram of Original Image');

% Folder path
folderPath = 'img_output/MaskedImages';

% Get all image files in the folder
imageFiles = dir(fullfile(folderPath, '*.png')); % Assuming image files are in PNG format; modify as needed

% Create a new figure
figure;

% Loop through each image file and generate histograms
for i = 1:numel(imageFiles)
    % Construct the full file path
    imagePath = fullfile(folderPath, imageFiles(i).name);
    
    % Generate a title using the file name
    [~, titleText, ~] = fileparts(imageFiles(i).name);
    titleText = ['Histogram of ', titleText];
    
    % Plot histograms on the same figure
    subplot(2, ceil(numel(imageFiles)/2), i);
    plotImageHistogram(imagePath, titleText);
end

% Turn off "hold" to ensure subsequent figures are not overlaid on the current figure
hold off;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Q3: Create an image which is a sub-image of the original image comprising the line:HD44780A00.
fprintf('Creating a sub-image comprising the line: HD44780A00...\n');
secondLineImage = getSecondLine(myImage);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Q4: Create a binary image using thresholding
fprintf('Creating a binary image using thresholding...\n');
thresholdValue = 0.47; 
binaryImage = imbinarize(secondLineImage, thresholdValue); 

figure;
imshow(binaryImage);
title('Binary Image');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Q5: Determine the outline(s) of characters in the image.
fprintf('Determining the outlines of characters in the image...\n');
edgeImage = getOutline(binaryImage, 'erodedImage', 'dilatedImage', 'cleanedImage', 'edgeImage');
figure;
% Display images processing
subplot(2, 3, 1);
highlightedSencondLine = imread("img_output\highlightedSecondLineImage.png");
imshow(highlightedSencondLine);
title('Highlighted Second Line');

subplot(2, 3, 2);
secondLineImage = imread("img_output\secondLineImage.png");
imshow(secondLineImage);
title('Image of Second Line');

subplot(2, 3, 3);
erodedImage = imread("img_output\erodedImage.png");
imshow(erodedImage);
title('Eroded Image');

subplot(2, 3, 4);
dilatedImage = imread("img_output\dilatedImage.png");
imshow(dilatedImage);
title('Dilated Image');

subplot(2, 3, 5);
cleanedImage = imread("img_output\cleanedImage.png");
imshow(cleanedImage);
title('Cleaned Image');

subplot(2, 3, 6);
imshow(edgeImage);
edgeImageImage = imread("img_output\edgeImage.png");
title('Edges of Characters');
hold on;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Q6: Segment the image to separate and label the different characters.
fprintf('Segmenting the image to separate and label the different characters...\n');
cleanedImage = imread('img_output\cleanedImage.png');
segmentAndLabelCharacters(cleanedImage, 'img_output\CroppedCharacters');

% Display character images
fprintf('Displaying segmented character images...\n');
displayCharacterImages('img_output\CroppedCharacters');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Resize character images for classification
fprintf('Resizing character images for CNN processing...\n');

inputFolder = 'img_output\CroppedCharacters';

imagefiles = dir(fullfile(inputFolder, '*.png'));
nfiles = length(imagefiles);

for i = 1:nfiles
    current_filename = fullfile(inputFolder, imagefiles(i).name);
    current_image = imread(current_filename);

    newImage = zeros(128, 128);

    startRow = floor((128 - size(current_image, 1)) / 2) + 1;
    startCol = floor((128 - size(current_image, 2)) / 2) + 1;

    newImage(startRow:startRow+size(current_image, 1)-1, startCol:startCol+size(current_image, 2)-1, :) = current_image;

    newImage = 1 - newImage; 

    [~, imageName, imageExt] = fileparts(imagefiles(i).name);
    outputFolder = 'img_output\Cropped_paddingCharaters'; 
    outputFilePath = fullfile(outputFolder, [imageName '' imageExt]);

    imwrite(newImage, outputFilePath);
end

% Display resized character images
fprintf('Displaying resized character images...\n');
displayCharacterImages('img_output\Cropped_paddingCharaters');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Q7: [All operations need to be run in CNN.m; CNN_ResultsFigure and MLP.m; MLP_ResultsFigure]

fprintf('All operations need to be run in CNN.m; CNN_ResultsFigure and MLP.m; MLP_ResultsFigure...\n');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Q8: [All operations need to be run in 
% CNNenhanced.m and CNN_ResultsFigure_AllEnhanced.m 
% and MLPenhanced.m and MLP_ResultsFigure_AllEnhanced.m; 
% CNN_Hyper_LearnRate,CNN_Hyper_MaxEpochs,
% MLP_Hyper_MaxEpochs,MLP_Hyper_LearnRate_MLP,MLP_Hyper_Minibatch]

fprintf('All operations need to be run in the corresponding files...\n');



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fprintf('All Work Completed!\n')
