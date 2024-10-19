
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
    outputFilePath = fullfile(outputFolder, [imageName '_output' imageExt]);

    imwrite(newImage, outputFilePath);
end
