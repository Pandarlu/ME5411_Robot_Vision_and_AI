function displayCharacterImages(inputDirectory)
    % Check if inputDirectory is not provided, use the default directory
    if nargin < 1
        inputDirectory = 'img_output\characters';
    end

    % Get a list of all image files in the specified directory
    imageFiles = dir(fullfile(inputDirectory, '*.png'));

    % Calculate the number of rows and columns for the subplot
    numImages = numel(imageFiles);
    numRows = ceil(sqrt(numImages));
    numCols = ceil(numImages / numRows);

    % Create a figure
    figure;

    % Iterate through each image file
    for i = 1:numImages
        % Read the image
        imagePath = fullfile(imageFiles(i).folder, imageFiles(i).name);
        characterImage = imread(imagePath);

        % Create a subplot using column-major order
        subplot(numCols, numRows, i); % Swap numRows and numCols

        % Display the image
        imshow(characterImage);

        % Add a title with the image filename
        [~, imageName, ~] = fileparts(imageFiles(i).name);
        title(['Image: ', imageName], 'Interpreter', 'none');
    end
end

