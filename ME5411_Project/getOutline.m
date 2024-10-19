function edgeImage = getOutline(binaryImage, varargin)
    % getOutline applies morphological operations and edge detection to a binary image.
    % It allows users to specify custom filenames for intermediate results.

    % Parameters:
    %   binaryImage: Binary input image.
    %   varargin: Variable number of input arguments for custom filenames.
    %             If not provided, default filenames will be used.

    % Apply morphological operations to reduce noise
    se1 = strel('disk', 2);
    se2 = strel('disk', 4);

    % Default filenames
    defaultFilenames = {'erodedImage', 'dilatedImage', 'cleanedImage', 'edgeImage'};
    
    % Parse user-provided filenames or use defaults
    if nargin > 1
        userFilenames = varargin;
    else
        userFilenames = defaultFilenames;
    end

    % Perform erosion operation
    erodedImage = imerode(binaryImage, se1);
    imwrite(erodedImage, ['img_output/', userFilenames{1}, '.png']);

    % Perform dilation operation
    dilatedImage = imdilate(erodedImage, se2);
    imwrite(dilatedImage, ['img_output/', userFilenames{2}, '.png']);

    % Use bwareaopen to hard cut off remaining small areas with pixel count < 100
    cleanedImage = bwareaopen(dilatedImage, 100);
    imwrite(cleanedImage, ['img_output/', userFilenames{3}, '.png']);

    % Apply Canny edge detection to the cleaned image
    edgeImage = edge(cleanedImage, 'Canny');
    imwrite(edgeImage, ['img_output/', userFilenames{4}, '.png']);
end
