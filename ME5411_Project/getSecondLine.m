function secondLineImage = getSecondLine(myImage, outputFilename)
    % getSecondLine extracts a sub-image corresponding to the second line from the input image.
    % It also highlights the region of interest and saves the original and extracted images.

    % Set default output filename if not provided
    if nargin < 2
        outputFilename = 'Image';
    end

    % Define the region of interest for the second line
    secondLineRegion = [1, 190, 990-1, 367-190];

    % Create a figure without displaying it
    h = figure('Visible', 'on');
    % h = figure('Visible', 'off');

    % Show the original image
    imshow(myImage);

    % Highlight the region on the original image
    hold on;
    rectangle('Position', secondLineRegion, 'EdgeColor', 'r', 'LineWidth', 3);

    % Save the original image with the highlighted rectangle
    saveas(h, fullfile('img_output', ['highlightedSecondLine', outputFilename, '.png']));

    % Extract the second line region
    secondLineImage = imcrop(myImage, secondLineRegion);

    % Save the extracted second line image
    imwrite(secondLineImage, fullfile('img_output', ['secondLine', outputFilename, '.png']));

    % Close the figure
    close(h);
end
