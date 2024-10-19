function plotImageHistogram(imagePath, titleText)
    % plotImageHistogram generates and plots the histogram of the input image.
    % Parameters:
    %   - imagePath: The file path to the input image.
    %   - titleText: The title for the histogram plot.

    % Read the image from the specified file path
    image = imread(imagePath);

    % Calculate the histogram
    [counts, binLocations] = imhist(image);
    % Plot the histogram
    bar(binLocations, counts);

    % Add title and labels
    title(titleText);
    xlabel('Gray Level');
    ylabel('Pixel Count');
end
