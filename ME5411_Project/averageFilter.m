function smoothedImage = averageFilter(myImage, filterSize)
    % averageFilter applies an averaging filter to smooth the input image.
    % The filter is a square matrix of ones normalized by the square of its size.

    % Get the size of the image
    [rows, cols] = size(myImage);

    % Define the averaging filter kernel
    filter = ones(filterSize) / (filterSize^2);

    % Initialize a result image of the same size as the original image
    smoothedImage = zeros(size(myImage));

    % Calculate the half-size of the neighborhood
    neighborhoodRadius = floor(filterSize / 2);

    % Perform average filtering
    for i = neighborhoodRadius + 1 : rows - neighborhoodRadius
        for j = neighborhoodRadius + 1 : cols - neighborhoodRadius
            % Extract the neighborhood
            neighborhood = double(myImage(i - neighborhoodRadius : i + neighborhoodRadius, j - neighborhoodRadius : j + neighborhoodRadius));

            % Apply the averaging filter kernel
            smoothedPixel = sum(sum(neighborhood .* filter));

            % Update the pixel value in the result image
            smoothedImage(i, j) = smoothedPixel;
        end
    end
end
