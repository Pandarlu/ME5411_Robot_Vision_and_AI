function smoothedImage = rotateFilter(myImage, filterSize)
    % rotateFilter applies a rotating filter to smooth the input image.
    % The filter is created by convolving a ones matrix with itself,
    % and then normalizing the result by dividing by the fourth power of filterSize.

    % Get the size of the image
    [rows, cols] = size(myImage);
    
    % Initialize the result image
    smoothedImage = zeros(rows, cols);
    
    % Define the filter weights
    filter0 = conv2(ones(filterSize), ones(filterSize));
    filter = filter0 / power(filterSize, 4);
    
    % Perform rotating filtering
    for i = filterSize : rows - filterSize + 1
        for j = filterSize : cols - filterSize + 1
            % Extract the neighborhood
            neighborhood = double(myImage(i - filterSize + 1 : i + filterSize - 1, j - filterSize + 1 : j + filterSize - 1));
            
            % Apply the convolution kernel
            smoothedPixel = sum(sum(neighborhood .* filter));
            
            % Update the pixel value in the result image
            smoothedImage(i, j) = smoothedPixel;
        end
    end
end
