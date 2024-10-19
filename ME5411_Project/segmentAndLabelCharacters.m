function segmentAndLabelCharacters(binaryImage, outputDirectory)
    % Perform connected component analysis
    stats = regionprops(binaryImage, 'BoundingBox');

    % Set thresholds to determine if the bounding box width is greater than 150
    minWidth = 30;
    maxWidth = 150;

    % Create a new cell array to store all bounding box information
    newBoundingBox = cell(1, numel(stats));

    % Initialize index
    newIndex = 1;

    % Iterate through each bounding box
    for i = 1:numel(stats)
        boundingBox = round(stats(i).BoundingBox);

        % Check if the bounding box width is greater than the threshold
        if boundingBox(3) > maxWidth
            % Reduce the bounding box width to half (split into left and right)
            newWidth = round(boundingBox(3) / 2);

            % Store new bounding box information (left part)
            newBoundingBox{newIndex} = [boundingBox(1), boundingBox(2), newWidth, boundingBox(4)];
            newIndex = newIndex + 1;

            % Store new bounding box information (right part)
            newBoundingBox{newIndex} = [boundingBox(1) + newWidth, boundingBox(2), newWidth, boundingBox(4)];
            newIndex = newIndex + 1;
        else 
            % Bounding box width is not greater than the threshold, store original bounding box information
            newBoundingBox{newIndex} = boundingBox;
            newIndex = newIndex + 1;
        end
    end

    % Initialize index
    newIndex = 10;

    % Display bounding boxes on the original image
    figure;
    imshow(binaryImage);
    hold on;

    % Check if newBoundingBox is not empty
    if ~isempty(newBoundingBox)
        % Iterate through newBoundingBox for uniform cropping and saving
        for i = 1:numel(newBoundingBox)
            boundingBox = round(newBoundingBox{i});
            boundingBox = boundingBox(:)';

            % Check if the bounding box size is greater than or equal to minWidth
            if boundingBox(3) >= minWidth
                % Crop the image
                croppedImage = imcrop(binaryImage, boundingBox);

                % Generate the filename for the cropped image
                outputFileName = sprintf('character%d.png', newIndex);
                newIndex = newIndex + 1;

                % Save the cropped image
                imwrite(croppedImage, fullfile(outputDirectory, outputFileName));

                % Display the bounding box
                x = boundingBox(1);
                y = boundingBox(2);
                width = boundingBox(3);
                height = boundingBox(4);
                rectangle('Position', [x, y, width, height], 'EdgeColor', 'r', 'LineWidth', 1);
            end
        end
    else
        % Handle the case when newBoundingBox is empty, e.g., print a message
        disp('No bounding boxes to display.');
    end

    hold off;
    title('Bounding Boxes on Original Image');
end
