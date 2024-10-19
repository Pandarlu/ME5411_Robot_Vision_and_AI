
nn_directory = 'TrainedCNN_enhance';
name_nn = "CNN_2";
test_dataset_directory = 'img_output\Cropped_paddingCharaters';

nn_fullpath = fullfile(nn_directory, name_nn);
load(nn_fullpath, 'net');

imagefiles = dir(fullfile(test_dataset_directory, '*.png'));
nfiles = length(imagefiles);

results = struct('Filename', {}, 'Label', {}, 'Scores', {});

for i = 1:nfiles
    current_filename = fullfile(test_dataset_directory, imagefiles(i).name);
    current_image = imread(current_filename);

    [label, scores] = classify(net, current_image);

    results(i).Filename = current_filename;
    results(i).Label = label;
    results(i).Scores = scores;
end

figure;

for i = 1:nfiles
    subplot(2, ceil(nfiles/2), i);

    current_image = imread(results(i).Filename);
    
    imshow(current_image);
    title(["Classified: " results(i).Label "Accuracy: " num2str(100 * max(results(i).Scores), 4)+"%"]);
end

