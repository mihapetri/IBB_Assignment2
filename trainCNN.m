% this section of code is used to train the cnn detector on the 
% pictures and their masks

clear
load layers 
counter = 1;
Files=dir('train_set');

for k=3:length(Files)
   fileName = join(['train_set/', Files(k).name]);
   
   mask_name = join(['train_set_mask/', Files(k).name]);
   mask = imread(mask_name);

   [r, c] = find(mask);
   row1 = min(r);
   row2 = max(r);
   col1 = min(c);
   col2 = max(c);
   
   trainData{counter} = fileName;
   trainLabels{counter} = [col1, row1, col2-col1, row2-row1];
   counter = counter + 1;
end

gtData = table(transpose(trainData), transpose(trainLabels));

save gtData.mat gtData

options = trainingOptions('sgdm', ...
        'MiniBatchSize', 64, ...
        'InitialLearnRate', 1e-3, ...
        'LearnRateSchedule', 'piecewise', ...
        'LearnRateDropFactor', 0.1, ...
        'LearnRateDropPeriod', 100, ...
        'MaxEpochs', 50, ...
        'Verbose', true);
    
train = trainRCNNObjectDetector(gtData, layers, options, ...
    'NegativeOverlapRange', [0 0.3], 'PositiveOverlapRange',[0.5 1]);

save train.mat train
