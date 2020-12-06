%This section of code is used to test and evaluate the trained model
load train

Files=dir('test_set');

counter = 1;

ldc = labelDefinitionCreator();

%Calculate bounding boxes for test images
for k=3:length(Files)
   fileName = join(['test_set/', Files(k).name]);
   im = imread(fileName);
   
   mask_name = join(['test_set_mask/', Files(k).name]);
   mask = imread(mask_name);

   [r, c] = find(mask);
   row1 = min(r);
   row2 = max(r);
   col1 = min(c);
   col2 = max(c);
   
   testData{counter} = fileName;
   testLabels{counter} = [col1, row1, col2-col1, row2-row1];
   counter = counter + 1;
end

labelNames{1} = "ear";

addLabel(ldc,'ears',labelType.Rectangle);
labelDefs = create(ldc);

imageFilenames = fullfile(toolboxdir('vision'),'visiondata',testData);

result(191,:) = struct('Boxes',[],'Scores',[]);

for i=1:length(testData)
    a = imread(string(testData(i)));
    
    [bbox, score, label] = detect(train, a, 'MiniBatchSize', 32);
    
    [score, idx] = max(score);
    
    bbox2 = testLabels(i);
    bboxList{i} = bbox(idx, :);
    
    area = rectint(bbox(idx, :), bbox2{1,1});
    overlap = bboxOverlapRatio(bbox(idx, :), bbox2{1,1});
    
    overlapList(i) = {overlap};
    
    result(i).Boxes = bbox(idx, :);
    result(i).Scores = score;
    
    
end

results = struct2table(result);

results(end,:) = [];

% change overlap for evaluation
overlap = 0.4;

load gtData

labelNames = {'ears'};
labelData = table(transpose(testLabels),'VariableNames',labelNames);

gTruth = groundTruth(groundTruthDataSource(testData),labelDefs ,labelData);

evaluationData = objectDetectorTrainingData(gTruth,'SamplingFactor', 1, 'WriteLocation', 'EvaluationData');

[ap,recall,precision] = evaluateDetectionPrecision(results,evaluationData(:,2), overlap);

[am,fppi,missRate] = evaluateDetectionMissRate(results, evaluationData(:,2), overlap);

subplot(1,2,1);
plot(recall,precision);
xlabel('Recall');
ylabel('Precision');
title(sprintf('Average precision = %.1f', ap));
grid on

subplot(1,2,2);
loglog(fppi,missRate);
xlabel('False positives per image');
ylabel('Log average miss rate');
title(sprintf('Log average miss rate = %.1f', am));
grid on

% you can change im_num to display different images from test set
im_num = 3;
a = imread(string(testData(im_num)));
bb1 = bboxList(im_num);
bb2 = testLabels(im_num);

detectimg = insertObjectAnnotation(a, 'rectangle', bb2{1,1}, "Ground Truth");

figure; imshow(detectimg)

 rectangle('Position',bb1{1,1},...
'EdgeColor','r','LineWidth',1 )


for i=1:length(overlapList)
   if isempty(overlapList{i}) == 1
       overlapList{i} = 0;
   end
end
disp("average overlap: ")
mean(cell2mat(overlapList))
