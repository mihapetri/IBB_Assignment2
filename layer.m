inputLayer = imageInputLayer([64 64 1]);

filterSize = [5 5];
numFilters = 32;

middleLayers = [
    
convolution2dLayer(filterSize,numFilters,'Padding',2)

reluLayer()

maxPooling2dLayer(3,'Stride',2)

convolution2dLayer(filterSize,numFilters,'Padding',2)
reluLayer()
averagePooling2dLayer(3, 'Stride',2)

convolution2dLayer(filterSize,2 * numFilters,'Padding',2)
reluLayer()
averagePooling2dLayer(3,'Stride',2)

];
finalLayers = [

fullyConnectedLayer(64)

reluLayer

fullyConnectedLayer(2)

softmaxLayer
classificationLayer
];

layers = [
    inputLayer
    middleLayers
    finalLayers
    ];

save layers.mat layers