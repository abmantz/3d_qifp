function [featureResults, featureComponentName, featureRootName, ...
    featureConfiguration] = featureStageCompute(globalFeatureConfig, ...
    localState, featureComponent)
%FEATURESTAGECOMPUTE Summary of this function goes here
%   Detailed explanation goes here

% Load Configuration 
localFeatureConfigFunction = str2func([featureComponent '.configuration']);
localFeatureConfig = localFeatureConfigFunction();

% Prepare input initialization parameters
preparedFeatureConfig = ...
    prepareInput(localFeatureConfig.inputArray, ...
                localState, globalFeatureConfig.(featureComponent));

% Find the function name to call 
featureFunctionName = findConfigValue(localFeatureConfig.configArray, ... 
                                                'functionToRun');
featureFunction = str2func([featureComponent '.' ...
    featureFunctionName]);

logger('INFO', ['Calling feature component ' featureComponent]);
% Run the function
try
    featureOutput = featureFunction(preparedFeatureConfig);
catch e 
    featureOutput = struct();
    featureOutput.output = [];
    featureOutput.featureRootName = preparedFeatureConfig.featureRootName;
    logger('ERROR', ['Was not able to run feature component: ' featureComponent]);
end
% Save it to the running cache
featureResults = featureOutput.output;
featureComponentName= featureComponent;
featureRootName = featureOutput.featureRootName;
featureConfiguration = prepareInput(localFeatureConfig.inputArray, ...
                localState, globalFeatureConfig.(featureComponent), true);
end

