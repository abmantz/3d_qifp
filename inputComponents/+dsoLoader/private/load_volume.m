function  outputStructure = load_volume(input)
% LOAD_VOLUME this function receives a Dicom Segmentation file and an table
%   with the Dicom Images files, then it loads a region of interest
%   containing the segmented values. It returns a 3D matrix.
%
% Input:
%   configurationArray      
%
% Output:
%
%
% Created by:       Sebastian Echegaray 
% Created on:       2013-04-02
% Edited by:        Sarah Mattonen 
%                      - moved scaling by slope and intercept to immediately after dicom read
%                      - cast the cropped image to double when rescaling
%                      - updated to scale by slice specific slope and intercept
% Edited on:        2017-04-06

%% Set custom dicom dictionary
dicomDictPath = strcat(strrep(which(mfilename),[mfilename '.m'],''), 'dicom-dict.txt');
dicomdict('set', dicomDictPath);

%% Initialization
% Copying input parameters to local variables to avoid reuse
DcmSegmentationObjectFileTable = input.DcmSegmentationObjectFileTable;
dsoUid = input.processingUid;
dicomSegmentationObjectFile = DcmSegmentationObjectFileTable(char(dsoUid));
dcmImageFileArray = input.DcmImageFileTable;
DcmImageFileSeriesNumberArray = input.DcmImageFileSeriesNumber;

DcmImageFileSeriesLocation = input.DcmImageFileSeriesLocation;
DcmImageFileSeriesLocationsAvailable = ...
 input.DcmImageFileSeriesLocationsAvailable;


configurationArray = struct();
configurationArray.LOAD_VOLUME_PADDING = input.padding;

%% Configuration and Validation

% Check that the padding is set
if ~isfield(configurationArray, 'LOAD_VOLUME_PADDING') 
    error('LOAD_VOLUME_PADDING was not defined. Dicom images cannot be load');
end

% Check if there the Padding is defined as one-dimension or 3-dimensional
if numel(configurationArray.LOAD_VOLUME_PADDING) == 1
    LOAD_VOLUME_PADDING = ...
        repmat(configurationArray.LOAD_VOLUME_PADDING, [3,1]);
elseif numel(configurationArray.LOAD_VOLUME_PADDING) == 3
    LOAD_VOLUME_PADDING = configurationArray.LOAD_VOLUME_PADDING;
else     
    error('LOAD_VOLUME_PADDING was expected to have 1 or 3 elements.');
end

%% Load Dicom Segmentation Value and Info
dicomSegmentationObjectMask = squeeze(dicomread(dicomSegmentationObjectFile));
dicomSegmentationObjectInfo = dicominfo(dicomSegmentationObjectFile);

% Instance ID of the second image in the original dicom series) 
% dicomImageSopInstanceUid = dicomSegmentationObjectInfo. ...
%     SharedFunctionalGroupsSequence.Item_1.DerivationImageSequence. ...
%     Item_1.SourceImageSequence.(['Item_' num2str(2)]). ...
%     ReferencedSOPInstanceUID;
dicomImageSopInstanceUid = dicomSegmentationObjectInfo. ...
    ReferencedSeriesSequence.Item_1.ReferencedInstanceSequence. ...
    (['Item_' num2str(1)]).ReferencedSOPInstanceUID;


% Metadata of the original image
dicomImageInfo = dicominfo(dcmImageFileArray(dicomImageSopInstanceUid));

% Get and extract image positions.
% numSlicesDSO = numel(fieldnames(dicomSegmentationObjectInfo. ...
%     SharedFunctionalGroupsSequence.Item_1.DerivationImageSequence. ...
%     Item_1.SourceImageSequence));
numSlicesDSO = numel(fieldnames(dicomSegmentationObjectInfo. ...
    ReferencedSeriesSequence.Item_1.ReferencedInstanceSequence));
zResolutions = zeros(numSlicesDSO,1);

%% Find Z vector direction
imageOrientation = dicomImageInfo.ImageOrientationPatient;
dc = zeros(2,3);
for row=1:2
    for col=1:3
        dc(row,col) = imageOrientation((row-1)*3+col);
    end
end
zVector =cross(dc(1,:), dc(2,:));

% Also save instance numbers and acquisition times for helping with loading
% later on.
dicomAcquisitionTimes = double.empty(numSlicesDSO, 0);
dicomInstanceNumbers = double.empty(numSlicesDSO, 0);

for nSDSO = 1:numSlicesDSO
    tmpSDSO = dicomSegmentationObjectInfo. ...
        ReferencedSeriesSequence.Item_1.ReferencedInstanceSequence. ...
        (['Item_' num2str(nSDSO)]).ReferencedSOPInstanceUID;

    tmpDicomImageInfo2 = dicominfo(dcmImageFileArray(tmpSDSO));
    zResolutions(nSDSO) = zVector * tmpDicomImageInfo2.ImagePositionPatient;
    
    % If acquisitionTime exists:
    if (isfield(tmpDicomImageInfo2, 'AcquisitionTime'))
        acquisitionTime = tmpDicomImageInfo2.AcquisitionTime;
        % And is not unknown
        if (~(strcmp(acquisitionTime, 'unknown') || isempty(acquisitionTime)))
            dicomAcquisitionTimes(nSDSO) = int32(str2double(acquisitionTime));
        end
    end
    
    % If instance number exists:
    if (isfield(tmpDicomImageInfo2, 'InstanceNumber'))
        instanceNumber = tmpDicomImageInfo2.InstanceNumber;
        % And is not unknown
        if (~(strcmp(instanceNumber, 'unknown') || isempty(instanceNumber)))
            dicomInstanceNumbers(nSDSO) = int32(str2double(instanceNumber));
        end
    end    
    
    if ((nSDSO > 1) && (zResolutions(nSDSO) == zResolutions(nSDSO-1)))
        continue
    end
end
nonSortedzResolutions = zResolutions;
% Sort image positions
zResolutions = sort(zResolutions);

% Find X and Y pixel determined
yDicomSegmentationResolution = dicomImageInfo.PixelSpacing(1);
xDicomSegmentationResolution = dicomImageInfo.PixelSpacing(2);

% Z voxel spacing determined by the minimum distance between slices
if (numel(zResolutions) > 1)
    zDicomSegmentationResolution = min(abs(diff(zResolutions)));
else
    zDicomSegmentationResolution = dicomImageInfo.SpacingBetweenSlices;
end
% Ugly hack to see if the DSO slices are within 10% of the mean distance
if any(((diff(zResolutions) - mean(diff(zResolutions)))/mean(diff(zResolutions))) > 0.1)
    warning('DSO has non-contiguous slices');
end

% Available locations
locationsAvailable = ... 
    DcmImageFileSeriesLocationsAvailable(dicomImageInfo.SeriesInstanceUID);

directedZLocationsAvailable = [locationsAvailable.directedZ];

% Find all locations that need to be loaded
minLocation = min(zResolutions);
maxLocation = max(zResolutions);
inbetweenLocationsMask = ...
    (directedZLocationsAvailable >= minLocation) & ...
    (directedZLocationsAvailable <= maxLocation);

unsortedInbetweenDirectedZLocations =  ...
                      directedZLocationsAvailable(inbetweenLocationsMask);
unsortedInbetweenLocations = locationsAvailable(inbetweenLocationsMask);

[inbetweenDirectedZLocations, inbetweenDirectedZLocationsIndex] = ...
        sort(unsortedInbetweenDirectedZLocations);

inbetweenLocations = ...
    unsortedInbetweenLocations(inbetweenDirectedZLocationsIndex);

[missingSlices, missingSlicesIndex]  = ...
    setdiff(inbetweenDirectedZLocations, nonSortedzResolutions);

% For all missing slices in the DSO:
zResolutions = zResolutions(end:-1:1);
nMissingSlices = numel(missingSlices);
slicesAdded  = nMissingSlices;

% Sanity checks for inbetween slices
minAcquisitionTime = min(dicomAcquisitionTimes);
maxAcquisitionTime = max(dicomAcquisitionTimes);
minInstanceNumber = min(dicomInstanceNumbers);
maxInstanceNumber = max(dicomInstanceNumbers);

for iMissingSlice = 1:nMissingSlices
    % Check if missing slice location is for the correct phase.
    missingSliceInfo = ...
        inbetweenLocations(missingSlicesIndex(iMissingSlice));
    % Check if Acquisition Time is available.
    if (isfield(missingSliceInfo, 'acquisitionTime'))
        if ((str2double(missingSliceInfo.acquisitionTime) < minAcquisitionTime) || ...
                str2double(missingSliceInfo.acquisitionTime) > maxAcquisitionTime )
            disp('Multiple phases found in the series.');
            slicesAdded = slicesAdded - 1;
            continue;
        end
    elseif (isfield(missingSliceInfo, 'instanceNumber'))
        warning('Acquisition time not available in missing slice, using instance number to group phases');
        if ((int32(str2double(missingSliceInfo.instanceNumber)) < minInstanceNumber) || ...
                int32(str2double(missingSliceInfo.instanceNumber)) > maxInstanceNumber )
            disp('Multiple phases found in the series.');
            slicesAdded = slicesAdded - 1;
            continue;
        end
    else
        warning('Couldn''t find acquisition time nor instance number in the metadata to separate slices, possible corruption in results');
    end
    
    missingSlice = missingSlices(iMissingSlice);
    insertPlace = find(diff((zResolutions - missingSlice) > 0));
    zResolutions = [zResolutions(1:insertPlace); ...
        missingSlice; ...
        zResolutions((insertPlace+1):end)];
    dicomSegmentationObjectMask = cat(3, ...
        dicomSegmentationObjectMask(:,:,1:insertPlace), ...
        false(size(dicomSegmentationObjectMask(:,:,1))), ....
        dicomSegmentationObjectMask(:,:,(insertPlace+1):end) ...
        );
    
end


%% Lets calculate the padding from mm to voxels.
yVolumePaddingInVoxels = ceil(LOAD_VOLUME_PADDING(1) ./ ...
    yDicomSegmentationResolution);
xVolumePaddingInVoxels = ceil(LOAD_VOLUME_PADDING(2) ./ ...
    xDicomSegmentationResolution);
zVolumePaddingInVoxels = ceil(LOAD_VOLUME_PADDING(3) ./ ...
    zDicomSegmentationResolution);

%% Lets find the bounding box for the segmentation
dicomSegmentationObjectZIndexArray = ...
    find(squeeze(sum(sum(dicomSegmentationObjectMask, 1), 2)));
dicomSegmentationObjectXIndexArray = ...
    find(squeeze(sum(sum(dicomSegmentationObjectMask, 3), 1)));
dicomSegmentationObjectYIndexArray = ...
    find(squeeze(sum(sum(dicomSegmentationObjectMask, 3), 2)));

% Z-Index

% firstDicomUid = dicomSegmentationObjectInfo.SharedFunctionalGroupsSequence.Item_1.DerivationImageSequence.Item_1.SourceImageSequence.(['Item_' num2str(dicomSegmentationObjectZIndexArray(1))]).ReferencedSOPInstanceUID;
% lastDicomUid  = dicomSegmentationObjectInfo.SharedFunctionalGroupsSequence.Item_1.DerivationImageSequence.Item_1.SourceImageSequence.(['Item_' num2str(dicomSegmentationObjectZIndexArray(end) - slicesAdded)]).ReferencedSOPInstanceUID;

firstDicomUid = dicomSegmentationObjectInfo.ReferencedSeriesSequence.Item_1.ReferencedInstanceSequence.(['Item_' num2str(dicomSegmentationObjectZIndexArray(1))]).ReferencedSOPInstanceUID;
lastDicomUid  = dicomSegmentationObjectInfo.ReferencedSeriesSequence.Item_1.ReferencedInstanceSequence.(['Item_' num2str(dicomSegmentationObjectZIndexArray(end) - slicesAdded)]).ReferencedSOPInstanceUID;


firstDicomImageInfo = dicominfo(dcmImageFileArray(firstDicomUid));
seriesUid = dicomImageInfo.SeriesInstanceUID;
lastDicomImageInfo = dicominfo(dcmImageFileArray(lastDicomUid));

if firstDicomImageInfo.InstanceNumber < lastDicomImageInfo.InstanceNumber
    signFlag = +1;
else 
    signFlag = -1;
end

dicomSegmentationObjectZFirstIndex = firstDicomImageInfo.InstanceNumber - (signFlag *zVolumePaddingInVoxels);
dicomSegmentationObjectZLastIndex = lastDicomImageInfo.InstanceNumber + (signFlag * zVolumePaddingInVoxels);

dicomSegmentationObjectZFirstIndexOrig = dicomSegmentationObjectZIndexArray(1);
dicomSegmentationObjectZLastIndexOrig  = dicomSegmentationObjectZIndexArray(end);

% Rows 
dicomSegmentationObjectYFirstIndex = dicomSegmentationObjectYIndexArray(1) - ...
    yVolumePaddingInVoxels;
dicomSegmentationObjectYLastIndex  = dicomSegmentationObjectYIndexArray(end) + ...
    yVolumePaddingInVoxels;

% Columns
dicomSegmentationObjectXFirstIndex = dicomSegmentationObjectXIndexArray(1) - ...
    xVolumePaddingInVoxels;
dicomSegmentationObjectXLastIndex  = dicomSegmentationObjectXIndexArray(end) + ...
    xVolumePaddingInVoxels;

%% Lets Load the Dicom Images. 

% Lets initialize the result array
dicomImageArray = zeros( ...
    dicomSegmentationObjectYLastIndex - dicomSegmentationObjectYFirstIndex + 1, ...
    dicomSegmentationObjectXLastIndex - dicomSegmentationObjectXFirstIndex + 1, ...
    abs(dicomSegmentationObjectZLastIndex - dicomSegmentationObjectZFirstIndex) + 1 ...
    );
dicomImageInfoArray = cell(...
    abs(dicomSegmentationObjectZLastIndex - dicomSegmentationObjectZFirstIndex) + 1, ...
    1);

skippedSlicesIndex = [];
for dicomSegmentationObjectSliceNo = ...
        dicomSegmentationObjectZFirstIndex:signFlag:dicomSegmentationObjectZLastIndex
    
    % Load the slice referred by the Dicom Segmentation Object Info
    try
        % Convert the index in the loop to a 1..numel index
        dicomImageIndex = dicomSegmentationObjectSliceNo - ...
            min(dicomSegmentationObjectZFirstIndex, dicomSegmentationObjectZLastIndex) + 1;

        dicomImageFile = DcmImageFileSeriesNumberArray([seriesUid '-' num2str(dicomSegmentationObjectSliceNo)]);
        dicomImageSlice = dicomread(dicomImageFile);
        dicomImageInfo = dicominfo(dicomImageFile);   

        % Crop the Slice into a ROI
        dicomImageSliceCropped = dicomImageSlice(...
            dicomSegmentationObjectYFirstIndex:dicomSegmentationObjectYLastIndex, ...
            dicomSegmentationObjectXFirstIndex:dicomSegmentationObjectXLastIndex);

        % Sarah moved this here to scale by Intercept and Slope if it exists for each slice 
        % Also changed to * by slope 
        if isfield(dicomImageInfo, 'RescaleIntercept')
             dicomImageSliceCropped = double(dicomImageSliceCropped) + dicomImageInfo.RescaleIntercept;
        end
        if isfield(dicomImageInfo, 'RescaleSlope')
             dicomImageSliceCropped = double(dicomImageSliceCropped) * dicomImageInfo.RescaleSlope;
        end
        
        
        % Store the cropped image and its info into the image stack 
        dicomImageArray(:,:,dicomImageIndex) = dicomImageSliceCropped;
        dicomImageInfoArray{dicomImageIndex} = dicomImageInfo;
    catch
        warning('Not enough space for full padding');
        skippedSlicesIndex = [skippedSlicesIndex, dicomImageIndex];
    end
end

% Remove Blank slices
if ~isempty(skippedSlicesIndex)
    dicomImageInfoArray(skippedSlicesIndex) = [];
    dicomImageArray(:,:,skippedSlicesIndex) = [];
end

%% Find the new segmentation mask to fit the cropped volume
dicomSegmentationObjectCropped = ...
    dicomSegmentationObjectMask(...
    dicomSegmentationObjectYFirstIndex:dicomSegmentationObjectYLastIndex, ...
    dicomSegmentationObjectXFirstIndex:dicomSegmentationObjectXLastIndex, ...
    dicomSegmentationObjectZFirstIndexOrig:dicomSegmentationObjectZLastIndexOrig  ...    
    );

padSliceMask = zeros(size(dicomSegmentationObjectCropped,1), size(dicomSegmentationObjectCropped,2), 'uint8');
padMask = repmat(padSliceMask, [1, 1, zVolumePaddingInVoxels]);
dicomSegmentationObjectCropped = cat(3, padMask, dicomSegmentationObjectCropped, padMask);

% Remove Blank slices
if ~isempty(skippedSlicesIndex)
    if signFlag == -1
        dicomSegmentationObjectCropped(:,:,end - skippedSlicesIndex + 1) = [];
    else
        dicomSegmentationObjectCropped(:,:,skippedSlicesIndex) = [];
    end
end

%% Create the result
outputStructure.intensityVOI = dicomImageArray;

% Scale by Intercept and Slope if it exists
% if isfield(dicomImageInfoArray{1}, 'RescaleIntercept')
%     outputStructure.intensityVOI = outputStructure.intensityVOI + ...
%         dicomImageInfoArray{1}.RescaleIntercept;
% end
% if isfield(dicomImageInfoArray{1}, 'RescaleSlope')
%     outputStructure.intensityVOI = outputStructure.intensityVOI / ...
%         dicomImageInfoArray{1}.RescaleSlope;
% end

outputStructure.infoVOI = dicomImageInfoArray;
outputStructure.infoVOI{1}.zResolution = zDicomSegmentationResolution;

try
    if signFlag > 0 
        outputStructure.segmentationVOI = dicomSegmentationObjectCropped;        
    else
        outputStructure.segmentationVOI = flip(dicomSegmentationObjectCropped, 3);
    end
catch
    outputStructure.segmentationVOI = dicomSegmentationObjectCropped;        
end
outputStructure.segmentationVOI = logical(outputStructure.segmentationVOI);
outputStructure.segmentationInfo = dicomSegmentationObjectInfo;
outputStructure.segmentationInfo.zResolution = zDicomSegmentationResolution;
end
