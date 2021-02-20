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
% Edited by:        Sarah Mattonen 
%                      - added check to determine if in plane indicies are
%                      out of bounds when adding padding (initially for MMG
%                      images with segmentation at image boundary)
% Edited on:        2019-04-26

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

%% BLARG

dcmFiles = input.DcmImageFileTable.values(); % 1xN cell array
Ndicom = size(dcmFiles);
Ndicom = Ndicom(2);
if Ndicom < 2
    error('BLARG');
end

dicomImageInfo = dicominfo(dcmFiles{2});
Nrows = dicomImageInfo.Rows;
Ncols = dicomImageInfo.Columns;

% do not need this for the moment

% dicomImageArray = zeros(Nrows, Ncols, Ndicom);
% dicomImageInfoArray = cell(Ndicom, 1);
% InstanceNumbers = zeros(1,Ndicom); % these already live in DcmImageFileSeriesLocationsAvailable.values(){1}.instanceNumber, but whatever
% 
% for i = 1:Ndicom
%     dicomImageFile = dcmFiles{i};
%     disp(['Reading ', dicomImageFile])
%     dicomImageSlice = dicomread(dicomImageFile);
%     dicomImageInfo = dicominfo(dicomImageFile);
%     
%     % Sarah moved this here to scale by Intercept and Slope if it exists for each slice 
%     % Also changed to * by slope 
%     if isfield(dicomImageInfo, 'RescaleIntercept')
%          dicomImageSlice = double(dicomImageSlice) + dicomImageInfo.RescaleIntercept;
%     end
%     if isfield(dicomImageInfo, 'RescaleSlope')
%          dicomImageSlice = double(dicomImageSlice) * dicomImageInfo.RescaleSlope;
%     end
%         
%     % Store the cropped image and its info into the image stack 
%     dicomImageArray(:,:,i) = dicomImageSlice;
%     dicomImageInfoArray{i} = dicomImageInfo;
% 
%     InstanceNumbers(i) = dicomImageInfo.InstanceNumber;
% end
% 
% % sort slices by InstanceNumber
% [InstanceNumbers, order] = sort(InstanceNumbers);
% dicomImageArray = dicomImageArray(:,:,order);
% dicomImageInfoArray = dicomImageInfoArray(order);
% 
% % check that the instance numbers are now 1,2,...,Ndicom (without gaps)
% crap = InstanceNumbers - (1:Ndicom);
% if min(crap) ~= 0 || max(crap) ~= 0
%     error('DICOM InstanceNumbers are not 1,2,...,N, and I am too stupid to deal with that.');
% end

% crudely find the Nifti segmentation file
dirname = fileparts(dcmFiles{1});
% why yes, we are resorting to the shell. because stupid.
% RL is shorthand for "was written by Mango"
[segfound, segfile] = system(['ls ' dirname '/SEG*RL.nii*']);
if segfound > 0
    error('Can''t find Nifti segmentation file (SEG*RL.nii in DICOM directory); too stupid to continue.')
end
segfile = strsplit(segfile); % because stupid
segfile = segfile{1}; % because stupid

% read in the Nifti segmentation file
seginfo = niftiinfo(segfile);
segarray = niftiread(segfile);
segsize = size(segarray);
segNrows = segsize(1);
segNcols = segsize(2);
segNslices = segsize(3);


% % % seems that the RAS origin of the segmentation is at size(segarray)/2?
% % % which... should?... coincide with center of the middle DICOM
% % % figure out where this is in DICOM RAS to get the offset
% % k = round(Ndicom/2);
% % % k = floor(Ndicom/2);
% % origin = dicomImageInfoArray{k}.ImagePositionPatient';
% % basisX = dicomImageInfoArray{k}.ImageOrientationPatient(1:3)';
% % basisY = dicomImageInfoArray{k}.ImageOrientationPatient(4:6)';
% % dcmij = 0.5*[Ncols Nrows]; % NB X is 2nd index (column), Y is 1st
% % % Below seems like it should have ij-1 in it, but empirically, it really
% % % really doesn't look like it. Indices are the worst.
% % dcmXY = [(dcmij(2))*dicomImageInfoArray{k}.PixelSpacing(2) (dcmij(1))*dicomImageInfoArray{k}.PixelSpacing(1)]; % offset from XY origin in mm
% % dcmLPS = origin + double(dcmXY)*[basisX; basisY];
% % dcmOriginRAS = [-dcmLPS(1) -dcmLPS(2) dcmLPS(3)];
% % % dcmOriginRAS = 0.5 * [-dcmLPS(1) -dcmLPS(2) dcmLPS(3)];
% % % k = ceil(Ndicom/2);
% % % origin = dicomImageInfoArray{k}.ImagePositionPatient';
% % % basisX = dicomImageInfoArray{k}.ImageOrientationPatient(1:3)';
% % % basisY = dicomImageInfoArray{k}.ImageOrientationPatient(4:6)';
% % % dcmij = 0.5*[Ncols Nrows]; % NB X is 2nd index (column), Y is 1st
% % % % Below seems like it should have ij-1 in it, but empirically, it really
% % % % really doesn't look like it. Indices are the worst.
% % % dcmXY = [(dcmij(2))*dicomImageInfoArray{k}.PixelSpacing(2) (dcmij(1))*dicomImageInfoArray{k}.PixelSpacing(1)]; % offset from XY origin in mm
% % % dcmLPS = origin + double(dcmXY)*[basisX; basisY];
% % % dcmOriginRAS = dcmOriginRAS + 0.5 * [-dcmLPS(1) -dcmLPS(2) dcmLPS(3)];
% 
% % % possibly better
% % % seems that the origin of the segmentation is at the center of the array
% % % assume this coincides with the dead center of the scan
% % axis = dicomImageInfoArray{Ndicom}.ImagePositionPatient' - ...
% %     dicomImageInfoArray{1}.ImagePositionPatient';
% % k = 1;
% % basisX = dicomImageInfoArray{k}.ImageOrientationPatient(1:3)';
% % basisY = dicomImageInfoArray{k}.ImageOrientationPatient(4:6)';
% % dcmLPS = dicomImageInfoArray{k}.ImagePositionPatient' + 0.5*axis + ...
% %     0.5*double(Ncols-1)*dicomImageInfoArray{k}.PixelSpacing(1)*basisX + ...
% %     0.5*double(Nrows-1)*dicomImageInfoArray{k}.PixelSpacing(2)*basisY;
% % dcmOriginRAS = [-dcmLPS(1) -dcmLPS(2) dcmLPS(3)];
% 
% % actually, the original approach seems closer...
% segorigin = transformPointsInverse(seginfo.Transform, [0 0 0]) + 1;
% axis = (dicomImageInfoArray{Ndicom}.ImagePositionPatient' - ...
%     dicomImageInfoArray{1}.ImagePositionPatient') / double(Ndicom-1);
% k = 1;%round(segorigin(3));
% origin = dicomImageInfoArray{k}.ImagePositionPatient';
% basisX = dicomImageInfoArray{k}.ImageOrientationPatient(1:3)';
% basisY = dicomImageInfoArray{k}.ImageOrientationPatient(4:6)';
% dcmXY = [(segorigin(1)-1)*dicomImageInfoArray{k}.PixelSpacing(2) (segorigin(2)-1)*dicomImageInfoArray{k}.PixelSpacing(1)];
% dcmLPS = origin + dcmXY*[basisX; basisY] + (segorigin(3)-k)*axis;
% dcmOriginRAS = [-dcmLPS(1) -dcmLPS(2) dcmLPS(3)];
% 
% 
% % Find the appropriate segmentation value for each voxel in the DICOM data.
% % Convert DICOM voxel coordinates to LPS, then to RAS, then to segmentation
% % voxel coordinates (and then Matlab array indices).
% segmentation = dicomImageArray * 0;
% [dcmi, dcmj] = meshgrid(1:Nrows, 1:Ncols); % should probably swap cols/rows?
% dcmij = double([dcmi(:), dcmj(:)]);
% clearvars dcmi dcmj
% nij = double(Nrows) * double(Ncols); % double to avoid overflowing short int
% disp('Interpolating segmentation to DICOM coordinates')
% for k = 1:Ndicom
%     origin = dicomImageInfoArray{k}.ImagePositionPatient';
%     basisX = dicomImageInfoArray{k}.ImageOrientationPatient(1:3)';
%     basisY = dicomImageInfoArray{k}.ImageOrientationPatient(4:6)';
%     % NB X is 2nd index (column), Y is 1st
%     % Below seems like it should have ij-1 in it, but empirically, it really
%     % really doesn't look like it. Indices are the worst.
%     dcmXY = [(dcmij(:,2)-1)*dicomImageInfoArray{k}.PixelSpacing(2) (dcmij(:,1)-1)*dicomImageInfoArray{k}.PixelSpacing(1)]; % offset from XY origin in mm
%     dcmLPS = repmat(origin, nij, 1) + dcmXY*[basisX; basisY];
%     dcmRAS = [-dcmLPS(:,1) -dcmLPS(:,2) dcmLPS(:,3)] - dcmOriginRAS;
%     dcmSEG = transformPointsInverse(seginfo.Transform, dcmRAS) + 1; % again seems to be better w/o accounting for 1-based indices...?
%     segmask = interp3(segarray, dcmSEG(:,2), dcmSEG(:,1), dcmSEG(:,3), 'nearest', 0); % again the X/Y column/row thing
%     %segmask = interp3(double(segarray), dcmSEG(:,2), dcmSEG(:,1), dcmSEG(:,3), 'linear', 0); % again the X/Y column/row thing
%     segmentation(:,:,k) = reshape(segmask, Ncols, Nrows)'; % AGAIN the X/Y column/row 
% end
% 
% % sanity check that the segmentation is aligned with the data
% % to be either eliminated or written to disk automatically
% [~, i] = max(sum(segmentation, [1 2])); % slice with the most segmented junk in it
% img = zeros(Nrows, Ncols, 3);
% img(:,:,1) = dicomImageArray(:,:,i);
% img(:,:,2) = dicomImageArray(:,:,i);
% img(:,:,3) = dicomImageArray(:,:,i);
% img = (img - min(img, [],'all')) / (max(img, [], 'all') - min(img, [], 'all'));
% img(:,:,2) = img(:,:,2) .* (1.0 - cast(segmentation(:,:,i), 'double'));
% image(img)

% read in the data in Nifti format, as converted from DICOM by Mango
ninfo=niftiinfo([dirname '/mri.nii.gz']);
nii=niftiread([dirname '/mri.nii.gz']);
niisize = size(nii);

% almost certainly there is no coordinate transformation is needed
if min(seginfo.Transform.T == ninfo.Transform.T, [], 'all') == 0
    disp('Interpolating segmentation to data coordinates')
    % not entirely sure X and Y are in the right order everywhere here
    niiorigin = transformPointsInverse(ninfo.Transform, [0 0 0]);
    niiRASorigin = transformPointsForward(ninfo.Transform, niiorigin);
    [niii, niij, niik] = meshgrid(1:niisize(1), 1:niisize(2), 1:niisize(3));
    niiIJK = [niii(:), niij(:), niik(:)];
    clearvars niii niij niik
    niiRAS = transformPointsForward(ninfo.Transform, niiIJK) - niiRASorigin;
    niiSEG = transformPointsInverse(seginfo.Transform, niiRAS);
    niisegmask = interp3(segarray, niiSEG(:,1), niiSEG(:,2), niiSEG(:,3), 'nearest', 0);
    NIIsegmentation = reshape(niisegmask, niisize(1), niisize(2), niisize(3));
else
    NIIsegmentation = segarray;
end

[~, i] = max(sum(NIIsegmentation, [1 2])); % slice with the most segmented junk in it
img = zeros(Ncols, Nrows, 3);
img(:,:,1) = nii(:,:,i);
img(:,:,2) = nii(:,:,i);
img(:,:,3) = nii(:,:,i);
img = (img - min(img, [],'all')) / (max(img, [], 'all') - min(img, [], 'all'));
img(:,:,2) = img(:,:,2) .* (1.0 - cast(NIIsegmentation(:,:,i), 'double'));
%image(img)
imwrite(img, [dirname '/segcheck.jpg']);

error('Breakpoints you say? Humbug!')


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

if size(dicomSegmentationObjectMask, 3) ~= numSlicesDSO
    error('Number of DICOMs referenced in the DSO header is not equal to the height of the pixel array');
end

%% Find Z vector direction
if dicomImageInfo.Modality == 'MG'
    zVector = 0;
else
    imageOrientation = dicomImageInfo.ImageOrientationPatient;
    dc = zeros(2,3);
    for row=1:2
        for col=1:3
            dc(row,col) = imageOrientation((row-1)*3+col);
        end
    end
    zVector =cross(dc(1,:), dc(2,:));
end
% Also save instance numbers and acquisition times for helping with loading
% later on.
dicomAcquisitionTimes = double.empty(numSlicesDSO, 0);
dicomInstanceNumbers = double.empty(numSlicesDSO, 0);

for nSDSO = 1:numSlicesDSO
    tmpSDSO = dicomSegmentationObjectInfo. ...
        ReferencedSeriesSequence.Item_1.ReferencedInstanceSequence. ...
        (['Item_' num2str(nSDSO)]).ReferencedSOPInstanceUID;

    tmpDicomImageInfo2 = dicominfo(dcmImageFileArray(tmpSDSO));
    
    if dicomImageInfo.Modality == 'MG'
        zResolutions(nSDSO) = 0;
    else
        zResolutions(nSDSO) = zVector * tmpDicomImageInfo2.ImagePositionPatient;
    end
    
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
if dicomImageInfo.Modality == 'MG'
    yDicomSegmentationResolution = dicomImageInfo.ImagerPixelSpacing(1);
    xDicomSegmentationResolution = dicomImageInfo.ImagerPixelSpacing(2);
else
    yDicomSegmentationResolution = dicomImageInfo.PixelSpacing(1);
    xDicomSegmentationResolution = dicomImageInfo.PixelSpacing(2);
end

% Z voxel spacing determined by the minimum distance between slices
if (numel(zResolutions) > 1)
    zDicomSegmentationResolution = min(abs(diff(zResolutions)));
elseif dicomImageInfo.Modality == 'MG'
    zDicomSegmentationResolution = 1;
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
        if ((round(str2double(missingSliceInfo.acquisitionTime)) < minAcquisitionTime) || ...
                round(str2double(missingSliceInfo.acquisitionTime)) > maxAcquisitionTime )
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
%numberofuids = numel(fieldnames(dicomSegmentationObjectInfo.ReferencedSeriesSequence.Item_1.ReferencedInstanceSequence));
%firstDicomUid = dicomSegmentationObjectInfo.ReferencedSeriesSequence.Item_1.ReferencedInstanceSequence.(['Item_' num2str(1)]).ReferencedSOPInstanceUID;
%lastDicomUid  = dicomSegmentationObjectInfo.ReferencedSeriesSequence.Item_1.ReferencedInstanceSequence.(['Item_' num2str(numberofuids)]).ReferencedSOPInstanceUID;


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

% Check added by Sarah to eliminate negative or out of bound indices if there isn't enough room for in plane padding. 
if dicomSegmentationObjectYFirstIndex < 1
    dicomSegmentationObjectYFirstIndex = 1;
end

if dicomSegmentationObjectYLastIndex > size(dicomSegmentationObjectMask,1)
    dicomSegmentationObjectYLastIndex = size(dicomSegmentationObjectMask,1);
end

if dicomSegmentationObjectXFirstIndex < 1
    dicomSegmentationObjectXFirstIndex = 1;
end

if dicomSegmentationObjectXLastIndex > size(dicomSegmentationObjectMask,2)
    dicomSegmentationObjectXLastIndex = size(dicomSegmentationObjectMask,2);
end

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
outputStructure.segmentationInfo.segmentationOrigin = ...
    [dicomSegmentationObjectYFirstIndex, ...
    dicomSegmentationObjectXFirstIndex, ...
    dicomSegmentationObjectZFirstIndexOrig];
end
