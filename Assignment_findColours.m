%load all the png file and solution .mat files. Run findColours on each one
%and see if the results match the actual answer. The calculate the overall
%score.

function colordetect = findColours(filename, varargin)
    % Load the solution file or generate random colors
    if ~isempty(varargin)
        % If solution file provided, load it
        mat_filename = varargin{1};
        fprintf('Loading %s\n', mat_filename);
        load(mat_filename, 'res');
    else
        % Generate random colors if no solution file provided
        res = rand(4, 4, 3); % Random colors in a 4x4 grid
    end
    
    % Load the image
    image = loadImage(filename);
    
    % Display image name
    fprintf('Processing image: %s\n', filename);
    
    % Display original image
    figure;
    subplot(1, 2, 1);
    imshow(image);
    title('Original Image');
    
    % Find circles in the image
    circleCoordinates = findCircles(image);
    
    % Check the filename to determine the operation
    if contains(filename, 'noise_') || contains(filename, 'org_')
        % Get colors directly from the original or noisy image
        result = getColours(image);
    elseif contains(filename, 'proj_') || contains(filename,'rot_')
        % Correct distortion and then get colors
        correctedImage = correctImage(circleCoordinates, image);
        result = getColours(correctedImage);
        
        % Display corrected image
        subplot(1, 2, 2);
        imshow(correctedImage);
        title('Corrected Image');
    else
        % Handle other cases where the filename format is not recognized
        disp('Wrong format or distorted image');
        result = [];
    end
    
    % Display the result
    disp(result)
    colordetect = result;
end

    


%reference : https://ww2.mathworks.cn/help/matlab/ref/im2double.html#s

function image=loadImage(filename)
image= imread(filename); %read image file
image=im2double(image);
% %change the image type to double
end

function circleCoordinates=findCircles(image)
%converts ti grayscale as its a color image
grayimg=rgb2gray(image); 
%calculate threshold on grayimg
threshold=graythresh(grayimg); 
%create binary image based on threshold
bin_img= imbinarize(grayimg,threshold);
%inverts the binary image to find dark circles on a light background
inv_bin_img=imcomplement(bin_img); 
%identify connected components in the inverted binary image
%calculate the area of each connected components
connected_components=bwconncomp(inv_bin_img);
areas=cellfun(@numel,connected_components.PixelIdxList);
%descending sorting
[area_sort,indices_sort]=sort(areas,'descend');
% Getting the coordinates of the first four largest black blobs
num_blobs = 5;
blob_coords = zeros(num_blobs, 2);
for i = 2:num_blobs
    blob_indices = connected_components.PixelIdxList{indices_sort(i)};
    [rows, cols] = ind2sub(size(inv_bin_img), blob_indices);
    blob_coords(i, :) = [ mean(cols),mean(rows)];
end
% Removing the first coordinate from the blob_coords matrix
blob_coords(1, :) = [];


% Sort the coordinates in clockwise order starting from bottom-left
sortedCoordinates = sortrows(blob_coords);

if sortedCoordinates(2,2) < sortedCoordinates(1,2)
    % If the second coordinate is below the first, swap them
    sortedCoordinates([1 2],:) = sortedCoordinates([2 1],:);
end

if sortedCoordinates(4,2) > sortedCoordinates(3,2)
    % If the fourth coordinate is above the third, swap them
    sortedCoordinates([3 4],:) = sortedCoordinates([4 3],:);
end

circleCoordinates=sortedCoordinates;


end


%reference : https://ww2.mathworks.cn/help/images/ref/fitgeotrans.html?#d126e115628
function outputImage = correctImage(Coordinates, image)

% Define a fixed box with coordinates
boxf = [[0 ,0]; [0 ,480];[480 ,480]; [480 ,0]];

% Calculating the transformation matrix from the given Coordinates to 
% transform the matrix to the fixed box using projective transformation
TF = fitgeotrans(Coordinates,boxf,'projective');

% Create an image reference object with the size of the input image
outview = imref2d(size(image));

% Apply the calculated transformation matrix to the input image
% and create a new image with fill value 255 (white) outside the boundaries of the input image
B = imwarp(image,TF,'fillvalues',255,outputview=outview);

% Crop the image to a size of 480x480
B = imcrop(B,[0 0 480 480]);

% Try to suppress the glare in the image using flat-field correction
B = imflatfield(B,40);

% Adjust the levels of the image to improve contrast
B = imadjust(B,[0.4 0.65]);

% Assign the corrected image to the outputImage variable
outputImage = B;
end




%reference : https://ww2.mathworks.cn/matlabcentral/answers/1827523-how-to-recognize-6-colors-of-a-face-rubik-s-cube-at-the-same-time?s_tid=prof_contriblnk
% gets the array of colours from the image
function colours=getColours(image)

% Convert the image to uint8 format
W=im2uint8(image);

% Median filter to suppress noise
W = medfilt3(W,[7 7 1]);

% Increase contrast
W = imadjust(W,stretchlim(W,0.025));

% Convert the RGB image to grayscale and threshold
Conimage = rgb2gray(W)>20;

% Remove positive specks from binary image
Conimage = bwareaopen(Conimage,100);

% Remove negative specks from binary image
Conimage = ~bwareaopen(~Conimage,100);

% Remove outer white region
Conimage = imclearborder(Conimage);

% Erode image
Conimage = imerode(Conimage,ones(10));

% Segmenting the image
[K O] = bwlabel(Conimage);

% Storing the average color of each region
Concolors = zeros(O,3);

% Getting the average color in each labeled region
for p = 1:O % step through patches
    each_pch = K==p;
    all_pch_areas = W(each_pch(:,:,[1 1 1]));
    Concolors(p,:) = mean(reshape(all_pch_areas,[],3),1);
end

% Normalizing the color values to the required range [0, 1]
Concolors = Concolors./255;

% Snapping centers to grid
Y = regionprops(Conimage,'centroid');
X = vertcat(Y.Centroid);
lim_X = [min(X,[],1); max(X,[],1)];
X = round((X-lim_X(1,:))./range(lim_X,1)*3 + 1);

% Reordering  the color samples
idx = sub2ind([4 4],X(:,2),X(:,1));
Concolors(idx,:) = Concolors;

% Specifing color names
clrnames = {'white','red','green','blue','yellow'};

% declaring a reference colors list in RGB
clrrefs = [1 1 1; 1 0 0; 0 1 0; 0 0 1; 1 1 0];

% measuring distance of colours in RGB
I = Concolors - permute(clrrefs,[3 2 1]);
I = squeeze(sum(I.^2,2));

% finding the nearest match
[~,idx] = min(I,[],2);

% Looking for the colour names in each patch
Colornames = reshape(clrnames(idx),4,4);

% Returns the array color names
colours= Colornames;

end