clear
clc
close all
addpath('helper_functions')

%% Setup
% path to the images folder
path_img_dir = '../data/detection';
% path to object ply file
object_path = '../data/teabox.ply';

% Read the object's geometry
% Here vertices correspond to object's corners and faces are triangles
[vertices, faces] = read_ply(object_path);

% Load the SIFT model from the previous task
load('sift_model.mat');


% TODO: setup camera intrinsic parameters using cameraParameters()
% focal lengths (square pixels since fx = fy):
fx = 2960.37845;
fy = fx;
% optical center:
cx = 1841.68855;
cy = 1235.23369;
% axis skew:
s = 0;
intrinsic_matrix = [
    fx,  0,  0;
     s, fy,  0;
    cx, cy,  1;
];
camera_params = cameraParameters(...
    'IntrinsicMatrix', intrinsic_matrix,...
    'ImageSize', [3680, 2456]...
);

%% Get all filenames in images folder

FolderInfo = dir(fullfile(path_img_dir, '*.JPG'));
Filenames = fullfile(path_img_dir, {FolderInfo.name} );
num_files = length(Filenames);


%% Match SIFT features of new images to the SIFT model with features computed in the task 1
% You should use VLFeat function vl_ubcmatch()

% Place SIFT keypoints and descriptors of new images here
keypoints=cell(num_files,1);
descriptors=cell(num_files,1);
% Place matches between new SIFT features and SIFT features from the SIFT
% model here
sift_matches=cell(num_files,1);

% Default threshold for SIFT keypoints matching: 1.5
% When taking higher value, match is only recognized if similarity is very high
% threshold_ubcmatch = 1.5;
%
% for i=1:num_files
%     fprintf('Calculating and matching sift features for image: %d \n', i)
%
% %     TODO: Prepare the image (img) for vl_sift() function
%     img = single(rgb2gray(imread(Filenames{i})));
%     [keypoints{i}, descriptors{i}] = vl_sift(img);
% %     Match features between SIFT model and SIFT features from new image
%     sift_matches{i} = vl_ubcmatch(descriptors{i}, model.descriptors, threshold_ubcmatch);
% end


% Save sift features, descriptors and matches and load them when you rerun the code to save time
% save('sift_matches.mat', 'sift_matches');
% save('detection_keypoints.mat', 'keypoints')
% save('detection_descriptors.mat', 'descriptors')

load('sift_matches.mat')
load('detection_keypoints.mat')
load('detection_descriptors.mat')


%% PnP and RANSAC
% Implement the RANSAC algorithm featuring also the following arguments:
% Reprojection error threshold for inlier selection - 'threshold_ransac'
% Number of RANSAC iterations - 'ransac_iterations'

% Pseudocode
% i Randomly select a sample of 4 data points from S and estimate the pose using PnP.
% ii Determine the set of data points Si from all 2D-3D correspondences
%   where the reprojection error (Euclidean distance) is below the threshold (threshold_ransac).
%   The set Si is the consensus set of the sample and defines the inliers of S.
% iii If the number of inliers is greater than we have seen so far,
%   re-estimate the pose using Si and store it with the corresponding number of inliers.
% iv Repeat the above mentioned procedure for N iterations (ransac_iterations).

% For PnP you can use estimateWorldCameraPose() function
% but only use it with 4 points and set the 'MaxReprojectionError' to the
% value of 10000 so that all these 4 points are considered to be inliers

% Place camera orientations, locations and best inliers set for every image here
cam_in_world_orientations = zeros(3,3,num_files);
cam_in_world_locations = zeros(1,3,num_files);
best_inliers_set = cell(num_files, 1);

ransac_iterations = 50;
threshold_ransac = 4;

for i = 1:num_files
    fprintf('Running PnP+RANSAC for image: %d \n', i)

%     TODO: Implement the RANSAC algorithm here

    best_inliers_size = 3; % inliers set must be at least 4 to re-estimate
    for j = 1:ransac_iterations
        % select 4 random sift correspondences
        idx = randperm(size(sift_matches{i}, 2), 4);
        idx = sift_matches{i}(:, idx);
        image_idx = idx(1, :);
        model_idx = idx(2, :);
        image_sample = keypoints{i}(1:2, image_idx)';
        world_sample = model.coord3d(model_idx, :);

        try
            % estimate pose
            [world_orientation, world_location] = estimateWorldCameraPose(...
                image_sample,...
                world_sample,...
                camera_params,...
                'MaxReprojectionError', 10000 ...
            );
        catch
            % sometimes it still fails with 10000 MaxReprojectionError
            continue
        end

        [R, t] = cameraPoseToExtrinsics(world_orientation, world_location);
        P = cameraMatrix(camera_params, R, t);

        image_matches = keypoints{i}(1:2, sift_matches{i}(1, :));
        world_matches = model.coord3d(sift_matches{i}(2, :), :);
        world_matches(:, 4) = 1;

        % project from world (U, V, W) to image plane (x, y, z)
        film_coords = world_matches*P;

        % get in pixels (u, v)
        reproj = film_coords(:, 1:2)./film_coords(:, 3);

        % the L2 norm of each match (Euclidian distance)
        diff = image_matches - reproj';
        L2_norms = vecnorm(diff, 2, 1);

        % is the number of inliers better than before?
        inliers_set = find(L2_norms < threshold_ransac);
        if (numel(inliers_set) > best_inliers_size)
            best_inliers_set{i} = inliers_set;
            best_inliers_size = numel(best_inliers_set{i});
        end
    end

    % re-estimate pose using the best inliers
    [cam_in_world_orientations(:,:,i), cam_in_world_locations(:,:,i)] = estimateWorldCameraPose(...
        image_matches(:, best_inliers_set{i})',...
        world_matches(best_inliers_set{i}, 1:3),...
        camera_params,...
        'MaxReprojectionError', 4 ... % TODO what should this be?
    );
end



%% Visualize inliers and the bounding box

% You can use the visualizations below or create your own one
% But be sure to present the bounding boxes drawn on the image to verify
% the camera pose

edges = [[1, 1, 1, 2, 2, 3, 3, 4, 5, 5, 6, 7]
    [2, 4, 5, 3, 6, 4, 7, 8, 6, 8, 7, 8]];

for i=1:num_files

    fh = figure();
    imshow(char(Filenames(i)), 'InitialMagnification', 'fit');
    title(sprintf('Image: %d', i))
    fh.WindowState = 'maximized';
    hold on

%   Plot inliers set
    PlotInlierOutlier(best_inliers_set{i}, camera_params, sift_matches{i}, model.coord3d, keypoints{i}, cam_in_world_orientations(:,:,i), cam_in_world_locations(:,:,i))
%   Plot bounding box
    points = project3d2image(vertices',camera_params, cam_in_world_orientations(:,:,i), cam_in_world_locations(:, :, i));
    for j=1:12
        plot(points(1, edges(:, j)), points(2, edges(:,j)), 'color', 'b');
    end
    hold off;
end