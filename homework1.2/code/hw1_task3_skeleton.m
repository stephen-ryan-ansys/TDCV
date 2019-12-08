clear
clc
close all
addpath('helper_functions')

%% Setup
% path to the images folder
path_img_dir = '../data/tracking/valid/img';
% path to object ply file
object_path = '../data/teabox.ply';
% path to results folder
results_path = '../data/tracking/valid/results';

% Read the object's geometry 
% Here vertices correspond to object's corners and faces are triangles
[vertices, faces] = read_ply(object_path);

% Create directory for results
if ~exist(results_path,'dir') 
    mkdir(results_path); 
end

% Load Ground Truth camera poses for the validation sequence
% Camera orientations and locations in the world coordinate system
load('gt_valid.mat')

% setup camera parameters (camera_params) using cameraParameters()
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

camera_params = cameraParameters('IntrinsicMatrix', intrinsic_matrix);


%% Get all filenames in images folder

FolderInfo = dir(fullfile(path_img_dir, '*.JPG'));
Filenames = fullfile(path_img_dir, {FolderInfo.name} );
num_files = length(Filenames);

% Place predicted camera orientations and locations in the world coordinate system for all images here
cam_in_world_orientations = zeros(3,3,num_files);
cam_in_world_locations = zeros(1,3,num_files);

%% Detect SIFT keypoints in all images

% You will need vl_sift() and vl_ubcmatch() functions
% download vlfeat (http://www.vlfeat.org/download.html) and unzip it somewhere
% Don't forget to add vlfeat folder to MATLAB path

% Place SIFT keypoints and corresponding descriptors for all images here
keypoints = cell(num_files,1); 
descriptors = cell(num_files,1); 

% for i=1:length(Filenames)
%     fprintf('Calculating sift features for image: %d \n', i)
%
% %   Prepare the image (img) for vl_sift() function
%     img = single(rgb2gray(imread(Filenames{i})));
%     [keypoints{i}, descriptors{i}] = vl_sift(img) ;
% end

% Save sift features and descriptors and load them when you rerun the code to save time
% save('sift_descriptors.mat', 'descriptors')
% save('sift_keypoints.mat', 'keypoints')

load('sift_descriptors.mat');
load('sift_keypoints.mat');

%% Initialization: Compute camera pose for the first image

% As the initialization step for tracking
% you need to compute the camera pose for the first image 
% The first image and it's camera pose will be your initial frame 
% and initial camera pose for the tracking process

% You can use estimateWorldCameraPose() function or your own implementation
% of the PnP+RANSAC from previous tasks

% You can get correspondences for PnP+RANSAC either using your SIFT model from the previous tasks
% or by manually annotating corners (e.g. with mark_images() function)

% imshow('../../homework1.1/code/vertices.png')
% title('Vertices numbering')
%
% num_points = 8;
% image_points = mark_image(Filenames{1}, num_points);
%
% save('image_points.mat', 'image_points');
load('image_points.mat');

[image_points, removed] = rmmissing(image_points(:, :));
world_points = vertices(~removed, :);

% Estimate camera position for the first image
[init_orientation, init_location] = estimateWorldCameraPose(image_points, world_points, camera_params, 'MaxReprojectionError', 2);

cam_in_world_orientations(:,:, 1) = init_orientation;
cam_in_world_locations(:,:, 1) = init_location;

% Visualise the pose for the initial frame
edges = [[1, 1, 1, 2, 2, 3, 3, 4, 5, 5, 6, 7]
    [2, 4, 5, 3, 6, 4, 7, 8, 6, 8, 7, 8]];
figure()
hold on;
imshow(char(Filenames(1)), 'InitialMagnification', 'fit');
title(sprintf('Initial Image Camera Pose'));
%   Plot bounding box
points = project3d2image(vertices',camera_params, cam_in_world_orientations(:,:,1), cam_in_world_locations(:, :, 1));
for j=1:12
    plot(points(1, edges(:, j)), points(2, edges(:,j)), 'color', 'b');
end
hold off;


%% IRLS nonlinear optimisation

% Now you need to implement the method of iteratively reweighted least squares (IRLS)
% to optimise reprojection error between consecutive image frames

% Method steps:

% 1) Back-project SIFT keypoints from the initial frame (image i) to the object using the
% initial camera pose and the 3D ray intersection code from the task 1. 
% This will give you 3D coordinates (in the world coordinate system) of the
% SIFT keypoints from the initial frame (image i) that correspond to the object

% 2) Find matches between descriptors of back-projected SIFT keypoints from the initial frame (image i) and the
% SIFT keypoints from the subsequent frame (image i+1) using vl_ubcmatch() from VLFeat library

% 3) Project back-projected SIFT keypoints onto the subsequent frame (image i+1) using 3D coordinates from the
% step 1 and the initial camera pose 

% 4) Compute the reprojection error between 2D points of SIFT
% matches for the subsequent frame (image i+1) and 2D points of projected matches
% from step 3

% 5) Implement IRLS: for each IRLS iteration compute Jacobian of the reprojection error with respect to the pose
% parameters and update the camera pose for the subsequent frame (image i+1)

% 6) Now the subsequent frame (image i+1) becomes the initial frame for the
% next subsequent frame (image i+2) and the method continues until camera poses for all
% images are estimated

% We suggest you to validate the correctness of the Jacobian implementation
% either using Symbolic toolbox or finite differences approach

threshold_ubcmatch = 6; % matching threshold for vl_ubcmatch()

% setup vertices for TriangleRayIntersection
vert1 = vertices(faces(:, 1) + 1, :);
vert2 = vertices(faces(:, 2) + 1, :);
vert3 = vertices(faces(:, 3) + 1, :);

for i = 1:(num_files - 1)
    % 1) Back-project initial frame to get 3D coords:
    size_total_sift_points = size(keypoints{i}, 2);
    model.coord3d = zeros(size_total_sift_points, 3);
    model.descriptors = zeros(128, size_total_sift_points, 'uint8');
    points_found = 0;

    m = ones(3, 1);
    P = camera_params.IntrinsicMatrix.'*[cam_in_world_orientations(:,:,i) -cam_in_world_orientations(:,:,i)*cam_in_world_locations(:,:,i).'];
    Q = P(:,1:3);
    q = P(:,4);
    orig = -inv(Q)*q;

    for j = 1:size_total_sift_points
        m(1:2) = keypoints{i}(1:2, j);
        ray = Q\m;
        [intersect, ~, ~, ~, xcoor] = TriangleRayIntersection(...
            orig',...
            ray',...
            vert1, vert2, vert3,...
            'planeType', 'one sided',... % necessary to ignore occluded faces
            'border', 'inclusive'... % include intersections on borders
        );
        if any(intersect)
            points_found = points_found + 1;
            idx = find(intersect);
            M = xcoor(idx, :);
            model.coord3d(points_found, :) = M(1, :);
            model.descriptors(:, points_found) = descriptors{i}(:, j);
        end
    end
    model.coord3d = model.coord3d(1:points_found, :);
    model.descriptors = model.descriptors(:, 1:points_found);

    % 2) Find SIFT matches
    img = single(rgb2gray(imread(Filenames{i+1})));
    sift_matches = vl_ubcmatch(descriptors{i+1}, model.descriptors, threshold_ubcmatch);
    image_matches_idx = sift_matches(1, :);
    model_matches_idx = sift_matches(2, :);

    % Implement IRLS method for the reprojection error optimisation
    % You can start with these parameters to debug your solution 
    % but you should also experiment with their different values
    threshold_irls = 0.000001; % update threshold for IRLS
    N = 200; % number of iterations

    fprintf('Performing IRLS for image: %d\n', i+1);
    % 3D reproj from camera i:
    M = model.coord3d(model_matches_idx, 1:3);
    % image points from camera i+1:
    m = keypoints{i+1}(1:2, image_matches_idx)';
    [R, t] = cameraPoseToExtrinsics(cam_in_world_orientations(:, :, i), cam_in_world_locations(:, :, i));
    % returned so that [x, y, z] = [X, Y, Z]*R + [t_X, t_Y, t_Z];
    % so need to use R' later when multiplying from left
    v = rotationMatrixToVector(R); % exponential coords as row vector
    theta = [v'; t']; % column vector
    n = 0; % t in paper
    lambda = 0.001;
    u = threshold_irls + 1;
    % calculate Jacobian as in the slides:
    J = energy_jacobian(camera_params, R, t, M);

    while (n < N && u > threshold_irls)
        % calculate energy
        [E, W, e] = energy_function(camera_params, R, t, M, m);

        delta = -inv(J'*W*J + lambda*eye(6))*(J'*W*e);
        updated_theta = theta + delta;
        updated_R = rotationVectorToMatrix(updated_theta(1:3));
        updated_t = updated_theta(4:6)';
        [updated_E, ~, ~] = energy_function(camera_params, updated_R, updated_t, M, m);

        if updated_E > E
            lambda = 10*lambda;
        else
            lambda = 0.1*lambda;
            theta = updated_theta;
            R = updated_R;
            t = updated_t;
            J = energy_jacobian(camera_params, R, t, M);
        end

        u = norm(delta);
        n = n + 1;
    end

    [cam_in_world_orientations(:, :, i+1), cam_in_world_locations(:, :, i+1)] = ...
        extrinsicsToCameraPose(R, t);
end


%% Plot camera trajectory in 3D world CS + cameras

figure()
% Predicted trajectory
visualise_trajectory(vertices, edges, cam_in_world_orientations, cam_in_world_locations, 'Color', 'b');
hold on;
% Ground Truth trajectory
visualise_trajectory(vertices, edges, gt_valid.orientations, gt_valid.locations, 'Color', 'g');
hold off;
title('\color{green}Ground Truth trajectory \color{blue}Predicted trajectory')

%% Visualize bounding boxes

figure()
for i=1:num_files
    
    imshow(char(Filenames(i)), 'InitialMagnification', 'fit');
    title(sprintf('Image: %d', i))
    hold on
    % Ground Truth Bounding Boxes
    points_gt = project3d2image(vertices',camera_params, gt_valid.orientations(:,:,i), gt_valid.locations(:, :, i));
    % Predicted Bounding Boxes
    points_pred = project3d2image(vertices',camera_params, cam_in_world_orientations(:,:,i), cam_in_world_locations(:, :, i));
    for j=1:12
        plot(points_gt(1, edges(:, j)), points_gt(2, edges(:,j)), 'color', 'g');
        plot(points_pred(1, edges(:, j)), points_pred(2, edges(:,j)), 'color', 'b');
    end
    hold off;
    
    filename = fullfile(results_path, strcat('image', num2str(i), '.png'));
    saveas(gcf, filename)
end

%% Bonus part

% Save estimated camera poses for the validation sequence using Vision TUM trajectory file
% format: https://vision.in.tum.de/data/datasets/rgbd-dataset/file_formats
% Then estimate Absolute Trajectory Error (ATE) and Relative Pose Error for
% the validation sequence using python tools from: https://vision.in.tum.de/data/datasets/rgbd-dataset/tools
% In this task you should implement you own function to convert rotation matrix to quaternion

% Save estimated camera poses for the test sequence using Vision TUM 
% trajectory file format

% Attach the file with estimated camera poses for the test sequence to your code submission
% If your code and results are good you will get a bonus for this exercise
% We are expecting the mean absolute translational error (from ATE) to be
% approximately less than 1cm

% TODO: Estimate ATE and RPE for validation and test sequences
frames = 6:30;
save_trajectory('pred_valid.txt', frames, cam_in_world_orientations, cam_in_world_locations);
% then run:
%../../../rgbd_benchmark_tools/scripts/evaluate_ate.py --plot ate_trajectory_plot.png --verbose gt_valid.txt pred_valid.txt
%../../../rgbd_benchmark_tools/scripts/evaluate_rpe.py --fixed_delta --delta_unit f --plot rte_trajectory_plot.png --verbose gt_valid.txt pred_valid.txt