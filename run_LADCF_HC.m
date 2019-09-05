function [results,sz] = run_LADCF_HC(seq, res_path, bSaveImage)

setup_paths();

% Feature specific parameters
hog_params.cell_size = 4;
hog_params.nDim = 31;

grayscale_params.colorspace='gray';
grayscale_params.cell_size = 4;

cn_params.tablename = 'CNnorm';
cn_params.useForGray = false;
cn_params.cell_size = 4;
cn_params.nDim = 10;

ic_params.tablename = 'intensityChannelNorm6';
ic_params.useForColor = false;
ic_params.cell_size = 4;

params.t_features = {
    struct('getFeature',@get_fhog,'fparams',hog_params),...
    struct('getFeature',@get_table_feature, 'fparams',cn_params),...
    struct('getFeature',@get_table_feature, 'fparams',ic_params),...
    struct('getFeature',@get_colorspace, 'fparams',grayscale_params),...  
};
params.feature_downsample_ratio = 4;    % Feature downsample ratio
params.t_global.cell_size = 4;          % Feature cell size

% Image sample parameters
params.search_area_shape = 'square';    % The shape of the samples
params.search_area_scale = 4.9;         % The search area
params.min_image_sample_size = 150^2;   % Minimum area of image samples
params.max_image_sample_size = 200^2;   % Maximum area of image samples

% Detection parameters
params.refinement_iterations = 1;       % Number of iterations used to refine the resulting position in a frame
params.newton_iterations = 5;           % The number of Newton iterations used for optimizing the detection score
params.clamp_position = false;          % Clamp the target position to be inside the image

% Learning parameters
params.rl = 0.95;
params.output_sigma_factor = 1/16;		% Label function sigma
params.lambda2 = 15;                    % The temporal consisitency parameter
params.fs_rate = 1.25;                  % Selected spatial feature threshold

% ADMM parameters
params.max_iterations = 2;
params.init_penalty_factor = 1;
params.penalty_scale_step = 5;

% Scale parameters for the translation model
params.number_of_scales = 5;            % Number of scales 
params.scale_step = 1.01;               % The scale ratio

% Visualisation
params.visualization = 1;               % Visualiza tracking and detection scores

% GPU
params.use_gpu = false;                 % Enable GPU or not (only cpu is supported in this version)
params.gpu_id = [];                     % Set the GPU id, or leave empty to use default

% Initialise
params.seq = seq;

% Run tracker
[results,sz] = tracker(params);
