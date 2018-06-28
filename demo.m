%%  demo for Learning Adaptive Discriminative Correlation Filter via 
%   Temporal Consistency Preserved Feature Selection Embedded Visual Tracking
close all;

% Add paths
setup_paths();

%  Load video 
base_path  = './sequences';
video = choose_video(base_path);
video_path = [base_path '/' video];
[seq, gt_boxes] = load_video(video_path,video);

% Run Hand-crafted feature based LADCF on cpu
results = run_LADCF_HC(seq);

%results
pd_boxes = results.res;
thresholdSetOverlap = 0: 0.05 : 1;
success_num_overlap = zeros(1, numel(thresholdSetOverlap));
res = calcRectInt(gt_boxes, pd_boxes);
for t = 1: length(thresholdSetOverlap)
    success_num_overlap(1, t) = sum(res > thresholdSetOverlap(t));
end
cur_AUC = mean(success_num_overlap) / size(gt_boxes, 1);
FPS_vid = results.fps;
display([video  '---->' '   FPS:   ' num2str(FPS_vid)   '    op:   '   num2str(cur_AUC)]);