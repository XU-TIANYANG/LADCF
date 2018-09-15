Demo for Learning Adaptive Discriminative Correlation Filters (LADCF) via Temporal Consistency preserving Spatial Feature Selection for Robust Visual Tracking (https://arxiv.org/abs/1807.11348)

Xu, Tianyang, Zhen-Hua Feng, Xiao-Jun Wu, and Josef Kittler. "Learning Adaptive Discriminative Correlation Filters via Temporal Consistency Preserving Spatial Feature Selection for Robust Visual Tracking." arXiv preprint arXiv:1807.11348 (2018).

Instruction for LADCF_HC Tracker:
Learning Adaptive Discriminative Correlation Filter on Low-dimensional Manifold (LADCF) utilises adaptive spatial regularizer to train low-dimensional discriminative correlation filters. We follow a single-frame learning and updating strategy: the filters are learned after tracking stage and then updated using a fixed rate [1]. We use HOG [2] and CN [3]. Code modules refer to ECO [4] in feature extraction.

Dependencies:
PDollar Toolbox [5], mtimesx and mexResize. 

Operating system:
Ubuntu 14.04 LTS, Matlab R2016a, CPU Intel(R) Xeon(R) E5-2643 

References:

[1] Henriques, João F., et al. "High-speed tracking with kernelized correlation filters." IEEE Transactions on Pattern Analysis and Machine Intelligence 37.3 (2015): 583-596.

[2] Dalal, Navneet, and Bill Triggs. "Histograms of oriented gradients for human detection." Computer Vision and Pattern Recognition, 2005. CVPR 2005. 

[3] Van De Weijer, Joost, et al. "Learning color names for real-world applications." IEEE Transactions on Image Processing 18.7 (2009): 1512-1523.

[4] Danelljan, Martin, et al. "Eco: Efficient convolution operators for tracking." Proceedings of the 2017 IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2017.

[5] PDollar Toolbox: https://pdollar.github.io/toolbox/  


