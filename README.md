# LADCF - No 1 Algorithm on the public dataset of [VOT2018](http://www.votchallenge.net/vot2018/)
Demo for Learning Adaptive Discriminative Correlation Filters (LADCF) via Temporal Consistency preserving Spatial Feature Selection for Robust Visual Tracking

## [Download the Paper](https://www.researchgate.net/publication/326696472_Learning_Adaptive_Discriminative_Correlation_Filters_via_Temporal_Consistency_Preserving_Spatial_Feature_Selection_for_Robust_Visual_Tracking)
>@article{xu2018learning,
  title={Learning Adaptive Discriminative Correlation Filters via Temporal Consistency Preserving Spatial Feature Selection for Robust Visual Tracking},
  author={Xu, Tianyang and Feng, Zhen-Hua and Wu, Xiao-Jun and Kittler, Josef},
  journal={arXiv preprint arXiv:1807.11348},
  year={2018}}

### Instruction for LADCF_HC Tracker:
Learning Adaptive Discriminative Correlation Filter on Low-dimensional Manifold (LADCF) utilises adaptive spatial regularizer to train low-dimensional discriminative correlation filters. We follow a single-frame learning and updating strategy: the filters are learned after tracking stage and then updated using a fixed rate [1]. We use HOG [2] and CN [3]. Code modules refer to ECO [4] in feature extraction.

#### Dependencies:
- [PDollar Toolbox](https://pdollar.github.io/toolbox)
- mtimesx (https://github.com/martin-danelljan/ECO/tree/master/external_libs/mexResize)
- mexResize (https://github.com/martin-danelljan/ECO/tree/master/external_libs/mtimesx) 

#### Operating system:
Ubuntu 14.04 LTS, Matlab R2016a, CPU Intel(R) Xeon(R) E5-2643 

#### References:
- [1] Henriques, João F., et al. "High-speed tracking with kernelized correlation filters." IEEE Transactions on Pattern Analysis and Machine Intelligence 37.3 (2015): 583-596.
- [2] Dalal, Navneet, and Bill Triggs. "Histograms of oriented gradients for human detection." Computer Vision and Pattern Recognition, 2005. CVPR 2005. 
- [3] Van De Weijer, Joost, et al. "Learning color names for real-world applications." IEEE Transactions on Image Processing 18.7 (2009): 1512-1523.
- [4] Danelljan, Martin, et al. "Eco: Efficient convolution operators for tracking." Proceedings of the 2017 IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2017.


