# SOD CNNs-based Read List       

In this repository, we mainly focus on deep learning based saliency methods (**2D RGB, 3D RGB-D, Video SOD and 4D Light Field**) and provide a summary (**Code and Paper**). We hope this repo can help you to better understand saliency detection in the deep learning era.        

--------------------------------------------------------------------------------------
 :heavy_exclamation_mark:  **2D SOD**: Add three ICCV21 papers                 
 :heavy_exclamation_mark:  **3D SOD**: Add two CVPR21 paper and two ACMM21 papers and one ICCV21 paper 
 :heavy_exclamation_mark:  **LF SOD**: Add one IEEE TCyB21 paper   
 :heavy_exclamation_mark:  **Video SOD** : Add four CVPR21, ACMM21 and ICCV21 papers. 

:running: **We will keep updating it.** :running:    
--------------------------------------------------------------------------------------


------
 

## Content:

1. <a href="#Overall"> An overview of the Paper List </a>
2. <a href="#2D RGB Saliency Detection"> 2D RGB Saliency Detection </a>
3. <a href="#3D RGB-D Saliency Detection">  3D RGB-D/T Saliency Detection </a>
4. <a href="#4D Light Field Saliency Detection"> 4D Light Field Saliency Detection </a>
5. <a href="#Video Salient Object Detection"> Video Saliency Detection </a>
6. <a href="#Earlier Methods"> Survery and earlier Methods </a>
7. <a href="#The SOD dataset download"> The SOD dataset download </a>
8. <a href="#Evaluation Metrics"> Evaluation Metrics </a>


------

   
# Overall <a id="Overall" class="anchor" href="Overall" aria-hidden="true"><span class="octicon octicon-link"></span></a>
![avatar](https://github.com/jiwei0921/SOD-CNNs-based-code-summary-/blob/master/SOD-2019.7.23.jpg)
    
# 2D RGB Saliency Detection <a id="2D RGB Saliency Detection" class="anchor" href="2D RGB Saliency Detection" aria-hidden="true"><span class="octicon octicon-link"></span></a>    

## 2021       
**No.** | **Pub.** | **Title** | **Links** 
:-: | :-: | :-  | :-: 
01 | **AAAI** | Structure-Consistent Weakly Supervised Salient Object Detection with Local Saliency Coherence | [Paper](https://arxiv.org/pdf/2012.04404.pdf)/[Code](https://github.com/siyueyu/SCWSSOD/tree/f8650567cbbc8df5bf6edc32a633c47a885574cd)
02 | **AAAI** | Pyramidal Feature Shrinking for Salient Object Detection | [Paper](https://www.aaai.org/AAAI21Papers/AAAI-1322.MaM.pdf)/Code  
03 | **AAAI** | Locate Globally, Segment Locally: A Progressive Architecture with Knowledge Review Network for Salient Object Detection | [Paper](https://www.aaai.org/AAAI21Papers/AAAI-4841.XuB.pdf)/[Code](https://github.com/bradleybin/Locate-Globally-Segment-locally-A-Progressive-Architecture-With-Knowledge-Review-Network-for-SOD)  
04 | **AAAI** | Multi-Scale Graph Fusion for Co-Saliency Detection | [Paper](https://ojs.aaai.org/index.php/AAAI/article/view/16951)/Code
05 | **AAAI** | Generating Diversified Comments via Reader-Aware Topic Modeling and Saliency Detection | [Paper](https://arxiv.org/pdf/2102.06856.pdf)/Code
06 | **ICIP** | Multiscale IoU: A Metric for Evaluation of Salient Object Detection with Fine Structures | [Paper](https://arxiv.org/pdf/2105.14572.pdf)/Code
07 | **TCSVT** | Weakly-Supervised Saliency Detection via Salient Object Subitizing | [Paper](https://arxiv.org/pdf/2101.00932.pdf)/Code 
08 | **TIP** | SAMNet: Stereoscopically Attentive Multi-scale Network for Lightweight Salient Object Detection | [Paper](https://ieeexplore.ieee.org/document/9381668)/[Code](https://github.com/yun-liu/FastSaliency) 
09 | **IJCAI** | C2FNet: Context-aware Cross-level Fusion Network for Camouflaged Object Detection | [Paper](https://arxiv.org/pdf/2105.12555.pdf)/[Code](https://github.com/thograce/C2FNet)
10 | **CVPR** | Railroad is not a Train: Saliency as Pseudo-pixel Supervision for Weakly Supervised Semantic Segmentation | [Paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Lee_Railroad_Is_Not_a_Train_Saliency_As_Pseudo-Pixel_Supervision_for_CVPR_2021_paper.pdf)/[Code](https://github.com/halbielee/EPS)
11 | **CVPR** | Prototype-Guided Saliency Feature Learning for Person Search | [Paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Kim_Prototype-Guided_Saliency_Feature_Learning_for_Person_Search_CVPR_2021_paper.pdf)/Code
12 | **CVPR** | Mesh Saliency: An Independent Perceptual Measure or A Derivative of Image Saliency? | [Paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Song_Mesh_Saliency_An_Independent_Perceptual_Measure_or_a_Derivative_of_CVPR_2021_paper.pdf)/[Code](https://github.com/rsong/MIMO-GAN)
13 | **CVPR** | Weakly-Supervised Instance Segmentation via Class-Agnostic Learning With Salient Images | [Paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Wang_Weakly-Supervised_Instance_Segmentation_via_Class-Agnostic_Learning_With_Salient_Images_CVPR_2021_paper.pdf)/[Code](https://github.com/hustvl/BoxCaseg)
14 | **CVPR** | DeepACG: Co-Saliency Detection via Semantic-Aware Contrast Gromov-Wasserstein Distance | [Paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Zhang_DeepACG_Co-Saliency_Detection_via_Semantic-Aware_Contrast_Gromov-Wasserstein_Distance_CVPR_2021_paper.pdf)/Code
15 | **CVPR** | Black-Box Explanation of Object Detectors via Saliency Maps | [Paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Petsiuk_Black-Box_Explanation_of_Object_Detectors_via_Saliency_Maps_CVPR_2021_paper.pdf)/Code
16 | **CVPR** | From Semantic Categories to Fixations: A Novel Weakly-Supervised Visual-Auditory Saliency Detection Approach | [Paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Wang_From_Semantic_Categories_to_Fixations_A_Novel_Weakly-Supervised_Visual-Auditory_Saliency_CVPR_2021_paper.pdf)/[Code](https://github.com/guotaowang/STANet)
17 | **CVPR** | CAMERAS: Enhanced Resolution and Sanity Preserving Class Activation Mapping for Image Saliency | [Paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Jalwana_CAMERAS_Enhanced_Resolution_and_Sanity_Preserving_Class_Activation_Mapping_for_CVPR_2021_paper.pdf)/[Code](https://github.com/VisMIL/CAMERAS)
18 | **CVPR** | Saliency-Guided Image Translation | [Paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Jiang_Saliency-Guided_Image_Translation_CVPR_2021_paper.pdf)/Code
19 | **CVPR** | Group Collaborative Learning for Co-Salient Object Detection | [Paper](https://arxiv.org/pdf/2104.01108.pdf)/[Code](https://github.com/fanq15/GCoNet)
20 | **CVPR** | Uncertainty-aware Joint Salient Object and Camouflaged Object Detection | [Paper](https://arxiv.org/pdf/2104.02628.pdf)/[Code](https://github.com/JingZhang617/Joint_COD_SOD)
21 | **ACMM** | Auto-MSFNet: Search Multi-scale Fusion Network for Salient Object Detection | [Paper](https://github.com/LiuTingWed/Auto-MSFNet)/[Code](https://github.com/LiuTingWed/Auto-MSFNet) 
22 | **IEEE TIP** | Decomposition and Completion Network for Salient Object Detection | [Paper](https://ieeexplore.ieee.org/abstract/document/9479697/figures#figures)/[Code](https://github.com/wuzhe71/DCN) 
:triangular_flag_on_post: 23 | **ICCV** | Visual Saliency Transformer | [Paper](https://arxiv.org/pdf/2104.12099.pdf)/[Code](https://github.com/nnizhang/VST#visual-saliency-transformer-vst) 
:triangular_flag_on_post: 24 | **ICCV** | Disentangled High Quality Salient Object Detection | [Paper](https://arxiv.org/pdf/2108.03551.pdf)/Code 
:triangular_flag_on_post: 25 | **ICCV** | iNAS: Integral NAS for Device-Aware Salient Object Detection | [Paper](https://mftp.mmcheng.net/Papers/21ICCV-iNAS.pdf)/[Code](https://mmcheng.net/inas/) 




## 2020       
**No.** | **Pub.** | **Title** | **Links** 
:-: | :-: | :-  | :-: 
01 | **AAAI** | Progressive Feature Polishing Network for Salient Object Detection | [Paper](https://arxiv.org/pdf/1911.05942.pdf)/[Code](https://github.com/chenquan-cq/PFPN)       
02 | **AAAI** | Global Context-Aware Progressive Aggregation Network for Salient Object Detection | [Paper](https://github.com/JosephChenHub/GCPANet/blob/master/GCPANet.pdf)/[Code](https://github.com/JosephChenHub/GCPANet)     
03 | **AAAI** | F3Net: Fusion, Feedback and Focus for Salient Object Detection | [Paper](https://arxiv.org/pdf/1911.11445.pdf)/[Code](https://github.com/weijun88/F3Net)    
04 | **AAAI** | Multi-spectral Salient Object Detection by Adversarial Domain Adaptation | [Paper](https://cse.sc.edu/~songwang/document/aaai20b.pdf)/Code 
05 | **AAAI** | Multi-Type Self-Attention Guided Degraded Saliency Detection | [Paper](https://cse.sc.edu/~songwang/document/aaai20a.pdf)/Code 
06 | **CVPR** | Weakly-Supervised Salient Object Detection via Scribble Annotations | [Paper](http://openaccess.thecvf.com/content_CVPR_2020/papers/Zhang_Weakly-Supervised_Salient_Object_Detection_via_Scribble_Annotations_CVPR_2020_paper.pdf)/[Code](https://github.com/JingZhang617/Scribble_Saliency)  
07 | **CVPR** | Taking a Deeper Look at the Co-salient Object Detection | [Paper](http://dpfan.net/wp-content/uploads/CoSalBenchmark_CVPR2020.pdf)/[Code](http://dpfan.net/CoSOD3K/)  
08 | **CVPR** | Multi-scale Interactive Network for Salient Object Detection | [Paper](https://drive.google.com/file/d/1gUYu0hO_8Xc5jgpzetuOVFDrqeSOiKZN/view?usp=sharing)/[Code](https://github.com/lartpang/MINet)  
09 | **CVPR** | Interactive Two-Stream Decoder for Accurate and Fast Saliency Detection | [Paper](http://openaccess.thecvf.com/content_CVPR_2020/papers/Zhou_Interactive_Two-Stream_Decoder_for_Accurate_and_Fast_Saliency_Detection_CVPR_2020_paper.pdf)/[Code](https://github.com/moothes/ITSD-pytorch)  
10 | **CVPR** | Label Decoupling Framework for Salient Object Detection | [Paper](http://openaccess.thecvf.com/content_CVPR_2020/papers/Wei_Label_Decoupling_Framework_for_Salient_Object_Detection_CVPR_2020_paper.pdf)/[Code](https://github.com/weijun88/LDF)  
11 | **CVPR** | Adaptive Graph Convolutional Network with Attention Graph Clustering for Co-saliency Detection | [Paper](https://openaccess.thecvf.com/content_CVPR_2020/papers/Zhang_Adaptive_Graph_Convolutional_Network_With_Attention_Graph_Clustering_for_Co-Saliency_CVPR_2020_paper.pdf)/Code
12 | **ECCV** | Highly Efficient Salient Object Detection with 100K Parameters | [Paper](http://mftp.mmcheng.net/Papers/20EccvSal100k.pdf)/[Code](https://github.com/MCG-NKU/Sal100K)
13 | **ECCV** | n-Reference Transfer Learning for Saliency Prediction | [Paper](http://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123530494.pdf)/[Code](https://github.com/luoyan407/n-reference)   
14 | **ECCV** | Gradient-Induced Co-Saliency Detection | [Paper](http://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123570443.pdf)/[Code](http://zhaozhang.net/coca.html)   
13 | **ECCV** | Learning Noise-Aware Encoder-Decoder from Noisy Labels by Alternating Back-Propagation for Saliency Detection | [Paper](http://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123620341.pdf)/[Code](https://github.com/JingZhang617/Noise-aware-ABP-Saliency)  
15 | **ECCV** | Suppress and Balance: A Simple Gated Network for Salient Object Detection | [Paper](https://arxiv.org/pdf/2007.08074.pdf)/[Code](https://github.com/Xiaoqi-Zhao-DLUT/GateNet-RGB-Saliency) 
16 | **IEEE TIP** | Dynamic Feature Integration for Simultaneous Detection of Salient Object, Edge and Skeleton | [Paper](http://mftp.mmcheng.net/Papers/20TIP-DFI.pdf)/[Code](https://github.com/backseason/DFI)
17 | **IEEE TIP** | CAGNet: Content-Aware Guidance for Salient Object Detection | [Paper](https://arxiv.org/abs/1911.13168)/[Code](https://github.com/Mehrdad-Noori/CAGNet)
18 | **IEEE TCYB** | Lightweight Salient Object Detection via Hierarchical Visual Perception Learning | [Paper](https://ieeexplore.ieee.org/document/9285193)/[Code](https://github.com/yun-liu/FastSaliency)
19 | **NeurIPS** | CoADNet: Collaborative Aggregation-and-Distribution Networks for Co-Salient Object Detection | [Paper](https://arxiv.org/pdf/2011.04887.pdf)/[Code](https://github.com/rmcong/CoADNet_NeurIPS20)
20 | **NeurIPS** | Few-Cost Salient Object Detection with Adversarial-Paced Learning | [Paper](https://papers.nips.cc/paper/2020/file/8fc687aa152e8199fe9e73304d407bca-Paper.pdf)/[Code](https://papers.nips.cc/paper/2020/file/8fc687aa152e8199fe9e73304d407bca-Supplemental.zip)
21 | **NeurIPS** | ICNet: Intra-saliency Correlation Network for Co-Saliency Detection | [Paper](https://proceedings.neurips.cc/paper/2020/file/d961e9f236177d65d21100592edb0769-Paper.pdf)/[Code](https://github.com/blanclist/ICNet)


  





## 2019       
**No.** | **Pub.** | **Title** | **Links** 
:-: | :-: | :-  | :-: 
01 | **CVPR** | AFNet: Attentive Feedback Network for Boundary-aware Salient Object Detection | [Paper](https://pan.baidu.com/s/1n-dRVC4sLWCmhhD5bnVXqg)/[Code](https://github.com/ArcherFMY/AFNet)  
02 | **CVPR** | BASNet: Boundary Aware Salient Object Detection | [Paper](http://openaccess.thecvf.com/content_CVPR_2019/html/Qin_BASNet_Boundary-Aware_Salient_Object_Detection_CVPR_2019_paper.html)/[Code](https://github.com/NathanUA/BASNet) 
03 | **CVPR** | CPD: Cascaded Partial Decoder for Accurate and Fast Salient Object Detection | [Paper](https://arxiv.org/pdf/1904.08739.pdf)/[Code](https://github.com/wuzhe71/CPD-CVPR2019)
04 | **CVPR** | Multi-source weak supervision for saliency detection | [Paper](https://arxiv.org/pdf/1904.00566.pdf)/[Code](https://github.com/zengxianyu/mws)
05 | **CVPR** | MLMSNet:A Mutual Learning Method for Salient Object Detection with intertwined Multi-Supervision | [Paper](https://pan.baidu.com/s/1EUxabfnEi_l5-ghUI3_qVQ)/[Code](https://github.com/JosephineRabbit/MLMSNet)
06 | **CVPR** | CapSal: Leveraging Captioning to Boost Semantics for Salient Object Detection | [Paper](https://openaccess.thecvf.com/content_CVPR_2019/papers/Zhang_CapSal_Leveraging_Captioning_to_Boost_Semantics_for_Salient_Object_Detection_CVPR_2019_paper.pdf)/[Code](https://github.com/zhangludl/code-and-dataset-for-CapSal)
07 | **CVPR** | PoolNet: A Simple Pooling-Based Design for Real-Time Salient Object Detection | [Paper](https://arxiv.org/pdf/1904.09569.pdf)/[Code](https://github.com/backseason/PoolNet) 
08 | **CVPR** | An Iterative and Cooperative Top-down and Bottom-up Inference Network for Salient Object Detection | [Paper](http://mftp.mmcheng.net/Papers/19cvprIterativeSOD.pdf)/Code
09 | **CVPR** | Pyramid Feature Attention Network for Saliency detection | [Paper](https://arxiv.org/pdf/1903.00179.pdf)/[Code](https://github.com/CaitinZhao/cvpr2019_Pyramid-Feature-Attention-Network-for-Saliency-detection)
10 | **AAAI** | Deep Embedding Features for Salient Object Detection | [Paper](https://pan.baidu.com/s/1HfyavmYB2NYUMe8CSe2qCw)/Code
11 | **ICIP** | Salient Object Detection Via Deep Hierarchical Context Aggregation And Multi-Layer Supervision | [Paper](https://github.com/ZhangC2/Saliency-DHCA-ML_S)/[Code](https://github.com/ZhangC2/Saliency-DHCA-ML_S)
12 | **IEEE TCSVT** | AADF-Net: Aggregating Attentional Dilated Features for Salient Object | [Paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8836095)/[Code](https://github.com/githubBingoChen/AADF-Net)
13 | **IEEE TCyb** | ROSA: Robust Salient Object Detection against Adversarial Attacks | [Paper](https://arxiv.org/pdf/1905.03434.pdf)/[Code](https://github.com/lhaof/ROSA-Robust-Salient-Object-Detection-Against-Adversarial-Attacks)
14 | **arXiv** | DSAL-GAN: DENOISING BASED SALIENCY PREDICTION WITH GENERATIVE ADVERSARIAL NETWORKS | [Paper](https://arxiv.org/pdf/1904.01215.pdf)/Code
15 | **arXiv** | SAC-Net: Spatial Attenuation Context for Salient Object Detection | [Paper](https://arxiv.org/pdf/1903.10152.pdf)/Code
16 | **arXiv** | SE2Net: Siamese Edge-Enhancement Network for Salient Object Detection | [Paper](https://arxiv.org/pdf/1904.00048.pdf)/Code
17 | **arXiv** | Region Refinement Network for Salient Object Detection | [Paper](https://arxiv.org/pdf/1906.11443.pdf)/Code
18 | **arXiv** | Contour Loss: Boundary-Aware Learning for Salient Object Segmentation | [Paper](https://arxiv.org/pdf/1908.01975.pdf)/Code
19 | **arXiv** | OGNet: Salient Object Detection with Output-guided Attention Module | [Paper](https://arxiv.org/pdf/1907.07449.pdf)/Code
20 | **arXiv** | Edge-guided Non-local Fully Convolutional Network for Salient Object Detection | [Paper](https://arxiv.org/pdf/1908.02460.pdf)/Code
21 | **ICCV** | FLoss:Optimizing the F-measure for Threshold-free Salient Object Detection | [Paper](http://openaccess.thecvf.com/content_ICCV_2019/papers/Zhao_Optimizing_the_F-Measure_for_Threshold-Free_Salient_Object_Detection_ICCV_2019_paper.pdf)/[Code](https://github.com/zeakey/iccv2019-fmeasure)
22  | **ICCV** | Stacked Cross Refinement Network for Salient Object Detection | [Paper](http://openaccess.thecvf.com/content_ICCV_2019/papers/Wu_Stacked_Cross_Refinement_Network_for_Edge-Aware_Salient_Object_Detection_ICCV_2019_paper.pdf)/[Code](https://github.com/wuzhe71/SCRN)
23 | **ICCV** | Selectivity or Invariance: Boundary-aware Salient Object Detection | [Paper](https://arxiv.org/pdf/1812.10066.pdf)/Code
24 | **ICCV** | HRSOD:Towards High-Resolution Salient Object Detection | [Paper](https://arxiv.org/pdf/1908.07274.pdf)/[Code](https://github.com/yi94code/HRSOD)
25 | **ICCV** | EGNet:Edge Guidance Network for Salient Object Detection | [Paper](http://mftp.mmcheng.net/Papers/19ICCV_EGNetSOD.pdf)/[Code](https://github.com/JXingZhao/EGNet)
26 | **ICCV** | Structured Modeling of Joint Deep Feature and Prediction Refinement for Salient Object Detection | [Paper](https://arxiv.org/pdf/1909.04366.pdf)/[Code](https://github.com/xuyingyue/DeepUnifiedCRF_iccv19)
27 | **ICCV** | Employing Deep Part-Object Relationships for Salient Object Detection | [Paper](http://openaccess.thecvf.com/content_ICCV_2019/papers/Liu_Employing_Deep_Part-Object_Relationships_for_Salient_Object_Detection_ICCV_2019_paper.pdf)/Code
28 | **NeurIPS** | Deep Robust Unsupervised Saliency Prediction With Self-Supervision | [Paper](https://arxiv.org/pdf/1909.13055.pdf)/[Code](https://drive.google.com/file/d/10GlmenXR7nEJyRlmPHouvHP-g9KfUW1F/view)   
29 | **CVPR** | Salient Object Detection With Pyramid Attention and Salient Edges | [Paper](https://www.researchgate.net/publication/332751907_Salient_Object_Detection_With_Pyramid_Attention_and_Salient_Edges)/[Code](https://github.com/wenguanwang/PAGE-Net)      

    

## 2018
**No.** | **Pub.** | **Title** | **Links** 
:-: | :-: | :-  | :-: 
01 | **CVPR** | A Bi-Directional Message Passing Model for Salient Object Detection | [Paper](https://pan.baidu.com/s/1akKVVipD8vIIv0XFrWND5Q)/[Code](https://github.com/zhangludl/A-bi-directional-message-passing-model-for-salient-object-detection)  
02 | **CVPR** | PiCANet: Learning Pixel-wise Contextual Attention for Saliency Detection | [Paper](http://arxiv.org/abs/1708.06433)/[Code](https://github.com/Ugness/PiCANet-Implementation)
03 | **CVPR** | PAGR: Progressive Attention Guided Recurrent Network for Salient Object Detection | [Paper](https://github.com/zhangxiaoning666/PAGR)/[Code](https://github.com/yangbinb/SalMetric/tree/master/PAGRN)
04 | **CVPR** | Learning to promote saliency detectors | [Paper](https://pan.baidu.com/s/1QvDmqruH8oU51_GrgsuXoA)/[Code](https://github.com/zengxianyu/lps)
05 | **CVPR** | Detect Globally, Refine Locally: A Novel Approach to Saliency Detection | [Paper](https://pan.baidu.com/s/1ydLI0koPfndehqMOAwrK_Q)/[Code](https://github.com/TiantianWang/CVPR18_detect_globally_refine_locally)
06 | **CVPR** | Salient Object Detection Driven by Fixation Prediction | [Paper](http://openaccess.thecvf.com/content_cvpr_2018/papers/Wang_Salient_Object_Detection_CVPR_2018_paper.pdf)/[Code](https://github.com/wenguanwang/ASNet)
07 | **IJCAI** | R3Net: Recurrent Residual Refinement Network for Saliency Detection | [Paper](https://www.ijcai.org/proceedings/2018/0095.pdf)/[Code](https://github.com/zijundeng/R3Net)
08 | **IJCAI** | LFR: Salient Object Detection by Lossless Feature Reflection | [Paper](https://pan.baidu.com/s/1DAyPHe_z0LJpKK8DxKF2dg)/[Code](https://github.com/Pchank/caffe-sal/blob/master/IIAU2018.md)
09 | **ECCV** | Contour Knowledge Transfer for Salient Object Detection | [Paper](http://link-springer-com-s.vpn.whu.edu.cn:9440/content/pdf/10.1007/978-3-030-01267-0_22.pdf)/[Code](https://github.com/lixin666/C2SNet)
10 | **ECCV** | Reverse Attention for Salient Object Detection | [Paper](http://arxiv.org/pdf/1807.09940)/[Code](https://github.com/ShuhanChen/RAS_ECCV18)
11 | **IEEE TIP** | An unsupervised game-theoretic approach to saliency detection | [Paper](https://pan.baidu.com/s/1U1O4oFK6ZALSghPjJv_5nA)/[Code](https://github.com/zengxianyu/uga)
12 | **arXiv** | Agile Amulet: Real-Time Salient Object Detection with Contextual Attention | [Paper](http://arxiv.org/pdf/1802.06960)/[Code](https://github.com/Pchank/caffe-sal/blob/master/IIAU2018.md)
13 | **arXiv** | HyperFusion-Net: Densely Reflective Fusion for Salient Object Detection | [Paper](http://arxiv.org/pdf/1804.05142)/[Code](https://github.com/Pchank/caffe-sal/blob/master/IIAU2018.md)
14 | **arXiv** | (TBOS)Three Birds One Stone: A Unified Framework for Salient Object Segmentation, Edge Detection and Skeleton Extraction | [Paper](https://arxiv.org/pdf/1803.09860.pdf)/Code
15 | **CVPR** | Deep Unsupervised Saliency Detection: A Multiple Noisy Labeling Perspective | [Paper](https://arxiv.org/abs/1803.10910)/[Code](https://github.com/kris-singh/Deep-Unsupervised-Saliency-Detection)


    
 
## 2017
**No.** | **Pub.** | **Title** | **Links** 
:-: | :-: | :-  | :-: 
01 | **CVPR** | DSS: Deeply Supervised Salient Object Detection with Short Connections | [Paper](http://arxiv.org/abs/1611.04849)/[Code](https://github.com/Andrew-Qibin/DSS)
02 | **CVPR** | Non-Local Deep Features for Salient Object Detection | [Paper](http://openaccess.thecvf.com/content_cvpr_2017/papers/Luo_Non-Local_Deep_Features_CVPR_2017_paper.pdf)/[Code](https://github.com/zhimingluo/NLDF)
03 | **CVPR** | Learning to Detect Salient Objects with Image-level Supervision | [Paper](http://openaccess.thecvf.com/content_cvpr_2017/papers/Wang_Learning_to_Detect_CVPR_2017_paper.pdf)/[Code](https://github.com/scott89/WSS)
04 | **CVPR** | SalGAN: visual saliency prediction with adversarial networks | [Paper](http://arxiv.org/abs/1701.01081)/[Code](https://github.com/Pchank/caffe-sal)
05 | **ICCV** | A Stagewise Refinement Model for Detecting Salient Objects in Images | [Paper](http://openaccess.thecvf.com/content_ICCV_2017/papers/Wang_A_Stagewise_Refinement_ICCV_2017_paper.pdf)/[Code](https://github.com/Pchank/caffe-sal)
06 | **ICCV** | Amulet: Aggregating Multi-level Convolutional Features for Salient Object Detection | [Paper](http://openaccess.thecvf.com/content_ICCV_2017/papers/Zhang_Amulet_Aggregating_Multi-Level_ICCV_2017_paper.pdf)/[Code](https://github.com/Pchank/caffe-sal)
07 | **ICCV** | Learning Uncertain Convolutional Features for Accurate Saliency Detection | [Paper](http://openaccess.thecvf.com/content_ICCV_2017/papers/Zhang_Learning_Uncertain_Convolutional_ICCV_2017_paper.pdf)/[Code](https://github.com/Pchank/caffe-sal)
08 | **ICCV** | Supervision by Fusion: Towards Unsupervised Learning of Deep Salient Object Detector  | [Paper](https://openaccess.thecvf.com/content_ICCV_2017/papers/Zhang_Supervision_by_Fusion_ICCV_2017_paper.pdf)/[Code](https://github.com/zhangyuygss/SVFSal.caffe)

 
  

## 2016
**No.** | **Pub.** | **Title** | **Links** 
:-: | :-: | :-  | :-: 
01 | **CVPR** | DHSNet: Deep hierarchical saliency network for salient object detection | [Paper](http://openaccess.thecvf.com/content_cvpr_2016/papers/Liu_DHSNet_Deep_Hierarchical_CVPR_2016_paper.pdf)/[Code](https://github.com/GuanWenlong/DHSNet-PyTorch)
02 | **CVPR** | ELD: Deep Saliency with Encoded Low level Distance Map and High Level Features | [Paper](http://www.arxiv.org/pdf/1604.05495v1.pdf)/[Code](https://github.com/gylee1103/SaliencyELD)
03 | **ECCV** | RFCN: Saliency detection with recurrent fully convolutional networks | [Paper](http://202.118.75.4/lu/Paper/ECCV2016/0865.pdf)/[Code](https://github.com/zengxianyu/RFCN)



# 3D RGB-D Saliency Detection <a id="3D RGB-D Saliency Detection" class="anchor" href="3D RGB-D Saliency Detection" aria-hidden="true"><span class="octicon octicon-link"></span></a> 


## 2021       
**No.** | **Pub.** | **Title** | **Links** 
:-: | :-: | :-  | :-: 
:triangular_flag_on_post: 01 | **CVPR** | Deep RGB-D Saliency Detection with Depth-Sensitive Attention and Automatic Multi-Modal Fusion | [Paper](https://arxiv.org/pdf/2103.11832.pdf)/[Code](https://github.com/sunpeng1996/DSA2F)   
:triangular_flag_on_post: 02 | **CVPR** | Calibrated RGB-D Saliency Object Detection | [Paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Ji_Calibrated_RGB-D_Salient_Object_Detection_CVPR_2021_paper.pdf)/[Code](https://github.com/jiwei0921/DCF)  
:triangular_flag_on_post: 03 | **AAAI** | RGB-D Salient Object Detection via 3D Convolutional Neural Networks | [Paper](https://arxiv.org/pdf/2101.10241.pdf)/[Code](https://github.com/PPOLYpubki/RD3D)
:triangular_flag_on_post: 04 | **IEEE TIP** | Hierarchical Alternate Interaction Network for RGB-D Salient Object Detection | [Paper](https://ieeexplore.ieee.org/document/9371407)/[Code](https://github.com/MathLee/HAINet)
:triangular_flag_on_post: 05 | **IEEE TIP** | CDNet: Complementary Depth Network for RGB-D Salient Object Detection | [Paper](https://ieeexplore.ieee.org/abstract/document/9366409)/[Code](https://github.com/blanclist/CDNet)
:triangular_flag_on_post: 06 | **ICME** | BTS-Net: Bi-directional Transfer-and-Selection Network for RGB-D Salient Object Detection | [Paper](https://arxiv.org/pdf/2104.01784.pdf)/[Code](https://github.com/zwbx/BTS-Net)
:triangular_flag_on_post: 07 | **ACMM** | Depth Quality-Inspired Feature Manipulation for Efficient RGB-D Salient Object Detection | [Paper](https://arxiv.org/pdf/2107.01779.pdf)/[Code](https://github.com/zwbx/DFM-Net)  
:triangular_flag_on_post: 08 | **ACMM** | TriTransNet RGB-D Salient Object Detection with a Triplet Transformer Embedding Network | Paper/[Code](https://github.com/liuzywen/TriTransNet-RGB-D-Salient-Object-Detection-with-a-Triplet-Transformer-Embedding-Network)
:triangular_flag_on_post: 09 | **ICCV** | RGB-D Saliency Detection via Cascaded Mutual Information Minimization | Paper/[Code](https://github.com/JingZhang617/cascaded_rgbd_sod)



## 2020
**No.** | **Pub.** | **Title** | **Links** 
:-: | :-: | :-  | :-: 
01 | **IEEE TIP** | ICNet: Information Conversion Network for RGB-D Based Salient Object Detection | [Paper](https://ieeexplore.ieee.org/document/9024241/authors)/[Code](https://github.com/MathLee/ICNet-for-RGBD-SOD)  
02 | **CVPR** | JL-DCF: Joint Learning and Densely-Cooperative Fusion Framework for RGB-D Salient Object Detection | [Paper](http://openaccess.thecvf.com/content_CVPR_2020/papers/Fu_JL-DCF_Joint_Learning_and_Densely-Cooperative_Fusion_Framework_for_RGB-D_Salient_CVPR_2020_paper.pdf)/[Code](https://github.com/kerenfu/JLDCF)  
03 | **CVPR** | UC-Net: Uncertainty Inspired RGB-D Saliency Detection via Conditional Variational Autoencoders | [Paper](http://openaccess.thecvf.com/content_CVPR_2020/papers/Zhang_UC-Net_Uncertainty_Inspired_RGB-D_Saliency_Detection_via_Conditional_Variational_Autoencoders_CVPR_2020_paper.pdf)/[Code](https://github.com/JingZhang617/UCNet)  
04 | **CVPR** | A2dele: Adaptive and Attentive Depth Distiller for Efficient RGB-D Salient Object Detection | [Paper](http://openaccess.thecvf.com/content_CVPR_2020/papers/Piao_A2dele_Adaptive_and_Attentive_Depth_Distiller_for_Efficient_RGB-D_Salient_CVPR_2020_paper.pdf)/[Code](https://github.com/OIPLab-DUT/CVPR2020-A2dele)  
05 | **CVPR** | Select, Supplement and Focus for RGB-D Saliency Detection | [Paper](http://openaccess.thecvf.com/content_CVPR_2020/papers/Zhang_Select_Supplement_and_Focus_for_RGB-D_Saliency_Detection_CVPR_2020_paper.pdf)/[Code](https://github.com/OIPLab-DUT/CVPR_SSF-RGBD)   
06 | **CVPR** | Learning Selective Self-Mutual Attention for RGB-D Saliency Detection | [Paper](https://openaccess.thecvf.com/content_CVPR_2020/papers/Liu_Learning_Selective_Self-Mutual_Attention_for_RGB-D_Saliency_Detection_CVPR_2020_paper.pdf)/[Code](https://github.com/nnizhang/S2MA)   
07 | **ECCV** | Accurate RGB-D Salient Object Detection via Collaborative Learning | [Paper](https://arxiv.org/pdf/2007.11782.pdf)/[Code](https://github.com/jiwei0921/CoNet)
08 | **ECCV** | Cross-Modal Weighting Network for RGB-D Salient Object Detection | [Paper](https://arxiv.org/pdf/2007.04901.pdf)/[Code](https://github.com/MathLee/CMWNet)
09 | **ECCV** | BBS-Net: RGB-D Salient Object Detection with a Bifurcated Backbone Strategy Network | [Paper](https://arxiv.org/pdf/2007.02713.pdf)/[Code](https://github.com/zyjwuyan/BBS-Net)
10 | **ECCV** | Hierarchical Dynamic Filtering Network for RGB-D Salient Object Detection | [Paper](https://arxiv.org/pdf/2007.06227.pdf)/[Code](https://github.com/lartpang/HDFNet)
11 | **ECCV** | Progressively Guided Alternate Refinement Network for RGB-D Salient Object Detection | [Paper](http://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123530511.pdf)/[Code](https://github.com/ShuhanChen/PGAR_ECCV20)
12 | **ECCV** | RGB-D Salient Object Detection with Cross-Modality Modulation and Selection | [Paper](https://arxiv.org/pdf/2007.07051.pdf)/[Code](https://github.com/Li-Chongyi/cmMS-ECCV20)
13 | **ECCV** | Cascade Graph Neural Networks for RGB-D Salient Object Detection | [Paper](https://arxiv.org/pdf/2008.03087.pdf)/[Code](https://github.com/LA30/Cas-Gnn)   
14 | **ECCV** | A Single Stream Network for Robust and Real-time RGB-D Salient Object Detection | [Paper](https://arxiv.org/pdf/2007.06811.pdf)/[Code](https://github.com/Xiaoqi-Zhao-DLUT/DANet-RGBD-Saliency)  
15 | **ECCV** | Asymmetric Two-Stream Architecture for Accurate RGB-D Saliency Detection | [Paper](http://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123730375.pdf)/[Code](https://github.com/sxfduter/ASTA)   
16 | **ACMM** | Is Depth Really Necessary for Salient Object Detection? | [Paper](https://arxiv.org/pdf/2006.00269.pdf)/[Code](https://github.com/JiaweiZhao-git/DASNet)
17 | **ACMM** | MMNet: Multi-Stage and Multi-Scale Fusion Network for RGB-D Salient Object Detection | [Paper](https://dl.acm.org/doi/pdf/10.1145/3394171.3413523)/[Code](https://github.com/gbliao/MMNet)
18 | **ACMM** | Feature Reintegration over Differential Treatment: A Top-down and Adaptive Fusion Network for RGB-D Salient Object Detection | [Paper](https://dl.acm.org/doi/pdf/10.1145/3394171.3413969)/[Code](https://github.com/jack-admiral/ACM-MM-FRDT)
19 | **IEEE TIP** | RGBD Salient Object Detection via Disentangled Cross-Modal Fusion | [Paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9165931)/[Code](https://github.com/haochen593/Disen_Fuse_TIP2020)
20 | **IEEE TIP** | Improved Saliency Detection in RGB-D Images Using Two-Phase Depth Estimation and Selective Deep Fusion | [Paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8976428)/Code
21 | **IEEE TIP** | Depth Potentiality-Aware Gated Attention Network for RGB-D Salient Object Detection | [Paper](https://arxiv.org/pdf/2003.08608.pdf)/[Code](https://github.com/JosephChenHub/DPANet)
22 | **IEEE TNNLS** | D3Net:Rethinking RGB-D Salient Object Detection: Models, Datasets, and Large-Scale Benchmarks | [Paper](https://arxiv.org/pdf/1907.06781.pdf)/[Code](https://github.com/DengPingFan/D3NetBenchmark)
23 | **IEEE TCSVT** | Revisiting Feature Fusion for RGB-T Salient Object Detection | [Paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9161021)/[Code](https://github.com/nexiakele/Revisiting-Feature-Fusion-for-RGB-T-Salient-Object-Detection)




## 2019
**No.** | **Pub.** | **Title** | **Links** 
:-: | :-: | :-  | :-: 
01 | **ICCV** | DMRA: Depth-induced Multi-scale Recurrent Attention Network for Saliency Detection | [Paper](http://openaccess.thecvf.com/content_ICCV_2019/papers/Piao_Depth-Induced_Multi-Scale_Recurrent_Attention_Network_for_Saliency_Detection_ICCV_2019_paper.pdf)/[Code](https://github.com/jiwei0921/DMRA_RGBD-SOD)
02 | **CVPR** | CPFP: Contrast Prior and Fluid Pyramid Integration for RGBD Salient Object Detection | [Paper](http://mftp.mmcheng.net/Papers/19cvprRrbdSOD.pdf)/[Code](https://github.com/JXingZhao/ContrastPrior)
03 | **IEEE TIP** | Three-stream Attention-aware Network for RGB-D Salient Object Detection | [Paper](http://ieeexplore.ieee.org/document/8603756/)/Code
04 | **IEEE PR** | Multi-modal fusion network with multi-scale multi-path and cross-modal interactions for RGB-D salient object detection | [Paper](https://www.sciencedirect.com/science/article/abs/pii/S0031320318303054)/Code
05 | **IEEE Access** | AFNet: Adaptive Fusion for RGB-D Salient Object Detection | [Paper](http://arxiv.org/abs/1901.01369?context=cs.CV)/[Code](https://github.com/Lucia-Ningning/Adaptive_Fusion_RGBD_Saliency_Detection)
06 | **arXiv** | CNN-based RGB-D Salient Object Detection: Learn, Select and Fuse | [Paper](https://arxiv.org/pdf/1909.09309.pdf)/Code
07 | **IEEE TIP** | RGB-T Salient Object Detection via Fusing Multi-Level CNN Features | [Paper](https://ieeexplore.ieee.org/abstract/document/8935533)/[Code](https://github.com/nexiakele/RGB-T-Salient-Object-Detection-via-Fusing-Multi-level-CNN-Features)
08 | **IEEE TMM** | RGB-T image saliency detection via collaborative graph learning | [Paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8744296)/Code

   

## 2018
**No.** | **Pub.** | **Title** | **Links** 
:-: | :-: | :-  | :-: 
01 | **CVPR** | PCA: Progressively Complementarity-aware Fusion Network for RGB-D Salient Object Detection | [Paper](https://www.researchgate.net/publication/329741351_Progressively_Complementarity-Aware_Fusion_Network_for_RGB-D_Salient_Object_Detection)/[Code](https://github.com/haochen593/PCA-Fuse_RGBD_CVPR18)
02 | **IEEE TIP** | Co-saliency detection for RGBD images based on multi-constraint feature matching and cross label propagation | [Paper](http://arxiv.org/abs/1710.05172)/[Code](https://github.com/rmcong/Results-for-2018TIP-RGBD-Co-saliency)
03 | **ICME** | PDNet: Prior-Model Guided Depth-enhanced Network for Salient Object Detection | [Paper](http://arxiv.org/pdf/1803.08636)/[Code](https://github.com/cai199626/PDNet)

  

## 2017
**No.** | **Pub.** | **Title** | **Links** 
:-: | :-: | :-  | :-: 
01 | **ICCV** | Learning RGB-D Salient Object Detection using background enclosure, depth contrast, and top-down features | [Paper](http://arxiv.org/pdf/1705.03607)/[Code](https://github.com/sshige/rgbd-saliency)
02 | **IEEE TIP** | DF: RGBD Salient Object Detection via Deep Fusion | [Paper](http://arxiv.org/pdf/1607.03333)/[Code](https://pan.baidu.com/s/1Y-PqAjuH9xREBjfl7H45HA)
03 | **IEEE TCyb** | CTMF: Cnns-based rgb-d saliency detection via cross-view transfer and multiview fusion | [Paper](http://ieeexplore.ieee.org/iel7/6221036/6352949/08091125.pdf)/[Code](https://github.com/haochen593/PCA-Fuse_RGBD_CVPR18)

  
## Traditional methods
**No.** | **Pub.** | **Title** | **Links** 
:-: | :-: | :-  | :-: 
01 | **MTA** | RGBD co-saliency detection via multiple kernel boosting and fusion | [Paper](http://www.onacademic.com/detail/journal_1000040179260010_4758.html)/[Code](https://github.com/ivpshu/RGBD-co-saliency-detection-via-multiple-kernel-boosting-and-fusion)
02 | **ICCV17** | An Innovative Salient Object Detection Using Center-Dark Channel Prior | [Paper](http://arxiv.org/abs/1710.04071v4)/[Code](https://github.com/ChunbiaoZhu/ACVR2017)
03 | **IEEE SPL** | Saliency detection for stereoscopic images based on depth confidence analysis and multiple cues fusion | [Paper](http://arxiv.org/abs/1710.05174)/[Code](https://github.com/rmcong/Code-for-DCMC-method)
04 | **IEEE SPL** | RGBD Co-saliency Detection via Bagging-Based Clustering | [Paper](http://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=7582474)/[Code](https://github.com/ivpshu/RGBD-co-saliency-detection-via-bagging-based-clustering)
05 | **CVPR** | Exploiting Global Priors for RGB-D Saliency Detection | [Paper](http://openaccess.thecvf.com/content_cvpr_workshops_2015/W14/html/Ren_Exploiting_Global_Priors_2015_CVPR_paper.html)/[Code](https://github.com/JianqiangRen/Global_Priors_RGBD_Saliency_Detection)




# 4D Light Field Saliency Detection  <a id="4D Light Field Saliency Detection" class="anchor" href="4D Light Field Saliency Detection" aria-hidden="true"><span class="octicon octicon-link"></span></a> 
**No.** | **Pub.** | **Title** | **Links** 
:-: | :-: | :-  | :-: 
01 | **TOMM** | MCA: Saliency Detection on Light Field: A Multi-Cue Approach | [Paper](http://www.linliang.net/wp-content/uploads/2017/07/ACMTOM_Saliency.pdf)/[Code](https://github.com/pencilzhang/HFUT-Lytro-dataset)
02 | **IJCAI** | DILF: Saliency Detection with a Deeper Investigation of Light Field | [Paper](http://pdfs.semanticscholar.org/4b17/fca1d67862e1fbffaf9ac64a1a73e0f20904.pdf)/[Code](https://github.com/pencilzhang/lightfieldsaliency_ijcai15)
03 | **CVPR** | WSC: A Weighted Sparse Coding Framework for Saliency Detection | [Paper](http://openaccess.thecvf.com/content_cvpr_2015/papers/Li_A_Weighted_Sparse_2015_CVPR_paper.pdf)/Code
04 | **IEEE PAMI** | Saliency Detection on Light-Field | [Paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=7570181)/[Code](https://download.csdn.net/download/deepvl/8076323?fps=1&locationNum=9)
05 | **ICCV** | Deep Learning for Light Field Saliency Detection | [Paper](http://openaccess.thecvf.com/content_ICCV_2019/papers/Wang_Deep_Learning_for_Light_Field_Saliency_Detection_ICCV_2019_paper.pdf)/[Code](https://github.com/OIPLab-DUT/ICCV2019_Deeplightfield_Saliency)
06 | **NeurIPS** | Memory-oriented Decoder for Light Field Salient Object Detection | [Paper](https://papers.nips.cc/paper/8376-memory-oriented-decoder-for-light-field-salient-object-detection.pdf)/[Code](https://github.com/jiwei0921/MoLF)
07 | **AAAI** | Exploit and Replace: An Asymmetrical Two-Stream Architecture for Versatile Light Field Saliency Detection | [Paper](https://drive.google.com/file/d/1uPkpB51MRMm_Zmvh1M2Z3nc3D8r32MR9/view?usp=drivesdk)/[Code](https://github.com/OIPLab-DUT/AAAI2020-Exploit-and-Replace-Light-Field-Saliency)
08 | **IEEE TCSVT** | A Multi-Task Collaborative Network for Light Field Salient Object Detection | [Paper](https://ieeexplore.ieee.org/abstract/document/9153018)/Code 
09 | **ArXiv** | DUT-LFSaliency: Versatile Dataset and Light Field-to-RGB Saliency Detection | [Paper](https://arxiv.org/pdf/2012.15124.pdf)/[Code](https://github.com/OIPLab-DUT/DUTLF-V2)   
10 | **ArXiv** | Learning Synergistic Attention for Light Field Salient Object Detection | [Paper](https://arxiv.org/pdf/2104.13916.pdf)/Code
11 | **ArXiv** | CMA-Net: A Cascaded Mutual Attention Network for Light Field Salient Object Detection | [Paper](https://arxiv.org/pdf/2105.00949.pdf)/Code
:triangular_flag_on_post: 12 | **IEEE TCyB** | PANet: Patch-Aware Network for Light Field Salient Object Detection | Paper/[Code](https://github.com/jyydlut/IEEE-TCYB-PANet)


      
# Video Salient Object Detection  <a id="Video Salient Object Detection" class="anchor" href="Video Salient Object Detection" aria-hidden="true"><span class="octicon octicon-link"></span></a> 

## 2021  
**No.** | **Pub.** | **Title** | **Links** 
:-: | :-: | :-  | :-: 
:triangular_flag_on_post: 01 | **CVPR** | Weakly Supervised Video Salient Object Detection | [Paper](https://arxiv.org/pdf/2104.02391.pdf)/[Code](https://github.com/wangbo-zhao/WSVSOD)     
:triangular_flag_on_post: 02 | **ArXiv** | Video Salient Object Detection via Adaptive Local-Global Refinement | [Paper](https://arxiv.org/pdf/2104.14360.pdf)/Code    
:triangular_flag_on_post: 03 | **ICIP** | Guidance and Teaching Network for Video Salient Object Detection | [Paper](https://arxiv.org/pdf/2105.10110.pdf)/[Code](https://github.com/GewelsJI/GTNet)    
:triangular_flag_on_post: 04 | **ACMM** | Multi-Source Fusion and Automatic Predictor Selection for Zero-Shot Video Object Segmentation | [Paper](https://arxiv.org/pdf/2108.05076.pdf)/[Code](https://github.com/Xiaoqi-Zhao-DLUT/Multi-Source-APS-ZVOS)   
:triangular_flag_on_post: 05 | **ICCV** | Dynamic Context-Sensitive Filtering Network for Video Salient Object Detection | Paper/[Code](https://github.com/Roudgers/DCFNet)   
:triangular_flag_on_post: 06 | **ICCV** | Full-Duplex Strategy for Video Object Segmentation | [Paper](https://arxiv.org/pdf/2108.03151.pdf)/[Code](https://github.com/GewelsJI/FSNet) 

## 2020  
**No.** | **Pub.** | **Title** | **Links** 
:-: | :-: | :-  | :-: 
01 | **CVPR** | STAViS: Spatio-Temporal AudioVisual Saliency Network | [Paper](https://openaccess.thecvf.com/content_CVPR_2020/papers/Tsiami_STAViS_Spatio-Temporal_AudioVisual_Saliency_Network_CVPR_2020_paper.pdf)/Code  
02 | **ECCV** | Unified Image and Video Saliency Modeling | [Paper](https://arxiv.org/pdf/2003.05477.pdf)/[Code](https://github.com/rdroste/unisal)    
03 | **ECCV** | Measuring the importance of temporal features in video saliency | [Paper](http://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123730664.pdf)/Code  
04 | **ECCV** | TENet: Triple Excitation Network for Video Salient Object Detection | [Paper](http://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123500205.pdf)/[Code](https://github.com/OliverRensu/TENet-Triple-Excitation-Network-for-Video-Salient-Object-Detection) 
05 | **IEEE TIP** | Learning Long-term Structural Dependencies for Video Salient Object Detection | [Paper](https://ieeexplore.ieee.org/document/9199537)/[Code](https://github.com/bowangscut/LSD_GCN-for-VSOD)  
06 | **IEEE Access** | Cross Complementary Fusion Network for Video Salient Object Detection | [Paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9250449)/[Code](https://github.com/zi-yang-w/CCNet) 
07 | **AAAI** | Pyramid Constrained Self-Attention Network for Fast Video Salient Object Detection | [Paper](http://mftp.mmcheng.net/Papers/20AAAI-PCSA.pdf)/[Code](https://github.com/guyuchao/PyramidCSA)   

## 2019  
**No.** | **Pub.** | **Title** | **Links** 
:-: | :-: | :-  | :-: 
01 | **ICCV** | Motion Guided Attention for Video Salient Object Detection | [Paper](https://arxiv.org/abs/1909.07061)/[Code](https://github.com/lhaof/Motion-Guided-Attention)  
02 | **ICCV** | Semi-Supervised Video Salient Object Detection Using Pseudo-Labels | [Paper](http://openaccess.thecvf.com/content_ICCV_2019/papers/Yan_Semi-Supervised_Video_Salient_Object_Detection_Using_Pseudo-Labels_ICCV_2019_paper.pdf)/[Code](https://github.com/Kinpzz/RCRNet-Pytorch)   
03 | **ICCV** | Temporally-Aggregating Spatial Encoder-Decoder Network for Video Saliency Detection | [Paper](http://openaccess.thecvf.com/content_ICCV_2019/html/Min_TASED-Net_Temporally-Aggregating_Spatial_Encoder-Decoder_Network_for_Video_Saliency_Detection_ICCV_2019_paper.html)/[Code](https://github.com/MichiganCOG/TASED-Net)   
04 | **ICCV** | RANet：Ranking attention Network for Fast Video Object Segmentation | [Paper](https://arxiv.org/abs/1908.06647)/[Code](https://github.com/Storife/RANet)   
05 | **CVPR** | Shifting More Attention to Video Salient Objection Detection | [Paper](https://github.com/DengPingFan/DAVSOD/blob/master/%5B2019%5D%5BCVPR%5D%5BOral%5D【SSAV】【DAVSOD】Shifting%20More%20Attention%20to%20Video%20Salient%20Object%20Detection.pdf)/[Code](https://github.com/DengPingFan/DAVSOD)   
06 | **CVPR** | Learning Unsupervised Video Object Segmentation through Visual Attention | [Paper](https://www.researchgate.net/publication/332751903_Learning_Unsupervised_Video_Object_Segmentation_Through_Visual_Attention)/[Code](https://github.com/wenguanwang/AGS)   
07 | **CVPR** | See More, Know More: Unsupervised Video Object Segmentation with Co-Attention Siamese Networks | [Paper](http://openaccess.thecvf.com/content_CVPR_2019/papers/Lu_See_More_Know_More_Unsupervised_Video_Object_Segmentation_With_Co-Attention_CVPR_2019_paper.pdf)/[Code](https://github.com/carrierlxk/COSNet)  



## 2018  
**No.** | **Pub.** | **Title** | **Links** 
:-: | :-: | :-  | :-: 
01 | **ECCV** | Pyramid Dilated Deeper CoonvLSTM for Video Salient Object Detection | [Paper](https://github.com/shenjianbing/PDBConvLSTM/blob/master/Pyramid%20Dilated%20Deeper%20CoonvLSTM%20for%20Video%20Salient%20Object%20Detection.pdf)/[Code](https://github.com/shenjianbing/PDB-ConvLSTM)
02 | **ECCV** | DeepVS: A Deep Learning Based Video Saliency Prediction Approach | [Paper](http://openaccess.thecvf.com/content_ECCV_2018/html/Lai_Jiang_DeepVS_A_Deep_ECCV_2018_paper.html)/[Code](https://github.com/remega/OMCNN_2CLSTM)
03 | **CVPR** | Revisiting Video Saliency: A Large-scale Benchmark and a New Model | [Paper](https://github.com/wenguanwang/DHF1K/blob/master/(pami19)DynamicSaliency.pdf)/[Code](https://github.com/wenguanwang/DHF1K)  
04 | **CVPR** | Flow Guided Recurrent Neural Encoder for Video Salient Object Detection | [Paper](http://openaccess.thecvf.com/content_cvpr_2018/CameraReady/1226.pdf)/Code  
05 | **IEEE TIP** | Video Salient Object Detection via Fully Convolutional Networks | [Paper](https://arxiv.org/pdf/1702.00871.pdf)/[Code](https://github.com/wenguanwang/ViSalientObject)

## 2017  
**No.** | **Pub.** | **Title** | **Links** 
:-: | :-: | :-  | :-: 
01 | **IEEE TIP** | Learning to Detect Video Saliency with HEVC Features | [Paper](https://ieeexplore.ieee.org/abstract/document/7742914/)/[Code](https://github.com/remega/Compressd_Domain_SaliencyPrediction)

# Earlier Methods  <a id="Earlier Methods" class="anchor" href="Earlier Methods" aria-hidden="true"><span class="octicon octicon-link"></span></a> 

**No.** | **Pub.** | **Title** | **Links** 
:-: | :-: | :-  | :-: 
01 | IEEE TIP15 | Salient object detection: A benchmark | [Paper](https://arxiv.org/pdf/1501.02741.pdf)/Code
02 | IEEE TCSVT18 | Review of visual saliency detectionwith comprehensive information | [Paper](https://arxiv.org/pdf/1803.03391.pdf)/Code
03 | ACM TIST18 | A review of co-saliency detection algorithms: Fundamentals, applications, and challenges | [Paper](https://arxiv.org/pdf/1604.07090.pdf)/Code
04 | IEEE TSP18 | Advanced deep-learning techniques for salient and category-specific object detection: A survey| [Paper](https://ieeexplore.ieee.org/document/8253582)/Code
05 | IJCV18 | Attentive systems: A survey | [Paper](https://link.springer.com/article/10.1007/s11263-017-1042-6)/Project
06 | ECCV18 | Salient Objects in Clutter: Bringing Salient Object Detection to the Foreground | [Paper](http://mftp.mmcheng.net/Papers/18ECCV-SOCBenchmark.pdf)/[Code](http://dpfan.net/socbenchmark/)
07 | CVM18 | Salient object detection: A survey | [Paper](https://link.springer.com/content/pdf/10.1007/s41095-019-0149-9.pdf)/Code
08 | IEEE TNNLS19 | Salient Object detection with deep learning: Areview | [Paper](https://arxiv.org/pdf/1807.05511.pdf)/Code
09 | arXiv19 |  Salient Object Detection in the Deep Learning Era-An In-Depth Survey | [Paper](https://arxiv.org/pdf/1904.09146.pdf)/[Code](https://github.com/wenguanwang/SODsurvey) 
10 | CVM21 | RGB-D Salient Object Detection: A Survey | [Paper](https://arxiv.org/abs/2008.00230)/[Code](https://github.com/taozh2017/RGBD-SODsurvey)

The part of the collection is thanks to [Deng-Ping Fan](http://dpfan.net) and [Tao Zhou](https://github.com/taozh2017).

* Salient Object Detection in the Deep Learning Era: An In-Depth Survey. [paper link](https://arxiv.org/pdf/1904.09146.pdf).
* This is a paper list published by another author. [here](https://github.com/ArcherFMY/Paper_Reading_List/blob/master/Image-01-Salient-Object-Detection.md)
* RGB-D Salient Object Detection: A Survey. [project link](https://github.com/taozh2017/RGBD-SODsurvey).



# Comparison with state-of-the-arts  <a id="Comparison with state-of-the-arts" class="anchor" href="Comparison with state-of-the-arts" aria-hidden="true"><span class="octicon octicon-link"></span></a> 
* [Here](https://github.com/ArcherFMY/sal_eval_toolbox) includes the performance comparison of almost all 2D salient object detection algorithms. 
* [Here](http://dpfan.net/d3netbenchmark/) includes the performance comparison of almost all 3D RGB-D salient object detection algorithms. 

# The SOD dataset download    <a id="The SOD dataset download" class="anchor" href="The SOD dataset download" aria-hidden="true"><span class="octicon octicon-link"></span></a> 
* 2D SOD datasets [download1](https://github.com/TinyGrass/SODdataset) or [download2](https://github.com/ArcherFMY/sal_eval_toolbox), [download3](https://github.com/magic428/awesome-segmentation-saliency-dataset).
* 3D SOD datasets [download](https://github.com/jiwei0921/RGBD-SOD-datasets).  
* 4D SOD datasets [download](https://github.com/jiwei0921/MoLF).
* Video SOD datasets [download](http://dpfan.net/DAVSOD/).

# Evaluation Metrics  <a id="Evaluation Metrics" class="anchor" href="Evaluation Metrics" aria-hidden="true"><span class="octicon octicon-link"></span></a> 
* Saliency maps evaluation.      
This link near all evaluation metrics for salient object detection including E-measure, S-measure, F-measure, MAE scores and PR curves or bar metrics.
You can found in [here](https://github.com/jiwei0921/Saliency-Evaluation-Toolbox).        

* Saliency Dataset evaluation.       
This repo can compute the ratio of obj.area and obj.contrast on binary saliency dataset. This Toolbox contains two evaluation metrics, including obj(object).area and obj.contrast.     
You can found in [here](https://github.com/jiwei0921/Saliency-dataset-evaluation).      

### AI Conference Deadlines
[Realted AI Conference deadline](https://aideadlin.es/?sub=ML,CV,NLP,RO,SP,DM)     
[Realted AI Conference Accepted Rate](https://github.com/lixin4ever/Conference-Acceptance-Rate)
