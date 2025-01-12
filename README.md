# SOD CNNs-based Read List       

In this repository, we mainly focus on deep learning based saliency methods (**2D RGB, 3D RGB-D/T, Video SOD and 4D Light Field**) and provide a summary (**Code and Paper**). We hope this repo can help you to better understand saliency detection in the deep learning era.        

--------------------------------------------------------------------------------------
 :heavy_exclamation_mark:  **2D SOD**: Add two ACM MM papers, seven CVPR24 papers, six ECCV'24 paper, two PAMI papers.                 
 :heavy_exclamation_mark:  **3D SOD**: Add one ECCV'24 paper, one ACM MM'24 paper.    
 :heavy_exclamation_mark:  **LF SOD**: Add two IEEE TCSVT papers, one arXiv'24 paper.   
 :heavy_exclamation_mark:  **Video SOD** :  Add one AAAI23 papers, one NeurIPS24 paper. 
 
 [Camouflaged Object Detection](https://arxiv.org/abs/2102.10274) is a closely-related task of SOD, with paper summary of [this link](https://github.com/ChunmingHe/awesome-concealed-object-segmentation).

:running: **We will keep updating it.** :running:    
--------------------------------------------------------------------------------------


------
 

## Content:

1. [An overview of the Paper List](#overall)
2. [2D RGB Saliency Detection](#2DSOD) 
3. [3D RGB-D/T Saliency Detection](#3DSOD) 
4. [4D Light Field Saliency Detection](#4DSOD) 
5. [Video Saliency Detection](#VSOD) 
6. [Survery and earlier Methods](#survey) 
7. [The SOD dataset download](#data) 
8. [Evaluation Metrics](#eval) 
9. [SOD Leaderboard](#leaderboard)



------

<a name="overall"></a>   
# Overall 
![avatar](https://github.com/jiwei0921/SOD-CNNs-based-code-summary-/blob/master/SOD-2019.7.23.jpg)
    
<a name="2DSOD"></a> 
# 2D RGB Saliency Detection <a id="2D RGB Saliency Detection" class="anchor" href="2D RGB Saliency Detection" aria-hidden="true"><span class="octicon octicon-link"></span></a>    

## 2025      
**No.** | **Pub.** | **Title** | **Links** 
:-: | :-: | :-  | :-: 
:triangular_flag_on_post: 01 | **PAMI** | Conditional Diffusion Models for Camouflaged and Salient Object Detection | [Paper](https://ieeexplore.ieee.org/abstract/document/10834569)/[Code](https://github.com/Rapisurazurite/CamoDiffusion)  

## 2024      
**No.** | **Pub.** | **Title** | **Links** 
:-: | :-: | :-  | :-: 
:triangular_flag_on_post: 01 | **WACV** | Unsupervised and semi-supervised co-salient object detection via segmentation frequency statistics | [Paper](https://arxiv.org/pdf/2311.06654.pdf)/Code   
:triangular_flag_on_post: 02 | **WACV** | 3SD: Self-Supervised Saliency Detection With No Labels | [Paper](https://openaccess.thecvf.com/content/WACV2024/papers/Yasarla_3SD_Self-Supervised_Saliency_Detection_With_No_Labels_WACV_2024_paper.pdf)/[Code](https://github.com/rajeevyasarla/3SD)    
:triangular_flag_on_post: 03 | **WACV** | Learning Saliency From Fixations | [Paper](https://openaccess.thecvf.com/content/WACV2024/papers/Djilali_Learning_Saliency_From_Fixations_WACV_2024_paper.pdf)/[Code](https://github.com/YasserdahouML/SalTR)     
:triangular_flag_on_post: 04 | **WACV** | Salient Object Detection for Images Taken by People With Vision Impairments | [Paper](https://openaccess.thecvf.com/content/WACV2024/papers/Reynolds_Salient_Object_Detection_for_Images_Taken_by_People_With_Vision_WACV_2024_paper.pdf)/[Code](https://vizwiz.org/tasks-and-datasets/salient-object-detection/)     
:triangular_flag_on_post: 05 | **WACV** | Defense Against Adversarial Cloud Attack on Remote Sensing Salient Object Detection | [Paper](https://arxiv.org/pdf/2311.06654.pdf)/Code  
:triangular_flag_on_post: 06 | **ICASSP** | Zero-Shot Co-salient Object Detection Framework | [Paper](https://arxiv.org/abs/2309.05499)/[Code](https://github.com/hkxiao/zs-cosod)  
:triangular_flag_on_post: 07 | **CVPR** | VSCode: General Visual Salient and Camouflaged Object Detection with 2D Prompt Learning | [Paper](https://arxiv.org/pdf/2311.15011.pdf)/Code
:triangular_flag_on_post: 08 | **AAAI** | WeakPCSOD: Overcoming the Bias of Box Annotations for Weakly Supervised Point Cloud Salient Object Detection | [Paper](https://ojs.aaai.org/index.php/AAAI/article/view/28403)/Code
:triangular_flag_on_post: 09 | **AAAI** | SeqRank: Sequential Ranking of Salient Objects | [Paper](https://ojs.aaai.org/index.php/AAAI/article/view/27964)/[Code](https://github.com/guanhuankang/SeqRank) 
:triangular_flag_on_post: 10 | **AAAI** | Finding Visual Saliency in Continuous Spike Stream | [Paper](https://ojs.aaai.org/index.php/AAAI/article/view/28610)/[Code](https://github.com/BIT-Vision/SVS) 
:triangular_flag_on_post: 11 | **CVPR** | COSALPURE: Learning Concept from Group Images for Robust Co-Saliency | [Paper](https://arxiv.org/pdf/2403.18554.pdf)/[Code](https://v1len.github.io/CosalPure/) 
:triangular_flag_on_post: 12 | **IJCAI** | Unified Unsupervised Salient Object Detection via Knowledge Transfer | [Paper](https://arxiv.org/pdf/2404.14759)/[Code](https://github.com/I2-Multimedia-Lab/A2S-v3) 
:triangular_flag_on_post: 13 | **TII** | MINet: Multi-scale Interactive Network for Real-time Salient Object Detection of Strip Steel Surface Defects | [Paper](https://arxiv.org/pdf/2405.16096)/[Code](https://github.com/Kunye-Shen/MINet) 
:triangular_flag_on_post: 14 | **ICML** | Size-invariance Matters: Rethinking Metrics and Losses for Imbalanced Multi-object Salient Object Detection | [Paper](https://arxiv.org/pdf/2405.09782)/[Code](https://github.com/Ferry-Li/SI-SOD) 
:triangular_flag_on_post: 15 | **ICML** | Spider: A Unified Framework for Context-dependent Concept Segmentation | [Paper](https://arxiv.org/pdf/2405.01002)/[Code](https://github.com/Xiaoqi-Zhao-DLUT/Spider-UniCDSeg) 
:triangular_flag_on_post: 16 | **ICML** | Diving into Underwater: Segment Anything Model Guided Underwater Salient Instance Segmentation and A Large-scale Dataset | Paper/[Code](https://github.com/LiamLian0727/USIS10K) 
:triangular_flag_on_post: 17 | **CVPR** | Domain Separation Graph Neural Networks for Saliency Object Ranking | [Paper](https://openaccess.thecvf.com/content/CVPR2024/papers/Wu_Domain_Separation_Graph_Neural_Networks_for_Saliency_Object_Ranking_CVPR_2024_paper.pdf)/[Code](https://github.com/Wu-ZJ/DSGNN) 
:triangular_flag_on_post: 18 | **CVPR** | Advancing Saliency Ranking with Human Fixations: Dataset Models and Benchmarks | [Paper](https://openaccess.thecvf.com/content/CVPR2024/papers/Deng_Advancing_Saliency_Ranking_with_Human_Fixations_Dataset_Models_and_Benchmarks_CVPR_2024_paper.pdf)/[Code](https://github.com/EricDengbowen/QAGNet) 
:triangular_flag_on_post: 19 | **CVPR** | Task-Adaptive Saliency Guidance for Exemplar-free Class Incremental Learning | [Paper](https://openaccess.thecvf.com/content/CVPR2024/papers/Liu_Task-Adaptive_Saliency_Guidance_for_Exemplar-free_Class_Incremental_Learning_CVPR_2024_paper.pdf)/[Code](https://github.com/scok30/tass) 
:triangular_flag_on_post: 20 | **CVPR** | Unsupervised Salient Instance Detection | [Paper](https://openaccess.thecvf.com/content/CVPR2024/papers/Tian_Unsupervised_Salient_Instance_Detection_CVPR_2024_paper.pdf)/Code
:triangular_flag_on_post: 21 | **CVPR** | DiffSal: Joint Audio and Video Learning for Diffusion Saliency Prediction | [Paper](https://openaccess.thecvf.com/content/CVPR2024/papers/Xiong_DiffSal_Joint_Audio_and_Video_Learning_for_Diffusion_Saliency_Prediction_CVPR_2024_paper.pdf)/[Code](https://junwenxiong.github.io/DiffSal) 
:triangular_flag_on_post: 22 | **TMM** | ADMNet: Attention-guided Densely Multi-scale Network for Lightweight Salient Object Detection | [Paper](https://ieeexplore.ieee.org/document/10555313)/[Code](https://github.com/Kunye-Shen/ADMNet)
:triangular_flag_on_post: 23 | **ACMMM** | Multi-Scale and Detail-Enhanced Segment Anything Model for Salient Object Detection | [Paper](https://arxiv.org/pdf/2408.04326)/[Code](https://github.com/BellyBeauty/MDSAM)
:triangular_flag_on_post: 24 | **ACMMM** | Instance-Level Panoramic Audio-Visual Saliency Detection and Ranking | [Paper](https://openreview.net/pdf?id=0Q9zTGHOda)/Code
:triangular_flag_on_post: 25 | **ECCV** | CONDA: Condensed Deep Association Learning for Co-Salient Object Detection | [Paper](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/06695.pdf)/[Code](https://github.com/dragonlee258079/CONDA)
:triangular_flag_on_post: 26 | **ECCV** | Self-supervised co-salient object detection via feature correspondences at multiple scales | [Paper](https://arxiv.org/pdf/2403.11107)/[Code](https://github.com/sourachakra/SCoSPARC)
:triangular_flag_on_post: 27 | **ECCV** | SHINE: Saliency-aware HIerarchical NEgative Ranking for Compositional Temporal Grounding | [Paper](https://arxiv.org/pdf/2407.05118)/[Code](https://github.com/zxccade/SHINE)  
:triangular_flag_on_post: 28 | **ECCV** | DSMix: Distortion-Induced Saliency Map Based Pre-training for No-Reference Image Quality Assessment | [Paper](https://arxiv.org/pdf/2407.03886)/[Code](https://github.com/I2-Multimedia-Lab/DSMix)
:triangular_flag_on_post: 29 | **ECCV** | Salience-Based Adaptive Masking: Revisiting Token Dynamics for Enhanced Pre-training | [Paper](https://arxiv.org/pdf/2404.08327)/Code
:triangular_flag_on_post: 30 | **ECCV** | Data Augmentation via Latent Diffusion for Saliency Prediction | [Paper](https://arxiv.org/pdf/2409.07307)/[Code](https://github.com/IVRL/AugSal)
:triangular_flag_on_post: 31 | **PAMI** | Divide-and-Conquer: Confluent Triple-Flow Network for RGB-T Salient Object Detection | [Paper](https://ieeexplore.ieee.org/document/10778650/authors#authors)/[Code](https://github.com/CSer-Tang-hao/ConTriNet_RGBT-SOD)


## 2023      
**No.** | **Pub.** | **Title** | **Links** 
:-: | :-: | :-  | :-: 
01 | **AAAI** | LeNo: Adversarial Robust Salient Object Detection Networks with Learnable Noise | [Paper](https://arxiv.org/abs/2210.15392)/[Code](https://github.com/ssecv/LeNo)  
02 | **AAAI** | Pixel is All You Need: Adversarial Trajectory-Ensemble Active Learning for Salient Object Detection | [Paper](https://arxiv.org/pdf/2212.06493.pdf)/Code  
03 | **AAAI** | Memory-aided Contrastive Consensus Learning for Co-salient Object Detection | [Paper](https://scholar.google.com/citations?view_op=list_works&hl=en&hl=en&user=TZRzWOsAAAAJ)/[Code](https://github.com/ZhengPeng7/MCCL#)   
04 | **TNNLS** | Multi-Projection Fusion and Refinement Network for Salient Object Detection in 360◦ Omnidirectional Image | [Paper](https://arxiv.org/pdf/2212.12378.pdf)/[Code](https://rmcong.github.io/proj_MPFRNet.html)  
05 | **IEEE TIP** | Boosting Broader Receptive Fields for Salient Object Detection | [Paper](https://ieeexplore.ieee.org/abstract/document/10006743)/[Code](https://github.com/iCVTEAM/BBRF-TIP)   
06 | **IEEE TPAMI** | Co-Salient Object Detection with Co-Representation Purification | [Paper](https://arxiv.org/pdf/2303.07670.pdf)/[Code](https://github.com/ZZY816/CoRP)   
07 | **CVPR** | Texture-guided Saliency Distilling for Unsupervised Salient Object Detection | [Paper](https://arxiv.org/pdf/2207.05921.pdf)/[Code](https://github.com/moothes/A2S-v2)   
08 | **CVPR** | Discriminative Co-Saliency and Background Mining Transformer for Co-Salient Object Detection | [Paper](https://arxiv.org/pdf/2305.00514.pdf)/[Code](https://github.com/dragonlee258079/DMT)   
09 | **CVPR** | Sketch2Saliency: Learning to Detect Salient Objects from Human Drawings | [Paper](https://arxiv.org/pdf/2303.11502.pdf)/[Code](https://ayankumarbhunia.github.io/Sketch2Saliency/)   
10 | **CVPR** | Boosting Low-Data Instance Segmentation by Unsupervised Pre-training with Saliency Prompt | [Paper](https://arxiv.org/pdf/2302.01171.pdf)/[Code](https://github.com/lifuguan/saliency_prompt)   
11 | **CVPR** | Pixels, Regions, and Objects: Multiple Enhancement for Salient Object Detection | [Paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Wang_Pixels_Regions_and_Objects_Multiple_Enhancement_for_Salient_Object_Detection_CVPR_2023_paper.pdf)/[Code](https://github.com/yiwangtz/MENet)   
12 | **CVPR** | Co-Salient Object Detection with Uncertainty-aware Group Exchange-Masking | [Paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Wu_Co-Salient_Object_Detection_With_Uncertainty-Aware_Group_Exchange-Masking_CVPR_2023_paper.pdf)/Code  
13 | **CVPR** | Modeling the Distributional Uncertainty for Salient Object Detection Models | [Paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Tian_Modeling_the_Distributional_Uncertainty_for_Salient_Object_Detection_Models_CVPR_2023_paper.pdf)/[Code](https://npucvr.github.io/Distributional_uncer/)  
14 | **ACM MM** | Recurrent Multi-scale Transformer for High-Resolution Salient Object Detection | [Paper](https://arxiv.org/pdf/2308.03826.pdf)/[Code](https://github.com/DrowsyMon/RMFormer)  
15 | **TOMM** | PAV-SOD: A New Task Towards Panoramic Audiovisual Saliency Detection | [Paper](https://drive.google.com/file/d/1-1RcARcbz4pACFzkjXcp6MP8R9CGScqI/view)/[Code](https://github.com/Jun-Pu/PAV-SOD)  
16 | **ICCV** | Counterfactual-based Saliency Map: Towards Visual Contrastive Explanations for Neural Networks | [Paper](https://openaccess.thecvf.com/content/ICCV2023/papers/Wang_Counterfactual-based_Saliency_Map_Towards_Visual_Contrastive_Explanations_for_Neural_Networks_ICCV_2023_paper.pdf)/Code
17 | **ACM MM** | Distortion-aware Transformer in 360° Salient Object Detection | [Paper](https://arxiv.org/abs/2308.03359)/[Code](https://github.com/yjzhao19981027/DATFormer/) 
18 | **ACM MM** | Towards End-to-End Unsupervised Saliency Detection with Self-Supervised Top-Down Context | [Paper](https://dl.acm.org/doi/pdf/10.1145/3581783.3612212)/Code 
19 | **ACM MM** | Partitioned Saliency Ranking with Dense Pyramid Transformers | [Paper](https://arxiv.org/pdf/2308.00236.pdf)/[Code](https://github.com/ssecv/PSR) 
20 | **ACM MM** | Co-Salient Object Detection with Semantic-Level Consensus Extraction and Dispersion | [Paper](https://arxiv.org/abs/2309.07753v1)/Code  
21 | **TMM** | Towards Complete and Detail-Preserved Salient Object Detection | [Paper](https://ieeexplore.ieee.org/document/10287608)/[Code](https://github.com/BarCodeReader/SelfReformer) 
22 | **arXiv** | Unified-modal Salient Object Detection via Adaptive Prompt Learning | [Paper](https://arxiv.org/pdf/2311.16835.pdf)/Code
23 | **arXiv** | All in One: RGB, RGB-D, and RGB-T Salient Object Detection | [Paper](https://arxiv.org/pdf/2311.14746.pdf)/Code
24 | **NeurIPS** | What Do Deep Saliency Models Learn about Visual Attention? | [Paper](https://arxiv.org/abs/2310.09679)/[Code](https://arxiv.org/pdf/2310.09679.pdf)  
25 | **PAMI** | CADC++: Advanced Consensus-Aware Dynamic Convolution for Co-Salient Object Detection | [Paper](https://www.computer.org/csdl/journal/tp/5555/01/10339864/1SBL7kZYYyA)/Code  
26 | **IEEE TIP** | USOD10K: A New Benchmark Dataset for Underwater Salient Object Detection | [Paper](https://ieeexplore.ieee.org/document/10102831)/[Code](https://github.com/LinHong-HIT/USOD10K)  
27 | **IEEE TMM** | Spectrum-driven Mixed-frequency Network for Hyperspectral Salient Object Detection | [Paper](https://ieeexplore.ieee.org/document/10313066/)/[Code](https://github.com/laprf/SMN)  
28 | **IEEE TIP** | Rethinking Object Saliency Ranking: A Novel Whole-flow Processing Paradigm | [Paper](https://arxiv.org/pdf/2312.03226.pdf)/[Code](https://github.com/MengkeSong/Saliency-Ranking-Paradigm) 






## 2022       
**No.** | **Pub.** | **Title** | **Links** 
:-: | :-: | :-  | :-: 
01 | **AAAI** | Unsupervised Domain Adaptive Salient Object Detection Through Uncertainty-Aware Pseudo-Label Learning | [Paper](https://www.aaai.org/AAAI22Papers/AAAI-604.YanP.pdf)/[Code](https://github.com/Kinpzz/UDASOD-UPL)  
02 | **AAAI** | A Causal Debiasing Framework for Unsupervised Salient Object Detection | [Paper](https://www.aaai.org/AAAI22Papers/AAAI-108.LinX.pdf)/[Code](https://github.com/Jaiharish-passion07/AI_Project)  
03 | **AAAI** | Energy-Based Generative Cooperative Saliency Prediction | [Paper](https://www.aaai.org/AAAI22Papers/AAAI-1516.ZhangJ.pdf)/[Code](https://github.com/JingZhang617/SalCoopNets)  
04 | **AAAI** | Weakly-Supervised Salient Object Detection Using Point Supervison | [Paper](https://www.aaai.org/AAAI22Papers/AAAI-461.GaoS.pdf)/[Code](https://github.com/shuyonggao/PSOD)    
05 | **AAAI** | TRACER: Extreme Attention Guided Salient Object Tracing Network | [Paper](https://arxiv.org/pdf/2112.07380.pdf)/[Code](https://github.com/Karel911/TRACER)  
06 | **AAAI** | I can find you! Boundary-guided Separated Attention Network for Camouflaged Object Detection | [Paper](https://www.aaai.org/AAAI22Papers/AAAI-6565.ZhuH.pdf)/[Code](https://github.com/WolfberryCoke/BSA-Net)  
07 | **WACV** | Recursive Contour-Saliency Blending Network for Accurate Salient Object Detection | [Paper](https://openaccess.thecvf.com/content/WACV2022/papers/Ke_Recursive_Contour-Saliency_Blending_Network_for_Accurate_Salient_Object_Detection_WACV_2022_paper.pdf)/[Code](https://github.com/BarCodeReader/RCSB-PyTorch)  
08 | **IEEE TPAMI** | PoolNet+: Exploring the Potential of Pooling for Salient Object Detection | [Paper](https://mftp.mmcheng.net/Papers/21PAMI-PoolNet.pdf)/[Code](http://mmcheng.net/poolnet/)  
09 | **IEEE TPAMI** | A Highly Efficient Model to Study the Semantics of Salient Object Detection | [Paper](https://mftp.mmcheng.net/Papers/21PAMI-Sal100K.pdf)/[Code](https://mmcheng.net/sod100k/)  
10 | **IEEE TGRS** | Lightweight Salient Object Detection in Optical Remote Sensing Images via Feature Correlation | [Paper](https://arxiv.org/abs/2201.08049)/[Code](https://github.com/MathLee/CorrNet)  
11 | **TOMM** | Disentangle Saliency Detection into Cascaded Detail Modeling and Body Filling | [Paper](https://arxiv.org/pdf/2202.04112.pdf)/[Code](https://github.com/KingJamesSong/DisentangleSaliency)    
12 | **TMM** | Noise-Sensitive Adversarial Learning for Weakly Supervised Salient Object Detection | [Paper](https://ieeexplore.ieee.org/abstract/document/9716868/authors#authors)/[Code](https://github.com/wuweia123/IEEE-TMM-NSALWSS) 
13 | **ArXiv** | Joint Learning of Salient Object Detection, Depth Estimation and Contour Extraction | [Paper](https://arxiv.org/pdf/2203.04895.pdf)/Code 
14 | **ArXiv** | A Unified Transformer Framework for Group-based Segmentation: Co-Segmentation, Co-Saliency Detection and Video Salient Object Detection | [Paper](https://arxiv.org/pdf/2203.04708.pdf)/[Code](https://github.com/suyukun666/UFO) 
15 | **IEEE TCyb** | Adjacent Context Coordination Network for Salient Object Detection in Optical Remote Sensing Images | [Paper](https://arxiv.org/pdf/2203.13664.pdf)/[Code](https://github.com/MathLee/ACCoNet) 
16 | **IEEE TCyb** | Edge-guided Recurrent Positioning Network for Salient Object Detection in Optical Remote Sensing Images | [Paper](https://ieeexplore.ieee.org/document/9756846)/[Code](https://github.com/Kunye-Shen/ERPNet) 
17 | **IEEE TCSVT** | Progressive Dual-attention Residual Network for Salient Object Detection | [Paper](https://ieeexplore.ieee.org/abstract/document/9745960)/Code 
18 | **IEEE TCyb** | Global-and-Local Collaborative Learning for Co-Salient Object Detection | [Paper](https://arxiv.org/pdf/2204.08917.pdf)/[Code](https://rmcong.github.io/proj_GLNet.html) 
19 | **ArXiv** | An Energy-Based Prior for Generative Saliency | [Paper](https://arxiv.org/pdf/2204.08803.pdf)/[Code](https://github.com/JingZhang617/EBMGSOD) 
20 | **IEEE TIP** | EDN: Salient Object Detection via Extremely-Downsampled Network | [Paper](https://arxiv.org/pdf/2012.13093.pdf)/[Code](https://github.com/yuhuan-wu/EDN) 
21 | **IEEE TPAMI** | Salient Object Detection via Integrity Learning | [Paper](https://arxiv.org/pdf/2101.07663.pdf)/[Code](https://github.com/mczhuge/ICON) 
22 | **IEEE TCSVT** | TCNet:Co-salient Object Detection via Parallel Interaction of Transformers and CNNs | [Paper](https://ieeexplore.ieee.org/document/9968016)/[Code](https://github.com/zhangqiao970914/TCNet)   
23 | **ArXiv** | Activation to Saliency: Forming High-Quality Labels for Unsupervised Salient Object Detection | [Paper](https://arxiv.org/abs/2112.03650)/[Code](https://github.com/moothes/A2S-USOD)  
24 | **CVPR** | Zoom In and Out: A Mixed-scale Triplet Network for Camouflaged Object Detection | [Paper](https://arxiv.org/pdf/2203.02688.pdf)/[Code](https://github.com/lartpang/ZoomNet)  
25 | **CVPR** | Can You Spot the Chameleon? Adversarially Camouflaging Images from Co-Salient Object Detection | [Paper](https://arxiv.org/pdf/2009.09258.pdf)/[Code](https://github.com/tsingqguo/jadena) 
26 | **CVPR** | Democracy Does Matter: Comprehensive Feature Mining for Co-salient Object Detection | [Paper](https://arxiv.org/pdf/2203.05787.pdf)/[Code](https://github.com/siyueyu/DCFM) 
27 | **CVPR** | Pyramid Grafting Network for One-Stage High Resolution Saliency Detection | [Paper](https://arxiv.org/pdf/2204.05041.pdf)/[Code](https://github.com/iCVTEAM/PGNet) 
28 | **CVPR** | Deep Saliency Prior for Reducing Visual Distraction | [Paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Aberman_Deep_Saliency_Prior_for_Reducing_Visual_Distraction_CVPR_2022_paper.pdf)/[Code](https://deep-saliency-prior.github.io/) 
29 | **CVPR** | Multi-Source Uncertainty Mining for Deep Unsupervised Saliency Detection | [Paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Wang_Multi-Source_Uncertainty_Mining_for_Deep_Unsupervised_Saliency_Detection_CVPR_2022_paper.pdf)/[Code](https://github.com/yifanw90/UMNet)   
30 | **CVPR** | Bi-Directional Object-Context Prioritization Learning for Saliency Ranking | [Paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Tian_Bi-Directional_Object-Context_Prioritization_Learning_for_Saliency_Ranking_CVPR_2022_paper.pdf)/[Code](https://github.com/GrassBro/OCOR) 
31 | **CVPR** | Does text attract attention on e-commerce images: A novel saliency prediction dataset and method | [Paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Jiang_Does_Text_Attract_Attention_on_E-Commerce_Images_A_Novel_Saliency_CVPR_2022_paper.pdf)/[Code](https://github.com/leafy-lee/E-commercial-dataset)  
32 | **CVPRW** | Pyramidal Attention for Saliency Detection | [Paper](https://arxiv.org/pdf/2204.06788.pdf)/[Code](https://github.com/tanveer-hussain) 
33 | **CVPRW** | Unsupervised Salient Object Detection with Spectral Cluster Voting | [Paper](https://openaccess.thecvf.com/content/CVPR2022W/L3D-IVU/papers/Shin_Unsupervised_Salient_Object_Detection_With_Spectral_Cluster_Voting_CVPRW_2022_paper.pdf)/[Code](https://github.com/NoelShin/selfmask) 
34 | **ECCV** | KD-SCFNet: Towards More Accurate and Efficient Salient Object Detection via Knowledge Distillation | [Paper](https://arxiv.org/pdf/2208.02178.pdf)/[Code](https://github.com/zhangjinCV/KD-SCFNet) 
35 | **ECCV** | Salient Object Detection for Point Clouds | [Paper](https://arxiv.org/pdf/2207.11889.pdf)/[Code](https://git.openi.org.cn/OpenPointCloud/PCSOD) 
36 | **PR** | BiconNet: An Edge-preserved Connectivity-based Approach for Salient Object Detection | [Paper](https://arxiv.org/pdf/2103.00334.pdf)/[Code](https://github.com/Zyun-Y/BiconNets) 
37 | **IEEE TCyb** | DNA: Deeply-supervised Nonlinear Aggregation for Salient Object Detection | [Paper](https://ieeexplore.ieee.org/document/9345433)/[Code](https://github.com/yun-liu/DNA) 
38 | **ACMM** | Synthetic Data Supervised Salient Object Detection | [Paper](http://www.digitalimaginggroup.ca/members/Shuo/ACM_Multimedia_2022_final_version.pdf)/[Code](https://github.com/wuzhenyubuaa/SODGAN) 
39 | **IEEE TCSVT** | A Weakly Supervised Learning Framework for Salient Object Detection via Hybrid Labels | [Paper](https://arxiv.org/pdf/2209.02957.pdf)/[Code](https://rmcong.github.io/proj_Hybrid-Label-SOD.html) 
40 | **CVPRW** | Unsupervised Salient Object Detection with Spectral Cluster Voting | [Paper](https://arxiv.org/pdf/2203.12614.pdf)/[Code](https://github.com/NoelShin/selfmask) 
41 | **IEEE TMM** | View-aware Salient Object Detection for 360° Omnidirectional Image | [Paper](https://arxiv.org/pdf/2209.13222.pdf)/[Code](https://github.com/JanySunny/ODI-SOD) 
42 | **ACCV** | Revisiting Image Pyramid Structure for High Resolution Salient Object Detection | [Paper](https://arxiv.org/abs/2209.09475)/[Code](https://github.com/plemeri/InSPyReNet) 
43 | **ECCV** | Saliency Hierarchy Modeling via Generative Kernels for Salient Object Detection | [Paper](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136880564.pdf)/Code
44 | **IEEE TIP** | Salient Object Detection via Dynamic Scale Routing | [Paper](https://arxiv.org/pdf/2210.13821.pdf)/[Code](https://github.com/wuzhenyubuaa/DPNet)
45 | **NeurIPS** | MOVE: Unsupervised Movable Object Segmentation and Detection | [Paper](https://arxiv.org/pdf/2210.07920.pdf)/Code
46 | **PR** | BiconNet: An Edge-preserved Connectivity-based Approach for Salient Object Detection | [Paper](https://arxiv.org/abs/2103.00334)/[Code](https://github.com/Zyun-Y/BiconNets)  

  



## 2021       
**No.** | **Pub.** | **Title** | **Links** 
:-: | :-: | :-  | :-: 
01 | **AAAI** | Structure-Consistent Weakly Supervised Salient Object Detection with Local Saliency Coherence | [Paper](https://arxiv.org/pdf/2012.04404.pdf)/[Code](https://github.com/siyueyu/SCWSSOD/tree/f8650567cbbc8df5bf6edc32a633c47a885574cd)
02 | **AAAI** | Pyramidal Feature Shrinking for Salient Object Detection | [Paper](https://www.aaai.org/AAAI21Papers/AAAI-1322.MaM.pdf)/[Code](https://github.com/iCVTEAM/PFSNet) 
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
23 | **ICCV** | Visual Saliency Transformer | [Paper](https://arxiv.org/pdf/2104.12099.pdf)/[Code](https://github.com/nnizhang/VST#visual-saliency-transformer-vst) 
24 | **ICCV** | Disentangled High Quality Salient Object Detection | [Paper](https://arxiv.org/pdf/2108.03551.pdf)/[Code](https://github.com/luckybird1994/HQSOD) 
25 | **ICCV** | iNAS: Integral NAS for Device-Aware Salient Object Detection | [Paper](https://mftp.mmcheng.net/Papers/21ICCV-iNAS.pdf)/[Code](https://mmcheng.net/inas/) 
26 | **ICCV** | Scene Context-Aware Salient Object Detection | [Paper](https://openaccess.thecvf.com/content/ICCV2021/papers/Siris_Scene_Context-Aware_Salient_Object_Detection_ICCV_2021_paper.pdf)/[Code](https://github.com/SirisAvishek/Scene_Context_Aware_Saliency) 
27 | **ICCV** | MFNet: Multi-Filter Directive Network for Weakly Supervised Salient Object Detection | [Paper](https://openaccess.thecvf.com/content/ICCV2021/papers/Piao_MFNet_Multi-Filter_Directive_Network_for_Weakly_Supervised_Salient_Object_Detection_ICCV_2021_paper.pdf)/[Code](https://github.com/OIPLab-DUT/MFNet) 
28 | **ICCV** | Salient Object Ranking with Position-Preserved Attention | [Paper](https://openaccess.thecvf.com/content/ICCV2021/papers/Fang_Salient_Object_Ranking_With_Position-Preserved_Attention_ICCV_2021_paper.pdf)/[Code](https://github.com/EricFH/SOR) 
29 | **ICCV** | Summarize and Search: Learning Consensus-aware Dynamic Convolution for Co-Saliency Detection | [Paper](https://openaccess.thecvf.com/content/ICCV2021/papers/Zhang_Summarize_and_Search_Learning_Consensus-Aware_Dynamic_Convolution_for_Co-Saliency_Detection_ICCV_2021_paper.pdf)/[Code](https://github.com/nnizhang/CADC) 
30 | **IEEE TIP** | Salient Object Detection with Purificatory Mechanism and Structural Similarity Loss | [Paper](https://arxiv.org/pdf/1912.08393.pdf)/[Code](https://github.com/Jinming-Su/PurNet) 
31 | **ACMM** | Complementary Trilateral Decoder for Fast and Accurate Salient Object Detection | [Paper](https://dl.acm.org/doi/pdf/10.1145/3474085.3475494)/[Code](https://github.com/zhaozhirui/CTDNet)
32 | **NeurIPS** | Learning Generative Vision Transformer with Energy-Based Latent Space for Saliency Prediction | [Paper](https://proceedings.neurips.cc/paper/2021/file/8289889263db4a40463e3f358bb7c7a1-Paper.pdf)/[Code](https://github.com/JingZhang617/EBMGSOD)   
33 | **NeurIPS** | Discovering Dynamic Salient Regions for Spatio-Temporal Graph Neural Networks | [Paper](https://proceedings.neurips.cc/paper/2021/file/398410ece9d7343091093a2a7f8ee381-Paper.pdf)/[Code](https://github.com/bit-ml/DyReg-GNN) 
34 | **IEEE TIP** | Progressive Self-Guided Loss for Salient Object Detection | [Paper](https://arxiv.org/pdf/2101.02412.pdf)/[Code](https://github.com/ysyscool/PSGLoss) 
35 | **IEEE TMM** | Dense Attention-guided Cascaded Network for Salient Object Detection of Strip Steel Surface Defects | [Paper](https://ieeexplore.ieee.org/document/9632537)/[Code](https://github.com/zxforchid/DACNet) 
36 | **IEEE TIP** | Rethinking the U-Shape Structure for Salient Object Detection | [Paper](https://mftp.mmcheng.net/Papers/21TIP-CII.pdf)/[Code](https://github.com/zal0302/CII) 




## 2020       
**No.** | **Pub.** | **Title** | **Links** 
:-: | :-: | :-  | :-: 
01 | **AAAI** | Progressive Feature Polishing Network for Salient Object Detection | [Paper](https://arxiv.org/pdf/1911.05942.pdf)/[Code](https://github.com/chenquan-cq/PFPN)       
02 | **AAAI** | Global Context-Aware Progressive Aggregation Network for Salient Object Detection | [Paper](https://github.com/JosephChenHub/GCPANet/blob/master/GCPANet.pdf)/[Code](https://github.com/JosephChenHub/GCPANet)     
03 | **AAAI** | F3Net: Fusion, Feedback and Focus for Salient Object Detection | [Paper](https://arxiv.org/pdf/1911.11445.pdf)/[Code](https://github.com/weijun88/F3Net)    
04 | **AAAI** | Multi-spectral Salient Object Detection by Adversarial Domain Adaptation | [Paper](https://cse.sc.edu/~songwang/document/aaai20b.pdf)/[Code](https://tsllb.github.io/MultiSOD.html) 
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
27 | **ICCV** | Employing Deep Part-Object Relationships for Salient Object Detection | [Paper](http://openaccess.thecvf.com/content_ICCV_2019/papers/Liu_Employing_Deep_Part-Object_Relationships_for_Salient_Object_Detection_ICCV_2019_paper.pdf)/[Code](https://github.com/liuyi1989/TSPOANet)  
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


<a name="3DSOD"></a>
# 3D RGB-D/T Saliency Detection <a id="3D RGB-D Saliency Detection" class="anchor" href="3D RGB-D Saliency Detection" aria-hidden="true"><span class="octicon octicon-link"></span></a> 

## 2024       
**No.** | **Pub.** | **Title** | **Links** 
:-: | :-: | :-  | :-: 
:triangular_flag_on_post: 01 | **ICASSP** | A Saliency Enhanced Feature Fusion based multiscale RGB-D Salient Object Detection Network | [Paper](https://arxiv.org/pdf/2401.11914.pdf)/Code  
:triangular_flag_on_post: 02 | **IJCV** | Cross-Modal Fusion and Progressive Decoding Network for RGB-D Salient Object Detection | [Paper](https://link.springer.com/article/10.1007/s11263-024-02020-y)/[Code](https://github.com/hu-xh/CPNet)  
:triangular_flag_on_post: 03 | **TMM** | UniTR: A Unified TRansformer-based Framework for Co-object and Multi-modal Saliency Detection | [Paper](https://ieeexplore.ieee.org/abstract/document/10444934)/[Code](https://github.com/ruohaoguo/UniTR) 
:triangular_flag_on_post: 04 | **IEEE TIP** | Quality-aware Selective Fusion Network for V-D-T Salient Object Detection | [Paper](https://arxiv.org/pdf/2405.07655)/[Code](https://github.com/Lx-Bao/QSFNet) 
:triangular_flag_on_post: 05 | **IEEE TCSVT** | Learning Adaptive Fusion Bank for Multi-modal Salient Object Detection | [Paper](https://arxiv.org/pdf/2406.01127)/[Code](https://github.com/Angknpng/LAFB) 
:triangular_flag_on_post: 06 | **IEEE TMM** | Alignment-Free RGBT Salient Object Detection: Semantics-guided Asymmetric Correlation Network and A Unified Benchmark | [Paper](https://arxiv.org/pdf/2406.00917)/[Code](https://github.com/Angknpng/SACNet) 
:triangular_flag_on_post: 07 | **ACMMM** | Backdoor Attacks on Bimodal Salient Object Detection with RGB-Thermal Data | [Paper](https://openreview.net/pdf?id=fBeeQlkIM8)/Code
:triangular_flag_on_post: 08 | **ECCV** | CoLA: Conditional Dropout and Language-driven Robust Dual-modal Salient Object Detection | [Paper](https://arxiv.org/pdf/2407.06780)/[Code](https://github.com/ssecv/CoLA)


## 2023      
**No.** | **Pub.** | **Title** | **Links** 
:-: | :-: | :-  | :-: 
01 | **TCSVT** | HRTransNet: HRFormer-Driven Two-Modality Salient Object Detection | [Paper](https://arxiv.org/pdf/2301.03036.pdf)/[Code](https://github.com/liuzywen/HRTransNet)  
02 | **IEEE TIP** | CAVER: Cross-Modal View-Mixed Transformer for Bi-Modal Salient Object Detection | [Paper](https://ieeexplore.ieee.org/abstract/document/10015667)/[Code](https://github.com/lartpang/CAVER) 
03 | **IEEE TIP** | LSNet: Lightweight Spatial Boosting Network for Detecting Salient Objects in RGB-Thermal Images | [Paper](https://ieeexplore.ieee.org/abstract/document/10042233)/[Code](https://github.com/zyrant/LSNet) 
04 | **ICME** | Scribble-Supervised RGB-T Salient Object Detection | [Paper](https://arxiv.org/pdf/2303.09733.pdf)/[Code](https://github.com/liuzywen/RGBTScribble-ICME2023) 
05 | **TCSVT** | Mutual Information Regularization for Weakly-supervised RGB-D Salient Object Detection | [Paper](https://arxiv.org/pdf/2306.03630.pdf)/[Code](https://github.com/baneitixiaomai/MIRV) 
06 | **Information Fusion** | An Interactively Reinforced Paradigm for Joint Infrared-Visible Image Fusion and Saliency Object Detection | [Paper](https://arxiv.org/abs/2305.09999)/[Code](https://github.com/wdhudiekou/IRFS) 
07 | **IEEE TMM** | CATNet: A Cascaded and Aggregated Transformer Network For RGB-D Salient Object Detection | [Paper](https://ieeexplore.ieee.org/abstract/document/10179145)/[Code](https://github.com/ROC-Star/CATNet/) 
:triangular_flag_on_post: 08 | **ACM MM** | Point-aware Interaction and CNN-induced Refinement Network for RGB-D Salient Object Detection | [Paper](https://arxiv.org/pdf/2308.08930.pdf)/[Code](https://github.com/rmcong/PICR-Net_ACMMM23) 
:triangular_flag_on_post: 09 | **IEEE TIP** | Depth Injection Framework for RGBD Salient Object Detection | [Paper](https://ieeexplore.ieee.org/document/10258039)/[Code](https://github.com/Zakeiswo/DIF) 
:triangular_flag_on_post: 10 | **ACM MM** | Modality Profile - A New Critical Aspect to be Considered When Generating RGB-D Salient Object Detection Training Set | [Paper](https://dl.acm.org/doi/pdf/10.1145/3581783.3611985)/[Code](https://github.com/XueHaoWang-Beijing/ModalityProfile_MM23/)
:triangular_flag_on_post: 11 | **ACM MM** | Saliency Prototype for RGB-D and RGB-T Salient Object Detection | [Paper](https://dl.acm.org/doi/pdf/10.1145/3581783.3612466)/[Code](https://github.com/ZZ2490/SPNet)  
:triangular_flag_on_post: 12 | **NeurIPS** | DVSOD: RGB-D Video Salient Object Detection | [Paper](https://openreview.net/pdf?id=Hm1Ih3uLII)/[Code](https://github.com/DVSOD/DVSOD-Baseline)  


## 2022       
**No.** | **Pub.** | **Title** | **Links** 
:-: | :-: | :-  | :-: 
01 | **CVMJ** | Specificity-preserving RGB-D Saliency Detection | [Paper](https://arxiv.org/abs/2108.08162)/[Code](https://github.com/taozh2017/SPNet?utm_source=catalyzex.com)   
02 | **AAAI** | Self-Supervised Pretraining for RGB-D Salient Object Detection | [Paper](https://www.aaai.org/AAAI22Papers/AAAI-4882.ZhaoX.pdf)/[Code](https://github.com/Xiaoqi-Zhao-DLUT/SSLSOD)   
03 | **IEEE TPAMI** | MobileSal: Extremely Efficient RGB-D Salient Object Detection | [Paper](https://mftp.mmcheng.net/Papers/21PAMI_MobileSal.pdf)/[Code](https://mmcheng.net/mobilesal/)   
04 | **IEEE TIP** | Boosting RGB-D Saliency Detection by Leveraging Unlabeled RGB Images | [Paper](https://arxiv.org/pdf/2201.00100.pdf)/[Code](https://github.com/Robert-xiaoqiang/DS-Net)   
05 | **IEEE TIP** | Learning Discriminative Cross-modality Features for RGB-D Saliency Detection | [Paper](https://ieeexplore.ieee.org/abstract/document/9678058)/Code  
06 | **IEEE TIP** | Weakly Supervised RGB-D Salient Object Detection with Prediction Consistency Training and Active Scribble Boosting | [Paper](https://ieeexplore.ieee.org/document/9720104)/[Code](https://github.com/XuYunqiu/scribbleRGB-DSOD)
07 | **ICLR** | Promoting Saliency From Depth: Deep Unsupervised RGB-D Saliency Detection | [Paper](https://openreview.net/pdf?id=BZnnMbt0pW)/[Code](https://github.com/jiwei0921/DSU)  
08 | **ArXiv** | DFTR: Depth-supervised Hierarchical Feature Fusion Transformer for Salient Object Detection | [Paper](https://arxiv.org/pdf/2203.06429.pdf)/Code  
09 | **ArXiv** | GroupTransNet: Group Transformer Network for RGB-D Salient Object Detection | [Paper](https://arxiv.org/pdf/2203.10785.pdf)/Code  
10 | **ArXiv** | CAVER: Cross-Modal View-Mixed Transformer for Bi-Modal Salient Object Detection | [Paper](https://arxiv.org/pdf/2112.02363.pdf)/[Code](https://github.com/lartpang/CAVER)
11 | **PR** | Encoder Deep Interleaved Network with Multi-scale Aggregation for RGB-D Salient Object Detection | [Paper](https://www.sciencedirect.com/science/article/abs/pii/S0031320322001479)/Code  
12 | **CVPRW** | Pyramidal Attention for Saliency Detection | [Paper](https://arxiv.org/pdf/2204.06788.pdf)/[Code](https://github.com/tanveer-hussain) 
13 | **TMM** | Depth-induced Gap-reducing Network for RGB-D Salient Object Detection: An Interaction, Guidance and Refinement Approach | [Paper](https://ieeexplore.ieee.org/document/9769984)/[Code](https://github.com/ssecv/DIGR-Net) 
14 | **TMM** | C2DFNet: Criss-Cross Dynamic Filter Network for RGB-D Salient Object Detection | [Paper](https://ieeexplore.ieee.org/document/9813422)/[Code](https://github.com/OIPLab-DUT/C2DFNet) 
15 | **ArXiv** | Dual Swin-Transformer based Mutual Interactive Network for RGB-D Salient Object Detection | [Paper](https://arxiv.org/pdf/2206.03105.pdf)/Code  
16 | **IEEE TCSVT** | Cross-Collaborative Fusion-Encoder Network for Robust RGB-Thermal Salient Object Detection | [Paper](https://ieeexplore.ieee.org/document/9801871)/[Code](https://github.com/gbliao/CCFENet)  
17 | **IEEE TIP** | Learning Implicit Class Knowledge for RGB-D Co-Salient Object Detection with Transformers | [Paper](https://ieeexplore.ieee.org/document/9810116)/[Code](https://github.com/nnizhang/CTNet)  
18 | **ACMM** | Depth-inspired Label Mining for Unsupervised RGB-D Salient Object Detection | [Paper](https://dl.acm.org/doi/pdf/10.1145/3503161.3548037?casa_token=9IfKDOr4970AAAAA:yWl9tbPTwlCtnXJE7-Vuj7rHxxBPi39zLVoeb1rgFwZEDVNdeK3Y8SYO0gkyT98kCKd2nhtI1Et2190)/[Code](https://github.com/youngtboy/DLM)  
19 | **3DV** | Robust RGB-D Fusion for Saliency Detection | [Paper](https://arxiv.org/pdf/2208.01762.pdf)/[Code](https://github.com/Zongwei97/RFnet)  
20 | **ArXiv** | Depth Quality-Inspired Feature Manipulation for Efficient RGB-D and Video Salient Object Detection | [Paper](https://arxiv.org/pdf/2208.03918.pdf)/[Code](https://github.com/zwbx/DFM-Net)  
21 | **ECCV** | SPSN: Superpixel Prototype Sampling Network for RGB-D Salient Object Detection | [Paper](https://arxiv.org/pdf/2207.07898.pdf)/[Code](https://github.com/Hydragon516/SPSN)  
22 | **ECCV** | MVSalNet:Multi-View Augmentation for RGB-D Salient Object Detection | [Paper](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136890268.pdf)/[Code](https://github.com/Heart-eartH/MVSalNet)  
23 | **IJCV** | Learnable Depth-Sensitive Attention for Deep RGB-D Saliency Detection with Multi-modal Fusion Architecture Search | [Paper](https://link.springer.com/article/10.1007/s11263-022-01646-0)/[Code](https://github.com/sunpeng1996/DSA2F)   
24 | **IEEE TNNLS** | 3-D Convolutional Neural Networks for RGB-D Salient Object Detection and Beyond | [Paper](https://ieeexplore.ieee.org/document/9889257)/[Code](https://github.com/QianChen98/RD3D)   
25 | **IEEE TIP** | Improving RGB-D Salient Object Detection via Modality-aware Decoder | [Paper](https://ieeexplore.ieee.org/abstract/document/9894275?casa_token=x6Stwtpf_igAAAAA:_ivL1dWDAHq29mTPgl4ctDVhwf6qbonXaQZ5t1PFqGwvDzVk4w28lEbwVt-9yQJ15C4zuI7TaFQ)/[Code](https://github.com/MengkeSong/MaD)   
26 | **IEEE TIP** | CIR-Net: Cross-modality interaction and refinement for RGB-D salient object detection | [Paper](https://arxiv.org/abs/2210.02843)/[Code](https://github.com/rmcong/CIRNet_TIP2022)   
27 | **IEEE TCSVT** | HRTransNet: HRFormer-Driven Two-Modality Salient Object Detection | [Paper](https://ieeexplore.ieee.org/abstract/document/9869666?casa_token=tYGCtPgo5kkAAAAA:WWYviL3djEpBBRvds_DtYaAfdqnV5Qvdq7DaS4b6Dk9lQc9beLj4hQ9T8fLNpYeU9ku71v96abg)/[Code](https://github.com/liuzywen/HRTransNet) 
28 | **IEEE TMM** | Does Thermal Really Always Matter for RGB-T Salient Object Detection? | [Paper](https://arxiv.org/pdf/2210.04266.pdf)/[Code](https://rmcong.github.io/proj_TNet.html) 
29 | **IEEE TCSVT** | Modality-Induced Transfer-Fusion Network for RGB-D and RGB-T Salient Object Detection | [Paper](https://ieeexplore.ieee.org/abstract/document/9925217?casa_token=gFFqPMx0N7sAAAAA:1DpXKX-b2jvTF1Zwcf-gtJkyj0ZW-lxbRcJb60rO0BiLFJqTbpg7Sl0VGhe2Ku62Rqtg2AfFyfY)/Code  
30 | **IEEE TIP** | Joint Learning of Salient Object Detection, Depth Estimation and Contour Extraction | [Paper](https://arxiv.org/pdf/2203.04895.pdf)/[Code](https://github.com/Xiaoqi-Zhao-DLUT/MMFT) 
31 | **IJCV** | Delving into Calibrated Depth for Accurate RGB-D Salient Object Detection | [Paper](https://link.springer.com/article/10.1007/s11263-022-01734-1)/[Code](https://github.com/jiwei0921/HiBo-UA) 
32 | **IEEE TCSVT** | MoADNet: Mobile Asymmetric Dual-Stream Networks for Real-Time and Lightweight RGB-D Salient Object Detection | [Paper](https://ieeexplore.ieee.org/document/9789193)/[Code](https://github.com/kingkung2016/MoADNet) 




## 2021       
**No.** | **Pub.** | **Title** | **Links** 
:-: | :-: | :-  | :-: 
01 | **CVPR** | Deep RGB-D Saliency Detection with Depth-Sensitive Attention and Automatic Multi-Modal Fusion | [Paper](https://arxiv.org/pdf/2103.11832.pdf)/[Code](https://github.com/sunpeng1996/DSA2F)   
02 | **CVPR** | Calibrated RGB-D Saliency Object Detection | [Paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Ji_Calibrated_RGB-D_Salient_Object_Detection_CVPR_2021_paper.pdf)/[Code](https://github.com/jiwei0921/DCF)  
03 | **AAAI** | RGB-D Salient Object Detection via 3D Convolutional Neural Networks | [Paper](https://arxiv.org/pdf/2101.10241.pdf)/[Code](https://github.com/PPOLYpubki/RD3D)
04 | **IEEE TIP** | Hierarchical Alternate Interaction Network for RGB-D Salient Object Detection | [Paper](https://ieeexplore.ieee.org/document/9371407)/[Code](https://github.com/MathLee/HAINet)
05 | **IEEE TIP** | CDNet: Complementary Depth Network for RGB-D Salient Object Detection | [Paper](https://ieeexplore.ieee.org/abstract/document/9366409)/[Code](https://github.com/blanclist/CDNet)
06 | **IEEE TIP** | RGB-D Salient Object Detection with Ubiquitous Target Awareness | [Paper](https://arxiv.org/pdf/2109.03425.pdf)/[Code](https://github.com/iCVTEAM/UTA)
07 | **ICME** | BTS-Net: Bi-directional Transfer-and-Selection Network for RGB-D Salient Object Detection | [Paper](https://arxiv.org/pdf/2104.01784.pdf)/[Code](https://github.com/zwbx/BTS-Net)
08 | **ACMM** | Depth Quality-Inspired Feature Manipulation for Efficient RGB-D Salient Object Detection | [Paper](https://arxiv.org/pdf/2107.01779.pdf)/[Code](https://github.com/zwbx/DFM-Net)  
09 | **ACMM** | TriTransNet RGB-D Salient Object Detection with a Triplet Transformer Embedding Network | [Paper](https://arxiv.org/pdf/2108.03990.pdf)/[Code](https://github.com/liuzywen/TriTransNet-RGB-D-Salient-Object-Detection-with-a-Triplet-Transformer-Embedding-Network)
10 | **ICCV** | RGB-D Saliency Detection via Cascaded Mutual Information Minimization | [Paper](https://arxiv.org/pdf/2109.07246.pdf)/[Code](https://github.com/JingZhang617/cascaded_rgbd_sod)
11 | **ICCV** | Specificity-preserving RGB-D Saliency Detection | [Paper](https://arxiv.org/pdf/2108.08162.pdf)/[Code](https://github.com/taozh2017/SPNet)
12 | **ACMM** | Cross-modality Discrepant Interaction Network for RGB-D Salient Object Detection | [Paper](https://arxiv.org/pdf/2108.01971.pdf)/[Code](https://github.com/1437539743/CDINet-ACM-MM21)
13 | **IEEE TIP** | Dynamic Selective Network for RGB-D Salient Object Detection | [Paper](https://ieeexplore.ieee.org/document/9605221/authors#authors)/[Code](https://github.com/Brook-Wen/DSNet)
14 | **IJCV** | CNN-based RGB-D Salient Object Detection: Learn, Select and Fuse | [Paper](https://link.springer.com/article/10.1007/s11263-021-01452-0)/Code
15 | **NeurIPS** | Joint Semantic Mining for Weakly Supervised RGB-D Salient Object Detection | [Paper](https://proceedings.neurips.cc/paper/2021/file/642e92efb79421734881b53e1e1b18b6-Paper.pdf)/[Code](https://github.com/jiwei0921/JSM)   
16 | **IEEE TMM** | CCAFNet: Crossflow and Cross-scale Adaptive Fusion Network for Detecting Salient Objects in RGB-D Images | [Paper](https://ieeexplore.ieee.org/document/9424966)/[Code](https://github.com/zyrant/CCAFNet)   
17 | **IEEE TETCI** | APNet: Adversarial-Learning-Assistance and Perceived Importance Fusion Network for All-Day RGB-T Salient Object Detection | [Paper](https://ieeexplore.ieee.org/document/9583676)/[Code](https://github.com/zyrant/APNet)   




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
06 | **IEEE TIP** | RGB-T Salient Object Detection via Fusing Multi-Level CNN Features | [Paper](https://ieeexplore.ieee.org/abstract/document/8935533)/[Code](https://github.com/nexiakele/RGB-T-Salient-Object-Detection-via-Fusing-Multi-level-CNN-Features)
07 | **IEEE TMM** | RGB-T image saliency detection via collaborative graph learning | [Paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8744296)/Code

   

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



<a name="4DSOD"></a>
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
08 | **IEEE TCSVT** | A Multi-Task Collaborative Network for Light Field Salient Object Detection | [Paper](https://ieeexplore.ieee.org/abstract/document/9153018)/[Code](https://github.com/zhangqiudan/MTCNet-Lightfield)  
09 | **ArXiv** | DUT-LFSaliency: Versatile Dataset and Light Field-to-RGB Saliency Detection | [Paper](https://arxiv.org/pdf/2012.15124.pdf)/[Code](https://github.com/OIPLab-DUT/DUTLF-V2)   
10 | **ArXiv** | Learning Synergistic Attention for Light Field Salient Object Detection | [Paper](https://arxiv.org/pdf/2104.13916.pdf)/Code
11 | **ArXiv** | CMA-Net: A Cascaded Mutual Attention Network for Light Field Salient Object Detection | [Paper](https://arxiv.org/pdf/2105.00949.pdf)/Code
12 | **IEEE TCyB** | PANet: Patch-Aware Network for Light Field Salient Object Detection | [Paper](https://ieeexplore.ieee.org/abstract/document/9517032)/[Code](https://github.com/jyydlut/IEEE-TCYB-PANet)
13 | **ACMM21** | Occlusion-aware Bi-directional Guided Network for Light Field Salient Object Detection | [Paper](https://dl.acm.org/doi/pdf/10.1145/3474085.3475312?casa_token=wbPMsKJlIUgAAAAA:YVsFNQb65PB4D6FGlMwtYtYi5nR4YCE1tJw_7frdEMm_exQIDw5dFzjIW0AjmwqlO1XEOEbz-g)/[Code](https://github.com/Timsty1/OBGNet)
14 | **ICCV21** | Light Field Saliency Detection with Dual Local Graph Learning and Reciprocative Guidance | [Paper](https://arxiv.org/pdf/2110.00698.pdf)/[Code](https://github.com/wangbo-zhao/2021ICCV-DLGLRG)
15| **CVPR22** | Learning from Pixel-Level Noisy Label : A New Perspective for Light Field Saliency Detection | [Paper](https://arxiv.org/pdf/2204.13456.pdf)/[Code](https://github.com/OLobbCode/NoiseLF)  
16| **NC** | MEANet: Multi-modal edge-aware network for light field salient object detection | [Paper](https://www.sciencedirect.com/science/article/abs/pii/S0925231222003502)/[Code](https://github.com/jiangyao-scu/MEANet)
17| **IEEE TIP** | Exploring Spatial Correlation for Light Field Saliency Detection: Expansion from a Single View | [Paper](https://ieeexplore.ieee.org/abstract/document/9894273?casa_token=1mIHAJs5QB4AAAAA:vvhqsmbsJWjL9qGTjvOUWngBkgn9BJGkPY6M91tm2Tp-mhswCbmhtIU7cr5R6qT4vCqsU9L57kw)/[Code]()
18| **IEEE TIP** | Geometry Auxiliary Salient ObjectDetection for Light Fields via Graph Neural Networks | [Paper](https://ieeexplore.ieee.org/document/9527158)/[Code](https://github.com/zhangqiudan/GeoSOD-Lightfield)  
19| **ACMM** | LFBCNet: Light Field Boundary-aware and Cascaded Interaction Network for Salient Object Detection | [Paper](https://dl.acm.org/doi/pdf/10.1145/3503161.3548275?casa_token=ifuWtYwl-roAAAAA:aSGUDEbp5YTrX7fxS0r7gEWq_kYKhOFom0VQ_6topWxvgArBopbmlvcAn7kXkjpo6jf9LEWX4vgivgU)/Code  
20| **IEEE TIP** | Weakly-Supervised Salient Object Detection on Light Fields | [Paper](https://ieeexplore.ieee.org/document/9900489/authors#authors)/Code
21| **IEEE TPAMI** | A Thorough Benchmark and a New Model for Light Field Saliency Detection | [Paper](https://www.computer.org/csdl/journal/tp/5555/01/10012539/1JNmt6JGKu4)/[Code](https://openi.pcl.ac.cn/OpenDatasets)  
22| **ICME23** | Guided Focal Stack Refinement Network for Light Field Salient Object Detection | [Paper](https://arxiv.org/pdf/2305.05260.pdf)/Code  
:triangular_flag_on_post: 23| **IEEE TCSVT** | LFTransNet: Light Field Salient Object Detection via a Learnable Weight Descriptor | [Paper](https://ieeexplore.ieee.org/abstract/document/10138590?casa_token=rJeI2PnLzwQAAAAA:nnJc89z7hCRfJH3C-GtVjybe1HL11dZVoWOxzZ45d4Jn623BW4ZM9bS8DdyBiuvW-2zeyW7fdYJgkQ)/[Code](https://github.com/liuzywen/LFTransNet)  
:triangular_flag_on_post: 24| **IEEE TCSVT** | Light Field Salient Object Detection with Sparse Views via Complementary and Discriminative Interaction Network | [Paper](https://ieeexplore.ieee.org/document/10168184)/[Code](https://github.com/GilbertRC/LFSOD-CDINet)  
:triangular_flag_on_post: 25| **arXiv** | LF Tracy: A Unified Single-Pipeline Approach for Salient Object Detection in Light Field Cameras | [Paper](https://browse.arxiv.org/abs/2401.16712)/[Code](https://github.com/FeiBryantkit/LF-Tracy)  


<a name="VSOD"></a>      
# Video Salient Object Detection  <a id="Video Salient Object Detection" class="anchor" href="Video Salient Object Detection" aria-hidden="true"><span class="octicon octicon-link"></span></a> 

## 2024  
**No.** | **Pub.** | **Title** | **Links** 
:-: | :-: | :-  | :-: 
:triangular_flag_on_post: 01 | **AAAI** | A Motion-aware Spatio-temporal Graph for Video Salient Object Ranking | [Paper](https://openreview.net/pdf?id=VUBtAcQN44)/[Code](https://github.com/zyf-815/VSOR/tree/main)  

## 2023  
**No.** | **Pub.** | **Title** | **Links** 
:-: | :-: | :-  | :-: 
01 | **AAAI** | Panoramic Video Salient Object Detection with Ambisonic Audio Guidance | [Paper](https://arxiv.org/pdf/2211.14419.pdf)/Code  

## 2022  
**No.** | **Pub.** | **Title** | **Links** 
:-: | :-: | :-  | :-: 
01 | **AAAI** | You Only Infer Once: Cross-Modal Meta-Transfer for Referring Video Object Segmentation | [Paper](https://www.aaai.org/AAAI22Papers/AAAI-1100.LiD.pdf)/[Code](https://github.com/Sparklins/YOFO)   
02 | **AAAI** | Siamese Network with Interactive Transformer for Video Object Segmentation | [Paper](https://www.aaai.org/AAAI22Papers/AAAI-702.LanM.pdf)/[Code](https://github.com/LANMNG/SITVOS)   
03 | **AAAI** | Iteratively Selecting an Easy Reference Frame Makes Unsupervised Video Object Segmentation Easier | [Paper](https://www.aaai.org/AAAI22Papers/AAAI-11964.LeeY.pdf)/Code   
04 | **AAAI** | Reliable Propagation-Correction Modulation for Video Object Segmentation | [Paper](https://www.aaai.org/AAAI22Papers/AAAI-4288.XuX.pdf)/[Code](https://github.com/JerryX1110/RPCMVOS)     
05 | **WACV** | Video Salient Object Detection via Contrastive Features and Attention Modules | [Paper](https://arxiv.org/pdf/2111.02368.pdf)/Code  
06 | **ICIP** | Depth-Cooperated Trimodal Network for Video Salient Object Detection | [Paper](https://arxiv.org/pdf/2202.06060.pdf)/[Code](https://github.com/luyukang/DCTNet)  
07 | **ArXiv** | Learning Video Salient Object Detection Progressively from Unlabeled Videos | [Paper](https://arxiv.org/abs/2204.02008)/Code  
08 | **ArXiv** | Rethinking Video Salient Object Ranking | [Paper](https://arxiv.org/abs/2203.17257)/Code  
09 | **ACMM** | Weakly Supervised Video Salient Object Detection via Point Supervision | [Paper](https://arxiv.org/pdf/2207.07269.pdf)/Code  
10 | **ECCV** | Hierarchical Feature Alignment Network for Unsupervised Video Object Segmentation | [Paper](https://arxiv.org/abs/2207.08485)/[Code](https://github.com/NUST-Machine-Intelligence-Laboratory/HFAN)  
11 | **ECCV** | XMem: Long-Term Video Object Segmentation with an Atkinson-Shiffrin Memory Model | [Paper](https://arxiv.org/pdf/2207.07115.pdf)/[Code](https://hkchengrex.github.io/XMem/)  
12 | **ACMM** | Bidirectionally Learning Dense Spatio-temporal Feature Propagation Network for Unsupervised Video Object Segmentation | [Paper](https://dl.acm.org/doi/pdf/10.1145/3503161.3548039?casa_token=xbckiU4No2wAAAAA:hpKejtoDLTyeTRtCNao2PHacfpfR7HRV38JOieDNbF-C67SAKaXTTswqs_yC8DDp7at-rUkYyc1N5I0)/Code  
13 | **NeurIPS** | Semi-Supervised Video Salient Object Detection Based on Uncertainty-Guided Pseudo Labels | [Paper](https://openreview.net/pdf?id=BOQr80FBX_)/[Code](https://github.com/Lanezzz/UGPL)   


## 2021  
**No.** | **Pub.** | **Title** | **Links** 
:-: | :-: | :-  | :-: 
01 | **CVPR** | Weakly Supervised Video Salient Object Detection | [Paper](https://arxiv.org/pdf/2104.02391.pdf)/[Code](https://github.com/wangbo-zhao/WSVSOD)     
02 | **ArXiv** | Video Salient Object Detection via Adaptive Local-Global Refinement | [Paper](https://arxiv.org/pdf/2104.14360.pdf)/Code    
03 | **ICIP** | Guidance and Teaching Network for Video Salient Object Detection | [Paper](https://arxiv.org/pdf/2105.10110.pdf)/[Code](https://github.com/GewelsJI/GTNet)    
04 | **ACMM** | Multi-Source Fusion and Automatic Predictor Selection for Zero-Shot Video Object Segmentation | [Paper](https://arxiv.org/pdf/2108.05076.pdf)/[Code](https://github.com/Xiaoqi-Zhao-DLUT/Multi-Source-APS-ZVOS)   
05 | **ICCV** | Dynamic Context-Sensitive Filtering Network for Video Salient Object Detection | [Paper](https://openaccess.thecvf.com/content/ICCV2021/papers/Zhang_Dynamic_Context-Sensitive_Filtering_Network_for_Video_Salient_Object_Detection_ICCV_2021_paper.pdf)/[Code](https://github.com/Roudgers/DCFNet)   
06 | **ICCV** | Full-Duplex Strategy for Video Object Segmentation | [Paper](https://arxiv.org/pdf/2108.03151.pdf)/[Code](https://github.com/GewelsJI/FSNet) 
07 | **ICCV** | Deep Transport Network for Unsupervised Video Object Segmentation | [Paper](https://openaccess.thecvf.com/content/ICCV2021/papers/Zhang_Deep_Transport_Network_for_Unsupervised_Video_Object_Segmentation_ICCV_2021_paper.pdf)/Code
08 | **IEEE TIP** | Exploring Rich and Efficient Spatial Temporal Interactions for Real Time Video Salient Object Detection | [Paper](https://ieeexplore.ieee.org/document/9390381)/[Code](https://github.com/guotaowang/STVS)


## 2020  
**No.** | **Pub.** | **Title** | **Links** 
:-: | :-: | :-  | :-: 
01 | **CVPR** | STAViS: Spatio-Temporal AudioVisual Saliency Network | [Paper](https://openaccess.thecvf.com/content_CVPR_2020/papers/Tsiami_STAViS_Spatio-Temporal_AudioVisual_Saliency_Network_CVPR_2020_paper.pdf)/[Code](https://github.com/atsiami/STAViS)  
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
08 | **IEEE TIP** | Improving Robust Video Saliency Detection based on Long-term Spatial-Temporal Information | [Paper](https://ieeexplore.ieee.org/document/8811767)/[Code](https://github.com/guotaowang/TIP_LSTI)  



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


<a name="survey"></a>  
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


<a name="data"></a>  
# The SOD dataset download    <a id="The SOD dataset download" class="anchor" href="The SOD dataset download" aria-hidden="true"><span class="octicon octicon-link"></span></a> 
* 2D SOD datasets [download1](https://github.com/TinyGrass/SODdataset) or [download2](https://github.com/ArcherFMY/sal_eval_toolbox), [download3](https://github.com/magic428/awesome-segmentation-saliency-dataset).
* 3D SOD datasets [download](https://github.com/jiwei0921/RGBD-SOD-datasets).  
* 4D SOD datasets [download](https://github.com/jiwei0921/MoLF).
* Video SOD datasets [download](http://dpfan.net/DAVSOD/).

<a name="eval"></a>
# Evaluation Metrics  <a id="Evaluation Metrics" class="anchor" href="Evaluation Metrics" aria-hidden="true"><span class="octicon octicon-link"></span></a> 
* Saliency maps evaluation.      
This link near all evaluation metrics for salient object detection including E-measure, S-measure, F-measure, MAE scores and PR curves or bar metrics.
You can found in [here](https://github.com/jiwei0921/Saliency-Evaluation-Toolbox).        

* Saliency Dataset evaluation.       
This repo can compute the ratio of obj.area and obj.contrast on binary saliency dataset. This Toolbox contains two evaluation metrics, including obj(object).area and obj.contrast.     
You can found in [here](https://github.com/jiwei0921/Saliency-dataset-evaluation).      

<a name="leaderboard"></a>
# Comparison with state-of-the-arts  <a id="Comparison with state-of-the-arts" class="anchor" href="Comparison with state-of-the-arts" aria-hidden="true"><span class="octicon octicon-link"></span></a> 
* [Here](https://paperswithcode.com/sota/salient-object-detection-on-duts-te) includes the performance comparison of almost all 2D salient object detection algorithms. 
* [Here](https://paperswithcode.com/task/rgb-d-salient-object-detection) includes the performance comparison of almost all 3D RGB-D salient object detection algorithms. 


### AI Conference Deadlines
[Realted AI Conference deadline](https://aideadlin.es/?sub=ML,CV,NLP,RO,SP,DM)     
[Realted AI Conference Accepted Rate](https://github.com/lixin4ever/Conference-Acceptance-Rate)
