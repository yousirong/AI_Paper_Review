# Paper Reading
AI 논문 정리

</details>


## Diffusion Model Contents
- [Classification](#classification)
- [Resources](#resources)
  - [Introductory Posts](#introductory-posts)
  - [Introductory Papers](#introductory-papers)
  - [Introductory Videos](#introductory-videos)
- [Papers](#papers)
  - [Must-read papers](#must-read-papers)
  - [Personalized](#personalized)
  - [Stable diffusion freeze](#stable-diffusion-freeze)
  - [Stable diffusion finetuning](#stable-diffusion-finetuning)
  - [Connection with other framworks](#connection-with-other-framworks)
  - [Image Generation](#image-generation)
  - [Image space guidance sampling](#image-space-guidance-sampling)
  - [Classifier guidance sampling](#classifier-guidance-sampling)
  - [Image Editing](#image-editing)
  - [Text-focused](#text-focused)
  - [Fast Sampling](#fast-sampling)
  - [Video Generation and Editing](#video-generation-and-editing)
  - [3D](#3d)
  - [수학기반향상](#수학기반향상)
  - [기타](#기타)



---

## Classification
- [Lenet-5(1998)](https://deep-learning-study.tistory.com/368), PyTorch Code [[Google Colab](https://github.com/Seonghoon-Yu/paper-implement-in-pytorch/blob/master/Classification/LeNet_5(1998).ipynb) / [Blog Posting](https://deep-learning-study.tistory.com/503)]

- [AlexNet(2012)](https://deep-learning-study.tistory.com/376), PyTorch Code [[Google Colab](https://github.com/Seonghoon-Yu/paper-implement-in-pytorch/blob/master/Classification/AlexNet(2012).ipynb) / [Blog Posting](https://deep-learning-study.tistory.com/518)]

- [Pseudo Label(2013)](https://deep-learning-study.tistory.com/553)

- PyTorch 구현 코드로 살펴보는 [Knowledge Distillation(2014)](https://deep-learning-study.tistory.com/699), PyTorch Code [[Google Colab](https://github.com/Seonghoon-Yu/Paper_Review_and_Implementation_in_PyTorch/blob/master/Classification/Knowledge_distillation(2014).ipynb) / [Blog Posting](https://deep-learning-study.tistory.com/700)], paper [[pdf](https://arxiv.org/abs/1503.02531)]

- [GoogLeNet(2014)](https://deep-learning-study.tistory.com/389), PyTorch Code [[Google Colab](https://github.com/Seonghoon-Yu/paper-implement-in-pytorch/blob/master/Classification/GoogLeNet(2014).ipynb) / [Blog Posting](https://deep-learning-study.tistory.com/523)]

- [VGGNet(2014)](https://deep-learning-study.tistory.com/398), PyTorch Code [[Google Colab](https://github.com/Seonghoon-Yu/paper-implement-in-pytorch/blob/master/Classification/VGGnet(2014).ipynb) / [Blog Posting](https://deep-learning-study.tistory.com/521)]

- [Inception-v3(2015)](https://deep-learning-study.tistory.com/517)

- [ResNet(2015)](https://deep-learning-study.tistory.com/473), PyTorch Code [[Google Colab](https://github.com/Seonghoon-Yu/paper-implement-in-pytorch/blob/master/Classification/ResNet(2015).ipynb) / [Blog Posting](https://deep-learning-study.tistory.com/534?category=983681)]

- [Pre-Activation ResNet(2016)](https://deep-learning-study.tistory.com/510), PyTorch Code [[Google Colab](https://github.com/Seonghoon-Yu/Paper_Review_and_Implementation_in_PyTorch/blob/master/Classification/PreAct_ResNet(2016).ipynb) / [Blog Posting](https://deep-learning-study.tistory.com/540)]

- [WRN, Wide Residual Networks(2016)](https://deep-learning-study.tistory.com/519), PyTorch Code [[Google Colab](https://github.com/Seonghoon-Yu/Paper_Review_and_Implementation_in_PyTorch/blob/master/Classification/Wide_ResNet(2016).ipynb) / [Blog Posting](https://deep-learning-study.tistory.com/542)]

- [SqueezeNet(2016)](https://deep-learning-study.tistory.com/520)

- [Inception-v4(2016)](https://deep-learning-study.tistory.com/525), PyTorch Code [[Google Colab](https://github.com/Seonghoon-Yu/Paper_Review_and_Implement_in_PyTorch/blob/master/Classification/Inceptionv4(2016).ipynb) / [Blog Posting](https://deep-learning-study.tistory.com/537)]

- [PyramidNet(2017)](https://deep-learning-study.tistory.com/526)

- [DenseNet(2017)](https://deep-learning-study.tistory.com/528), PyTorch Code [[Google Colab](https://github.com/Seonghoon-Yu/Paper_Review_and_Implementation_in_PyTorch/blob/master/Classification/DenseNet(2017).ipynb) / [Blog Posting](https://deep-learning-study.tistory.com/545)]

- [Xception(2017)](https://deep-learning-study.tistory.com/529), PyTorch Code [[Google Colab](https://github.com/Seonghoon-Yu/Paper_Review_and_Implementation_in_PyTorch/blob/master/Classification/Xception(2017).ipynb) / [Blog Posting](https://deep-learning-study.tistory.com/548)]

- [MobileNetV1(2017)](https://deep-learning-study.tistory.com/532), PyTorch Code [[Google Colab](https://github.com/Seonghoon-Yu/Paper_Review_and_Implementation_in_PyTorch/blob/master/Classification/Xception(2017).ipynb) / [Blog Posting](https://deep-learning-study.tistory.com/549)]

- [ResNext(2017)](https://deep-learning-study.tistory.com/533), PyTorch Code [[Google Colab](https://github.com/Seonghoon-Yu/Paper_Review_and_Implementation_in_PyTorch/blob/master/Classification/ResNext(2017).ipynb) / [Blog Posting](https://deep-learning-study.tistory.com/558)]

- [PolyNet(2017)](https://deep-learning-study.tistory.com/535)

- [Residual Attention Network(2017)](https://deep-learning-study.tistory.com/536), PyTorch Code [[Google Colab](https://github.com/Seonghoon-Yu/Paper_Review_and_Implementation_in_PyTorch/blob/master/Classification/Residual_Attention_Network(2017).ipynb) / [Blog Posting](https://deep-learning-study.tistory.com/555)]

- [DPN(2017)](https://deep-learning-study.tistory.com/538)

- [Non-local Neural Network(2017)](https://deep-learning-study.tistory.com/749), paper [[pdf](https://arxiv.org/abs/1711.07971)]

- [SENet(2018)](https://deep-learning-study.tistory.com/539), PyTorch Code [[Google Colab](https://github.com/Seonghoon-Yu/Paper_Review_and_Implementation_in_PyTorch/blob/master/Classification/SENet(2018).ipynb) / [Blog Posting](https://deep-learning-study.tistory.com/561)]

- [MobileNetV2(2018)](https://deep-learning-study.tistory.com/541)

- [ShuffleNet(2018)](https://deep-learning-study.tistory.com/544)

- [NasNet(2018)](https://deep-learning-study.tistory.com/543)

- [PNasNet(2018)](https://deep-learning-study.tistory.com/546)

- [ShuffleNet(2018)](https://deep-learning-study.tistory.com/547)

- [CondenseNet(2018)](https://deep-learning-study.tistory.com/550)

- [CBAM(2018)](https://deep-learning-study.tistory.com/666), paper [[pdf](https://arxiv.org/abs/1807.06521)]

- [Bag of Tricks(2019)](https://deep-learning-study.tistory.com/569)

- [MobileNetV3(2019)](https://deep-learning-study.tistory.com/551)

- [EfficientNet(2019)](https://deep-learning-study.tistory.com/552), PyTorch Code [[Google Colab](https://github.com/Seonghoon-Yu/Paper_Review_and_Implementation_in_PyTorch/blob/master/Classification/EfficientNet(2019).ipynb) / [Blog Posting](https://deep-learning-study.tistory.com/563)]

- [SKNet(2019)](https://deep-learning-study.tistory.com/669), paper [[pdf](https://arxiv.org/abs/1903.06586)]

- [BiT(2019)](https://deep-learning-study.tistory.com/723). paper [[pdf](https://arxiv.org/abs/1912.11370)]

- [Noisy Student(2020)](https://deep-learning-study.tistory.com/554)

- [FixEfficientNet(2020)](https://deep-learning-study.tistory.com/557)

- [Meta Pseudo Labels(2020)](https://deep-learning-study.tistory.com/560)

- [Noise or Signal: The Role of Image Backgrounds in Object Recognition(2020)](https://deep-learning-study.tistory.com/693), paper [[pdf](https://arxiv.org/abs/2006.09994)]

- [VIT(2020)](https://deep-learning-study.tistory.com/716), paper [[pdf](https://arxiv.org/abs/2010.11929)], PyTorch Code [[Google Colab](https://github.com/Seonghoon-Yu/Paper_Review_and_Implementation_in_PyTorch/blob/master/Classification/ViT(2020).ipynb) / [Blog Posting](https://deep-learning-study.tistory.com/807)]

- [Deit(2020)](https://deep-learning-study.tistory.com/806), paper [[pdf](https://arxiv.org/abs/2106.07023)]

- [EfficientNetV2(2021)](https://deep-learning-study.tistory.com/567)

- [Knowledge distillation: A good teacher is patient and consitent(2021)](https://deep-learning-study.tistory.com/701), paper [[pdf](https://arxiv.org/abs/2106.05237)]

- [MLP-Mixer(2021)](https://deep-learning-study.tistory.com/720), paper [[pdf](https://arxiv.org/pdf/2105.01601.pdf)]

- [CvT(2021)](https://deep-learning-study.tistory.com/816), paper [[pdf](https://arxiv.org/abs/2103.15808)]

- [CeiT(2021)](https://deep-learning-study.tistory.com/811), paper [[pdf](https://arxiv.org/abs/2103.11816)]

- [Early Convolutions Help Transformers See Better(2021)](https://deep-learning-study.tistory.com/818), paper [[pdf](https://arxiv.org/abs/2106.14881)]

- [BoTNet(2021)](https://deep-learning-study.tistory.com/821), paper [[pdf](https://arxiv.org/abs/2101.11605)]

- [Conformer(2021)](https://deep-learning-study.tistory.com/852), paper [[pdf](https://arxiv.org/abs/2105.03889)]

- [Delving Deep into the Generalization of Vision Transformers under Distribution Shifts(2021)](https://deep-learning-study.tistory.com/824), paper [[pdf](https://arxiv.org/abs/2106.07617)]

- [Scaling Vision Transformers(2021)](https://deep-learning-study.tistory.com/828), paper [[pdf](https://arxiv.org/abs/2106.04560)]

- [CMT(2021)](https://deep-learning-study.tistory.com/829), paper [[pdf](https://arxiv.org/abs/2107.06263)]

## Object Detection

- [IoU(Intersection over Union)를 이해하고 파이토치로 구현하기](https://deep-learning-study.tistory.com/402)
- [비-최대 억제(NMS, Non-maximum Suppression)를 이해하고 파이토치로 구현하기](https://deep-learning-study.tistory.com/403)
- [mAP(mean Average Precision)을 이해하고 파이토치로 구현하기](https://deep-learning-study.tistory.com/407)

- [R-CNN(2013)](https://deep-learning-study.tistory.com/410)

- [SPPnet(2014)](https://deep-learning-study.tistory.com/445)

- [Fast R-CNN(2014)](https://deep-learning-study.tistory.com/456)

- [Faster R-CNN(2015)](https://deep-learning-study.tistory.com/464) 미완성 [PyTorch Code](https://github.com/Seonghoon-Yu/Paper_Review_and_Implementation_in_PyTorch/blob/master/Object_Detection/Faster_R_CNN(2015)_%EB%AF%B8%EC%99%84%EC%84%B1.ipynb)

- [SSD(2016)](https://deep-learning-study.tistory.com/477)

- [YOLO v1(2016)](https://deep-learning-study.tistory.com/430)

- [R-FCN(2016)](https://deep-learning-study.tistory.com/570)

- [OHEM(2016)](https://deep-learning-study.tistory.com/501)

- [DSSD(2017)](https://deep-learning-study.tistory.com/566)

- [YOLO v2(2017)](https://deep-learning-study.tistory.com/433)

- [FPN(2017)](https://deep-learning-study.tistory.com/491)

- [RetinaNet(2017)](https://deep-learning-study.tistory.com/504) PyTorch Code [[Google Colab](https://github.com/Seonghoon-Yu/Paper_Review_and_Implementation_in_PyTorch/blob/master/RetinaNet(2017).ipynb) / [Blog Posting](https://deep-learning-study.tistory.com/616)]
- [RON(2017)](https://deep-learning-study.tistory.com/572)

- [DCN(2017)](https://deep-learning-study.tistory.com/575)

- [CoupleNet(2017)](https://deep-learning-study.tistory.com/602)

- [Soft-NMS(2017)](https://deep-learning-study.tistory.com/606)

- [RefineDet(2018)](https://deep-learning-study.tistory.com/609)

- [Cascade R-CNN(2018)](https://deep-learning-study.tistory.com/605)

- [YOLO v3(2018)](https://deep-learning-study.tistory.com/509), PyTorch Code [[Google Colab](https://github.com/Seonghoon-Yu/Paper_Review_and_Implementation_in_PyTorch/blob/master/Object_Detection/YOLOv3(2018).ipynb) / [Blog Posting](https://deep-learning-study.tistory.com/568)]

- [CornerNet(2018)](https://deep-learning-study.tistory.com/613)

- [M2Det(2019)](https://deep-learning-study.tistory.com/620)

- [CenterNet(2019)](https://deep-learning-study.tistory.com/622), paper [[pdf](https://arxiv.org/abs/1904.08189)]

- [Gaussian YOLOv3(2019)](https://deep-learning-study.tistory.com/624), paper [[pdf](https://arxiv.org/pdf/1904.04620)]

- [FCOS(2019)](https://deep-learning-study.tistory.com/625), paper [[pdf](https://arxiv.org/pdf/1904.01355.pdf)]

- [YOLOv4(2020)](https://deep-learning-study.tistory.com/626), paper [[pdf](https://arxiv.org/abs/2004.10934)]

- [EfficientDet(2020)](https://deep-learning-study.tistory.com/627), paper [[pdf](https://arxiv.org/abs/1911.09070)] 

- [CSPNet(2020)](https://deep-learning-study.tistory.com/632), paper [[pdf](https://arxiv.org/abs/1911.11929)]

- [DIoU Loss(2020)](https://deep-learning-study.tistory.com/634), paper [[pdf](https://arxiv.org/abs/1911.08287)], [Code](https://github.com/Seonghoon-Yu/Paper_Review_and_Implementation_in_PyTorch/blob/master/Object_Detection/CIoU_Loss.py)

- [CircleNet(2020)](https://deep-learning-study.tistory.com/661), paper [[pdf](https://arxiv.org/pdf/2006.02474.pdf)]

- [DETR(2020)](https://deep-learning-study.tistory.com/748), paper [[pdf](https://arxiv.org/abs/2005.12872)]

- [ACT(2020)](https://deep-learning-study.tistory.com/789), paper [[pdf](https://arxiv.org/abs/2011.09315)]

- [Deformable DETR(2020)](https://deep-learning-study.tistory.com/825), paper [[pdf](https://arxiv.org/abs/2010.04159)]

- Localization Distillation for Dense Object Detection(2102)

- [CenterNet2(2021)](https://deep-learning-study.tistory.com/670), paper [[pdf](https://arxiv.org/abs/2103.07461)]

- [Swin Transformer(2021)](https://deep-learning-study.tistory.com/728), paper [[pdf](https://arxiv.org/pdf/2103.14030v1.pdf)]

- [YOLOr(2021)](https://deep-learning-study.tistory.com/739), paper [[pdf](https://arxiv.org/pdf/2105.04206v1.pdf)]

- [YOLOS(2021)](https://deep-learning-study.tistory.com/826), paper [[pdf](https://arxiv.org/abs/2106.00666)]

- Dynamic Head, Unifying Object Detection Heads with Attention(2021), paper [[pdf](https://arxiv.org/abs/2106.08322)]

- [Pix2Seq(2021)](https://deep-learning-study.tistory.com/866), paper [[pdf](https://arxiv.org/abs/2109.10852)]

- Anchor DETR, Query Design for Transformer-Based Object Detection(2021), paper [[pdf](https://arxiv.org/abs/2109.07107)]

- DAB-DETR, Dynamic Anchor Boxes are Better Queries for DETR(2022), paper [[pdf](https://arxiv.org/abs/2201.12329)]

- DN-DETR, Accelerate DETR Training by Introducing Query DeNoising(2022), paper [[pdf](https://arxiv.org/abs/2203.01305)]

- DINO, DETR with Imporved DeNoising Anchor Boxes for End-to-End Object Detection(2022), paper [[pdf](https://arxiv.org/abs/2203.03605)]



## Segmentation

- [DeepLabV1(2014)](https://deep-learning-study.tistory.com/564)

- [FCN(2015)](https://deep-learning-study.tistory.com/562)

- [DeConvNet(2015)](https://deep-learning-study.tistory.com/565)

- [DilatedNet(2015)](https://deep-learning-study.tistory.com/664), paper [[pdf](https://arxiv.org/abs/1511.07122)]

- PyTorch 구현 코드로 살펴보는 [SegNet(2015)](https://deep-learning-study.tistory.com/672), paper [[pdf](https://arxiv.org/pdf/1511.00561.pdf)]

- [PSPNet(2016)](https://deep-learning-study.tistory.com/864), paper [[pdf](https://arxiv.org/abs/1612.01105)]

- [DeepLabv3(2017)](https://deep-learning-study.tistory.com/877), paper [[pdf](https://arxiv.org/abs/1706.05587)]

- [Mask R-CNN(2017)](https://deep-learning-study.tistory.com/571)

- [PANet(2018)](https://deep-learning-study.tistory.com/637), paper [[pdf](https://arxiv.org/abs/1803.01534)]

- [Panoptic Segmentation(2018)](https://deep-learning-study.tistory.com/861), paper [[pdf](https://arxiv.org/abs/1801.00868)]

- Weakly- and Semi-Supervised Panoptic Segmentation(2018), paper [[pdf](https://arxiv.org/abs/1808.03575)]

- [Panoptic Segmentation with a Joint Semantic and Instance Segmentation Network(2018)](https://deep-learning-study.tistory.com/862), paper [[pdf](https://arxiv.org/abs/1809.02110)]

- [Single Network Panoptic Segmentation for Street Scene Understanding(2019)](https://deep-learning-study.tistory.com/863), paper [[pdf](https://arxiv.org/abs/1902.02678)]

- [Panoptic Feature Pyramid Networks(2019)](https://deep-learning-study.tistory.com/867), paper [[pdf](https://arxiv.org/abs/1901.02446)]

- [IMP: Instance Mask Projection for High Accuracy Semantic Segmentation of Things(2019)](https://deep-learning-study.tistory.com/865), paper [[pdf](https://arxiv.org/abs/1906.06597)]

- [Object-Contextual Representations for Semantic Segmentation(2019)](https://deep-learning-study.tistory.com/894), paper [[pdf](https://arxiv.org/abs/1909.11065)]

- [CondInst, Conditional Convolution for Instance Segmentation(2020)](https://deep-learning-study.tistory.com/961), paper [[pdf](https://arxiv.org/abs/2003.05664)]

- Max-DeepLab, End-to-End Panoptic Segmentation wtih Mask Transformers, paper [[pdf](https://arxiv.org/abs/2012.00759)]

- [MaskFormer, Per-Pixel Classification is Not All You Need for Semantic Segmentation(2021)](https://deep-learning-study.tistory.com/940), paper [[pdf](https://arxiv.org/abs/2107.06278)]

- [Open-World Entity Segmentation(2021)](https://deep-learning-study.tistory.com/962), paper [[pdf](https://arxiv.org/abs/2107.14228)]

- Prompt based Multi-modal Image Segmentation(2021), paper [[pdf](https://arxiv.org/abs/2112.10003)]

- DenseCLIP, Language-Guided Dense Prediction with Context-Aware Prompting, paper [[pdf](https://arxiv.org/abs/2112.10003)]

- [Mask2Former, Masked-attention Mask Transformer for Universal Image Segmentation(2021)](https://arxiv.org/abs/2112.01527)

- [SeMask<, Semantically Masked Transformers for Semantic Segmentation(2021)](https://arxiv.org/abs/2112.12782)



## Self-supervised Learning
- [Constrative Loss(2006)](https://deep-learning-study.tistory.com/724), paper [[pdf](http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf)]

- [Exemplar-CNN(2014)](https://deep-learning-study.tistory.com/715), paper [[pdf](https://arxiv.org/abs/1406.6909)]

- [Unsupervised Learning of Visual Representation using Videos](https://deep-learning-study.tistory.com/773), paper [[pdf](https://arxiv.org/abs/1505.00687)]

- [Context Prediction(2015)](https://deep-learning-study.tistory.com/717), paper [[pdf](https://arxiv.org/abs/1505.05192)]

- [Jigsaw Puzzles(2016)](https://deep-learning-study.tistory.com/719), paper [[odf](https://arxiv.org/pdf/1603.09246.pdf)]

- [Colorful Image Coloriztion(2016)](https://deep-learning-study.tistory.com/722), paper [[pdf](https://arxiv.org/abs/1603.08511)]

- [Deep InfoMax(2018)](https://deep-learning-study.tistory.com/768), paper [[pdf](https://arxiv.org/abs/1808.06670)]

- [Deep Cluster(2018)](https://deep-learning-study.tistory.com/766), paper [[pdf](https://arxiv.org/abs/1807.05520)]

- [IIC(2018)](https://deep-learning-study.tistory.com/784), paper [[pdf](https://arxiv.org/abs/1807.06653)]

- [Rotation(2018)](https://deep-learning-study.tistory.com/804), paper [[pdf](https://arxiv.org/abs/1803.07728)]

- [Unsupervised Feature Learning via Non-Parametric Instance Discrimination(2018)](https://deep-learning-study.tistory.com/769), paper [[pdf](https://arxiv.org/abs/1805.01978)]

- [ADMIN(2019)](https://deep-learning-study.tistory.com/817), paper [[pdf](https://arxiv.org/abs/1906.00910)]

- [Contrastive Multiview Coding(2019)](https://deep-learning-study.tistory.com/814), paper [[pdf](https://deep-learning-study.tistory.com/814)]

- [MoCo(2019)](https://deep-learning-study.tistory.com/730), paper [[pdf](https://arxiv.org/abs/1911.05722)]

- [SeLa(2019)](https://deep-learning-study.tistory.com/760), paper [[pdf](https://arxiv.org/abs/1911.05371)]

- [SimCLR(2020)](https://deep-learning-study.tistory.com/731), paper [[pdf](https://arxiv.org/abs/2002.05709)]

- [MoCov2(2020)](https://deep-learning-study.tistory.com/743), PyTorch Code [[Google Colab](https://github.com/Seonghoon-Yu/MoCov2_Pytorch_tutorial/blob/main/MoCov2.ipynb) / [Blog Posting](https://deep-learning-study.tistory.com/744)], paper [[pdf](https://arxiv.org/pdf/2003.04297.pdf)]

- [SimSiam(2020)](https://deep-learning-study.tistory.com/745), paper [[pdf](https://arxiv.org/pdf/2011.10566.pdf)]

- [Understanding the Behaviour of Contrastive Loss(2020)](https://deep-learning-study.tistory.com/753), paper [[pdf](https://arxiv.org/pdf/2012.09740.pdf)]

- [BYOL(2020)](https://deep-learning-study.tistory.com/753), paper [[pdf](https://arxiv.org/abs/2012.09740)]

- [SwAV(2020)](https://deep-learning-study.tistory.com/761), paper [[pdf](https://arxiv.org/abs/2006.09882)]

- [PCL(2020)](https://deep-learning-study.tistory.com/822), paper [[pdf](https://arxiv.org/abs/2005.04966)]

- [SimCLRv2(2020)](https://deep-learning-study.tistory.com/778), paper [[pdf](https://arxiv.org/abs/2006.10029)]

- [Supervised Contrastive Learning(2020)](https://deep-learning-study.tistory.com/819), paper [[pdf](https://arxiv.org/abs/2004.11362)]

- [DenseCL(2020), Dense Contrastive Learning for Self-Supervised Visual Pre-Training](https://deep-learning-study.tistory.com/935), paper [[pdf](https://arxiv.org/abs/2011.09157)]

- [DetCo(2021)](https://deep-learning-study.tistory.com/843), paper [[pdf](https://arxiv.org/abs/2102.04803)

- [SCRL(2021)](https://deep-learning-study.tistory.com/831), paper [[pdf](https://arxiv.org/abs/2103.06122)]

- [MoCov3(2021)](https://deep-learning-study.tistory.com/746), paper [[pdf](https://arxiv.org/abs/2104.02057)]

- [DINO(2021)](https://deep-learning-study.tistory.com/827), paper [[pdf](https://arxiv.org/abs/2104.14294)]

- [EsViT(2021)](https://deep-learning-study.tistory.com/845), paper [[pdf](https://arxiv.org/abs/2106.09785)]

- [Masked Autoencoders Are Scalable Vision Learners(2021)](https://deep-learning-study.tistory.com/907), paper [[pdf](https://arxiv.org/abs/2111.06377)]



## Video SSL

- [Tracking Emerges by Colorizing Videos(2018)](https://deep-learning-study.tistory.com/830), paper [[pdf](https://arxiv.org/pdf/1806.09594.pdf)]

- [Self-supervised Learning for Video Correspondence Flow(2019)](https://deep-learning-study.tistory.com/832), paper [[pdf](https://arxiv.org/abs/1905.00875)]


- [Learning Correspondence from the Cycle-consistency of Time(2019)](https://deep-learning-study.tistory.com/833), paper [[pdf](https://arxiv.org/abs/1903.07593)]

- [Joint-task Self-supervised Learning for Temporal Correspondence(2019)](https://deep-learning-study.tistory.com/835), paper [[pdf](https://arxiv.org/abs/1909.11895)]

- [MAST(2020)](https://deep-learning-study.tistory.com/836), https://arxiv.org/abs/2002.07793

- [Space-Time Correspondence as a Contrastive Random Walk(2020)](https://deep-learning-study.tistory.com/839), paper [[pdf](https://arxiv.org/abs/2006.14613)]

- [Contrastive Transformation for Self-supervised Correspondence Learning(2020)](https://deep-learning-study.tistory.com/837), paper [[pdf](https://arxiv.org/abs/2012.05057)]

- [Mining Better Samples for Contrastive Learning of Temporal Correspondence(2021)](https://deep-learning-study.tistory.com/840), paper [[pdf](https://openaccess.thecvf.com/content/CVPR2021/html/Jeon_Mining_Better_Samples_for_Contrastive_Learning_of_Temporal_Correspondence_CVPR_2021_paper.html)]

- [VFS(2021)](https://deep-learning-study.tistory.com/841), paper [[pdf](https://arxiv.org/abs/2103.17263)]

- [Contrastive Learning of Image Representations with Cross-Video Cycle-Consistency](https://deep-learning-study.tistory.com/842), paper [[pdf](https://arxiv.org/abs/2105.06463)]

- [ViCC(2021)](https://deep-learning-study.tistory.com/847), paper [[pdf](https://arxiv.org/abs/2106.10137)]


## Semi-supervised Learning

- [Temporal ensembling for semi-supervised learning(2016)](https://deep-learning-study.tistory.com/757) , paper [[pdf](https://arxiv.org/abs/1610.02242)]

- [Mean teachers are better role models(2017)](https://deep-learning-study.tistory.com/758), paper [[pdf](https://arxiv.org/abs/1703.01780)]

- [Consistency-based Semi-supervised Learning for Object Detection(2019)](https://deep-learning-study.tistory.com/735), paper [[pdf](https://papers.nips.cc/paper/2019/hash/d0f4dae80c3d0277922f8371d5827292-Abstract.html)]

- [PseudoSeg, Designing Pseudo Labels for Semantic Segmentation(2020)](https://deep-learning-study.tistory.com/953), paper [[pdf](https://arxiv.org/abs/2010.09713)]

- [ReCo, Bootstrapping Semantic Segmentation with Regional Contrast(2021)](https://deep-learning-study.tistory.com/868), paper [[pdf](https://arxiv.org/abs/2104.04465)]

- [Semi-Supervised Semantic Segmentation with Cross Pseudo Supervision(2021)](https://deep-learning-study.tistory.com/948), paper [[pdf](https://arxiv.org/abs/2106.01226)]

- [Soft Teacher(2021), End-to-End Semi-Supervised Object Detection with Soft Teacher](https://deep-learning-study.tistory.com/949), paper [[pdf](https://arxiv.org/abs/2106.09018)]

- [CaSP(2021), Class-agnostic Semi-Supervised Pretraining for Detection & Segmentation](https://deep-learning-study.tistory.com/960), paper [[pdf](https://arxiv.org/abs/2112.04966)]


## weakly
- [Class Activation Map(CAM), Learning Deep Features for Discriminative Localization](https://deep-learning-study.tistory.com/954), paper [[pdf](https://arxiv.org/abs/1512.04150)]

- [Grad-CAM, Visual Explanations from Deep Networks via Gradient based Localization](https://deep-learning-study.tistory.com/955), paper [[pdf](https://arxiv.org/abs/1610.02391)]

- [Zoom-CAM, Generating Fine-grained Pixel Annotations from Image Labels(2020)](https://deep-learning-study.tistory.com/956), paper [[pdf](https://arxiv.org/abs/2010.08644)]

- [GETAM: Gradient-weighted Element-wise Transformer Attention Map for Weakly-supervised Semantic Segmentation(2021)](https://deep-learning-study.tistory.com/958), paper [[pdf](https://arxiv.org/abs/2112.02841)]



## Video Recognition
- [Learning Spatiotemporal Features with 3D Convolutional Network(2014)](https://deep-learning-study.tistory.com/751), paper [[pdf](https://arxiv.org/abs/1412.0767)]

- [Quo Vadis, Action Recognition? A New Model and the Kinetics Dataset(2017)](https://deep-learning-study.tistory.com/756), paper [[pdf](https://arxiv.org/abs/1705.07750)]

- [SlowFast Networks for Video Recognition(2018)](https://deep-learning-study.tistory.com/765), paper [[pdf](https://arxiv.org/abs/1812.03982)]

- [GCNet(2019)](https://deep-learning-study.tistory.com/780), paper [[pdf](https://arxiv.org/abs/1904.11492)]

- [Drop an Octave(2019)](https://deep-learning-study.tistory.com/788), paper [[pdf](https://arxiv.org/abs/1904.05049)]

- [STM(2019)](https://deep-learning-study.tistory.com/800), paper [[pdf](https://arxiv.org/abs/1908.02486)]

- [X3D(2020)](https://deep-learning-study.tistory.com/855), paper [[pdf](https://arxiv.org/abs/2004.04730)]

- [VTN(2021)](https://deep-learning-study.tistory.com/850). paper [[pdf](https://arxiv.org/abs/2102.00719)]


## Video Recognition - Two Stream CNNs
- [TSN(2016)](https://github.com/open-mmlab/mmaction2/blob/main/configs/recognition/tsn/README.md), paper[[pdf](https://arxiv.org/abs/1608.00859)]

- [STA-CNN: Convolutional Spatial-Temporal Attention Learning for Action Recognition](https://ieeexplore.ieee.org/abstract/document/9058999?casa_token=t5IrgS4Ik0cAAAAA:Zb0vp9DXrDPRGfB4C9T-3S9K65IA6KdO7s94Sf-ycEfd2p4iTvqpWsS-qh0UModvS7SkW_C9vg)
  
- [MS-TCN++: Multi-Stage Temporal Convolutional Network for Action Segmentation](https://ieeexplore.ieee.org/abstract/document/9186840?casa_token=Yu7SyGrRwF4AAAAA:yhrFDLKvSATn_ah1uduRh8EqOPn-2PrLqkej5WO9vHPzacXE5FfoA68HZWVEFS-d8LC8W9MgKA)

- [STA-TSN: Spatial-Temporal Attention Temporal Segment Network for action recognition in video](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0265115)

## Video Recognition - Frame-based Models 

- [TSM: Temporal Shift Module for Efficient Video Understanding(2018)](https://www.notion.so/Temporal-Shift-Module-TSM-80418b91bde04305837e08e430b8e2bd), paper [[pdf](https://arxiv.org/abs/1811.08383)]

- [TRN: Temporal Relational Reasoning in Videos(2018)](https://www.notion.so/Temporal-Relation-Network-TRN-eea3f311b48b4aeea73d37d0e3f8f795), paper [[pdf](https://arxiv.org/abs/1711.08496)]

- [TDN: Temporal Difference Network(2021)](https://www.notion.so/Temporal-Difference-Network-TDN-58709b9838b34b68984457b3bee067cf), paper [[pdf](https://arxiv.org/abs/2012.10071)]



## Video Recognition - Transformer

- [TimeSformer(2021)](https://deep-learning-study.tistory.com/848), paper [[pdf](https://arxiv.org/abs/2102.05095)], Youtube [[link](https://youtu.be/xSf40PZjTxQ)]

- [ViViT(2021)](https://deep-learning-study.tistory.com/838), paper [[pdf](https://arxiv.org/abs/2103.15691)]

- [MViT(2021)](https://deep-learning-study.tistory.com/849), paper [[pdf](https://arxiv.org/abs/2104.11227)] 

- [X-ViT(2021)](https://deep-learning-study.tistory.com/856), paper [[pdf](https://arxiv.org/abs/2106.05968)]

- [Video Swin Transformer(2021)](https://deep-learning-study.tistory.com/846), paper [[pdf](https://arxiv.org/abs/2106.13230)]

- [Towards Training Stronger Video Vision Transformers for EPIC-KITCHENS-100 Action Recognition(2021)](https://deep-learning-study.tistory.com/859), paper [[pdf](https://arxiv.org/abs/2106.05058)]

- [VLF(2021)](https://deep-learning-study.tistory.com/857), paper [[pdf](https://arxiv.org/abs/2107.00451)], Youtube [[link](https://youtu.be/OtVHC1s3yzg)]



## Video Segmentation

- [VisTR(2020)](https://deep-learning-study.tistory.com/834), paper [[pdf](https://arxiv.org/abs/2011.14503)]

## Zero Shot Classification
- [DeViSE, A Deep Visual-Semantic Embedding Model(2013)](https://deep-learning-study.tistory.com/909), paper [[pdf](https://papers.nips.cc/paper/2013/hash/7cce53cf90577442771720a370c3c723-Abstract.html)]

- [Zero-shot Learning via Shared-Reconstruction-Graph Pursuit(2017)](https://deep-learning-study.tistory.com/910), paper [[pdf](https://arxiv.org/abs/1711.07302)]

- [A Generative Adversarial Approach for Zero-Shot Learning from Noisy Texts(2017)](https://deep-learning-study.tistory.com/903), paper [[pdf](https://arxiv.org/abs/1712.01381)]

- [f-VAEGAN-D2, A Feature Generating Framework for Any Shot Learning(2019)](https://deep-learning-study.tistory.com/944), paper [[pdf](https://arxiv.org/abs/1903.10132)]

- [TCN(2019), Transferable Contrastive Network for Generalized Zero-Shot Learning](https://deep-learning-study.tistory.com/927), paper [[pdf](https://arxiv.org/abs/1908.05832)]

- [Rethinking Zero-Shot Learning: A Conditional Visual Classification Perspective(2019)](https://deep-learning-study.tistory.com/928), paper [[pdf](https://arxiv.org/abs/1909.05995)]

- [Convolutional Prototype Learning for Zero-Shot Recognition(2019)](https://deep-learning-study.tistory.com/931), paper [[pdf](https://arxiv.org/abs/1910.09728)]


- [DRN, Class-Prototype Discriminative Network for Generalized Zero-Shot Learning(2020)](https://deep-learning-study.tistory.com/930), paper [[pdf](https://ieeexplore.ieee.org/abstract/document/8966463)]

- [DAZLE(2020), Fine-Grained Generalized Zero-Shot Learning via Dense Attribute-Based Attention](https://deep-learning-study.tistory.com/915), paper [[pdf](https://openaccess.thecvf.com/content_CVPR_2020/papers/Huynh_Fine-Grained_Generalized_Zero-Shot_Learning_via_Dense_Attribute-Based_Attention_CVPR_2020_paper.pdf)]

- [IPN(2021), Isometric Propagation Network for Generalized Zero-Shot Learning](https://deep-learning-study.tistory.com/932), paper [[pdf](https://arxiv.org/abs/2102.02038)]


- [CE-GZSL(2021), Contrastive Embedding for Generalized Zero-Shot Learning](https://deep-learning-study.tistory.com/926), paper [[pdf](https://arxiv.org/abs/2103.16173)]

- [Task-Independent Knowledge Makes for Transferable Represenatations for Generalized Zero-Shot Learning(2021)](https://deep-learning-study.tistory.com/934), paper [[pdf](https://arxiv.org/abs/2104.01832)]

- [Zero-Shot Learning via Contrastive Learning on Dual Knowledge Graphs(2021)](https://deep-learning-study.tistory.com/933), paper [[pdf](https://ieeexplore.ieee.org/document/9607851)]

- [FREE: Feature Refinement for Generalized Zero-Shot Learning(2021)](https://deep-learning-study.tistory.com/936), paper [[pdf](https://arxiv.org/abs/2107.13807)]

- [ALIGN(2021), Scaling Up Visual and Vision-Language Representation Learning with Noisy Text Supervision](https://deep-learning-study.tistory.com/916), paper [[pdf](https://arxiv.org/abs/2102.05918)]

- [LiT: Zero-Shot Transfer with Locked-image Text Tuning(2021)](https://deep-learning-study.tistory.com/911), paper [[pdf](https://arxiv.org/abs/2111.07991)]

- [Generalized Category Discovery(2022)](https://deep-learning-study.tistory.com/945), paper [[pdf](https://arxiv.org/abs/2201.02609)]

## Zero Shot Detection

- Synthesizing the Unseen for Zero-shot Object Detection(2020), paper [[pdf](https://arxiv.org/abs/2010.09425)] 

- [ViLD(2021), Open-Vocabulary Object Detection via Vision and Language Knowledge Distillation](https://deep-learning-study.tistory.com/947), paper [[pdf](https://arxiv.org/abs/2104.13921)]

- Robust Region Feature Synthesizer for Zero-Shot Object Detection(2022), paper [[pdf](https://arxiv.org/abs/2201.00103)]

- [Detic(2022), Detecting Twenty-thousand Classes using Image-level Supervision](https://deep-learning-study.tistory.com/957), paper [[pdf](https://arxiv.org/abs/2201.02605)]


## Zero Shot Segmentation

- [Zero-Shot Semantic Segmentation(2019)](https://deep-learning-study.tistory.com/872), paper [[pdf](https://arxiv.org/abs/1906.00817)]

- [Semantic Projection Network for Zero- and Few-Label Semantic Segmentation(2020)](https://deep-learning-study.tistory.com/876), paper [[pdf](https://openaccess.thecvf.com/content_CVPR_2019/papers/Xian_Semantic_Projection_Network_for_Zero-_and_Few-Label_Semantic_Segmentation_CVPR_2019_paper.pdf)]

- [Learning unbiased zero-shot semantic segmentation networks via transductive transfer(2020)](https://deep-learning-study.tistory.com/891), paper [[pdf](https://arxiv.org/abs/2007.00515)]

- [A review of Generalized Zero-Shot Learning Methods(2020)](https://deep-learning-study.tistory.com/873), paper [[pdf](https://arxiv.org/abs/2011.08641)]

- [Consistent Structural Relation Learning for Zero-Shot Segmentation(2020](https://deep-learning-study.tistory.com/885), paper [[pdf](https://proceedings.neurips.cc/paper/2020/hash/7504adad8bb96320eb3afdd4df6e1f60-Abstract.html)]

- [Uncertainty-Aware Learning for Zero-Shot Semantic Segmentation(2020)](https://deep-learning-study.tistory.com/884), paper [[pdf](https://proceedings.neurips.cc/paper/2020/hash/f73b76ce8949fe29bf2a537cfa420e8f-Abstract.html)]

- [Context-aware Feature Generation for Zero-shot Semantic Segmentation(2020)](https://deep-learning-study.tistory.com/874), paper [[pdf](https://arxiv.org/abs/2008.06893)]

- [Recursive Training for Zero-Shot Semantic Segmentation(2021)](https://deep-learning-study.tistory.com/889), paper [[pdf](https://arxiv.org/abs/2103.00086)]

- [Zero-Shot Instance Segmentation(2021)](https://deep-learning-study.tistory.com/892), paper [[pdf](https://arxiv.org/abs/2104.06601)]

- [A Closer Look at Self-training for Zero-Label Segmantic Segmentation(2021)](https://deep-learning-study.tistory.com/883), paper [[pdf](https://arxiv.org/abs/2104.11692)]

- [Prototypical Matching and Open Seg Rejection for Zero-Shot Semantic Segmentation(2021)](https://deep-learning-study.tistory.com/929), paper [[pdf](https://openaccess.thecvf.com/content/ICCV2021/papers/Zhang_Prototypical_Matching_and_Open_Set_Rejection_for_Zero-Shot_Semantic_Segmentation_ICCV_2021_paper)]

- [SIGN(2021), Spatial-information Incorporated Generative Network for GGeneralized Zero-shot Semantic Segmentation](https://deep-learning-study.tistory.com/875), paper [[pdf](https://arxiv.org/abs/2108.12517)]

- [Exploiting a Joint Embedding Space for Generalized Zero-Shot Semantic Segmentation(2021)](https://deep-learning-study.tistory.com/879), paper [[pdf](https://arxiv.org/abs/2108.06536)]

- [Languager Driven Semantic Segmentation(2021)](https://deep-learning-study.tistory.com/890), paper [[pdf](https://openreview.net/forum?id=RriDjddCLN)]

- Zero-Shot Semantic Segmentation via Spatial and Multi-Scale Aware Visual Class Embedding, paper [[pdf](https://www.semanticscholar.org/paper/Zero-Shot-Semantic-Segmentation-via-Spatial-and-Cha-Wang/9d3d4d413125bb27681117b947320717d8deadfe)]

- [DenseCLIP: Extract Free Dence Labels from CLIP(2021)](https://deep-learning-study.tistory.com/946), paper [[pdf](https://arxiv.org/abs/2112.01071)]

- [Decoupling Zero-Shot Semantic Segmentation(2021)](https://deep-learning-study.tistory.com/943), paper [[pdf](https://arxiv.org/abs/2112.07910)]

- [A Simple Baseline for Zero-Shot Semantic Segmentation with Pre-trained Vision-language Model(2021)](https://deep-learning-study.tistory.com/939), paper [[pdf](https://arxiv.org/pdf/2112.14757.pdf)]

- [Open-Vocabulary Image Segmentation(2021)](https://deep-learning-study.tistory.com/942), paper [[pdf](https://arxiv.org/abs/2112.12143)]


## Few-Shot, Meta Learning
- [Matching Networks for One Shot Learning(2016)](https://deep-learning-study.tistory.com/941), paper [[pdf](https://arxiv.org/abs/1606.04080)]

- [Learning to Compare: Relation Network for Few-Shot Learning(2017)](https://deep-learning-study.tistory.com/937), paper [[pdf](https://arxiv.org/abs/1711.06025)]


## Prompting and Vision-Language Model

cv

- [CPT, Colorful Prompt Tuning for Pre-trained Vision-Language Models](https://arxiv.org/abs/2109.11797)

- [CoOp(2021), Learning to Prompt for Vision-Language Models](https://arxiv.org/abs/2109.01134)

- [CLIP-Adapter, Better Vision-Language Models with Feature Adapters(2021)](https://arxiv.org/abs/2110.04544)

- [Tip-Adapter, Training-free CLIP-Adapter for Better Vision-Language Modeling](https://arxiv.org/abs/2111.03930)

- [Prompt-Based Multi-Model Image Segmentation(2021)](https://arxiv.org/abs/2112.10003)

- [DenseCLIP, Language-Guided Dense Prediction with Context-Aware Prompting(2021)](https://arxiv.org/abs/2112.01518)

- Prompting Visual-Language Models for Efficient Video Understanding, paper [[pdf](https://arxiv.org/abs/2112.04478)]

- Conditianl Prompt Learning for Visiona-Language Models, paper [[pdf](https://arxiv.org/abs/2203.05557)]

nlp

- [Pre-train, Prompt, and Predict: A Systematic Survey of Prompting Methods in Natural Language Processing(2021)](https://arxiv.org/abs/2107.13586)




## Image Processing
- PyTorch 구현 코드로 살펴보는 [SRCNNe(2014)](https://deep-learning-study.tistory.com/687), PyTorch Code [[Google Colab](https://github.com/Seonghoon-Yu/Paper_Review_and_Implementation_in_PyTorch/blob/master/Super_Resolution/SRCNN(2014).ipynb) / [Blog Posting](https://deep-learning-study.tistory.com/688)], paper [[pdf](https://arxiv.org/abs/1501.00092)]

- [FlowNet(2015)](https://deep-learning-study.tistory.com/870), paper [[pdf](https://arxiv.org/abs/1504.06852)]

- [PWC-Net(2017)](https://deep-learning-study.tistory.com/871), paper [[pdf](https://arxiv.org/abs/1709.02371)]

- [Residual Non-local Attention Networks for Image Restoration(2019)](https://deep-learning-study.tistory.com/853), paper [[pdf](https://arxiv.org/abs/1903.10082)]



## 3D Vision

- [Convolutional-Recursive Deep Learning for 3D Object Classification(2012)](https://deep-learning-study.tistory.com/694), paper [[pdf](https://papers.nips.cc/paper/2012/file/3eae62bba9ddf64f69d49dc48e2dd214-Paper.pdf)]

- [PointNet(2016)](https://deep-learning-study.tistory.com/702), paper [[pdf](https://arxiv.org/abs/1612.00593)]

- [Set Transformer(2018)](https://deep-learning-study.tistory.com/777), paper [[pdf](https://arxiv.org/abs/1810.00825)]

- [Centroid Transformer(2021)](https://deep-learning-study.tistory.com/795), paper [[pdf](https://arxiv.org/abs/2102.08606)]



## NLP
- PyTorch 코드로 살펴보는 [Seq2Seq(2014)](https://deep-learning-study.tistory.com/685), PyTorch Code [[Google Colab](https://github.com/Seonghoon-Yu/Paper_Review_and_Implementation_in_PyTorch/blob/master/NLP/Seq2Seq(2014).ipynb) / [Blog Posting](https://deep-learning-study.tistory.com/686)], paper [[pdf](https://arxiv.org/abs/1409.3215)]

- PyTorch 코드로 살펴보는 [GRU(2014)](https://deep-learning-study.tistory.com/691), paper [[pdf](https://arxiv.org/abs/1406.1078)]

- PyTorch 코드로 살펴보는 [Attention(2015)](https://deep-learning-study.tistory.com/697), paper [[odf](https://arxiv.org/pdf/1409.0473.pdf)]

- PyTorch 코드로 살펴보는 [Convolutional Sequence to Sequence Learning(2017)](https://deep-learning-study.tistory.com/704), paper [[pdf](https://arxiv.org/pdf/1705.03122.pdf)]

- PyTorch 코드로 살펴보는 [Transforemr(2017)](https://deep-learning-study.tistory.com/710), paper [[pdf](https://arxiv.org/abs/1706.03762)]

- [BERT(2018)](https://deep-learning-study.tistory.com/770), paper [[pdf](https://arxiv.org/abs/1810.04805)]

- [ELECTRA: Pre-training Text Encoders as Discriminators Rather Than Generators(2020)](https://deep-learning-study.tistory.com/921), paper [[pdf](https://arxiv.org/abs/2003.10555)]

### GAN
- PyTorch 구현 코드로 살펴보는 [GAN(2014)](https://deep-learning-study.tistory.com/638), PyTorch Code [[Google Colab](https://github.com/Seonghoon-Yu/Paper_Review_and_Implementation_in_PyTorch/blob/master/GAN/GAN(2014).ipynb) / [Blog Posting](https://deep-learning-study.tistory.com/639)], paper [[pdf](https://arxiv.org/pdf/1406.2661.pdf)]

- PyTorch 구현 코드로 살펴보는 [CGAN(2014)](https://deep-learning-study.tistory.com/640), PyTorch Code [[Google Colab](https://github.com/Seonghoon-Yu/Paper_Review_and_Implementation_in_PyTorch/blob/master/GAN/CGAN(2014).ipynb) / [Blog Posting](https://deep-learning-study.tistory.com/641)], paper [[pdf](https://arxiv.org/abs/1411.1784)]

- [Generative Moment Matching Network(2015)](https://deep-learning-study.tistory.com/893), paper [[pdf](https://arxiv.org/abs/1502.02761)]

- PyTorch 구현 코드로 살펴보는 [DCGAN(2015)](https://deep-learning-study.tistory.com/642), PyTorch Code [[Google Colab](https://github.com/Seonghoon-Yu/Paper_Review_and_Implementation_in_PyTorch/blob/master/GAN/pix2pix(2016).ipynb) / [Blog Posting](https://deep-learning-study.tistory.com/646)], paper [[pdf](https://arxiv.org/abs/1511.06434)]

- PyTorch 구현 코드로 살펴보는 [Pix2Pix(2016)](https://deep-learning-study.tistory.com/645), PyTorch Code [[Google Colab](https://github.com/Seonghoon-Yu/Paper_Review_and_Implementation_in_PyTorch/blob/master/GAN/DCGAN(2015).ipynb) / [Blog Posting](https://deep-learning-study.tistory.com/643)], paper [[pdf](https://arxiv.org/abs/1611.07004)]

### Diffusion Model
- PyTorch 구현 코드로 살펴보는 [Implementation of Denoising Diffusion Probabilistic Model in Pytorch](https://github.com/lucidrains/denoising-diffusion-pytorch?tab=readme-ov-file), PyTorch Code [[Google Colab]()]

### Diffusion Model based Anomaly Detection

- [Anoddpm: Anomaly detection with denoising diffusion probabilistic models using simplex noise(2022)], paper [[pdf](https://openaccess.thecvf.com/content/CVPR2022W/NTIRE/papers/Wyatt_AnoDDPM_Anomaly_Detection_With_Denoising_Diffusion_Probabilistic_Models_Using_Simplex_CVPRW_2022_paper.pdf)]

- [Unsupervised surface anomaly detection with diffusion probabilistic model(2023)], paper [[pdf](https://openaccess.thecvf.com/content/ICCV2023/papers/Zhang_Unsupervised_Surface_Anomaly_Detection_with_Diffusion_Probabilistic_Model_ICCV_2023_paper.pdf)]

- [Multimodal Motion Conditioned Diffusion Model for Skeleton-based Video Anomaly Detection(2023)], paper [[pdf](https://openaccess.thecvf.com/content/ICCV2023/papers/Flaborea_Multimodal_Motion_Conditioned_Diffusion_Model_for_Skeleton-based_Video_Anomaly_Detection_ICCV_2023_paper.pdf)]



    
  
# Resources
## Introductory Posts

**What are Diffusion Models?** \
[[Website](https://lilianweng.github.io/lil-log/2021/07/11/diffusion-models.html)] \
11 Jul 2021 \
전반적인 Generative 모델에 대한 소개와 Diffusion model의 수식적인 정리가 깔끔하게 되어있음. 복숭아 아이콘 사이트.

**Generative Modeling by Estimating Gradients of the Data Distribution** \
[[Website](https://yang-song.github.io/blog/2021/score/)] \
5 May 2021 \
Yang Song 블로그. Score-based models를 기존의 MCMC 부터 시작하여 차근차근 훑어줌. 추천.

## Introductory Papers

**Understanding Diffusion Models: A Unified Perspective** \
*Calvin Luo* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2208.11970)] \
25 Aug 2022 \
기존 Diffusion 논문들의 notation이 다 달라서 힘든데, 이 논문이 전체적인 정리를 싹 다 해놓음. 구글에서 썼고, VAE에서 부터 classifier guidance까지 수식적으로도 총정리 해둠. 수학 증명이 안되는 부분이 있다면 참고하기 좋음.

**A Survey on Generative Diffusion Model** \
*Hanqun Cao, Cheng Tan, Zhangyang Gao, Guangyong Chen, Pheng-Ann Heng, Stan Z. Li* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2209.02646)] \
6 Sep 2022 \
Survey.

**Diffusion Models: A Comprehensive Survey of Methods and Applications** \
*Ling Yang, Zhilong Zhang, Shenda Hong, Wentao Zhang, Bin Cui* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2209.00796)] \
9 Sep 2022 \
Survey.


# Papers


## Must-read papers

**Deep Unsupervised Learning using Nonequilibrium Thermodynamics** \
*Jascha Sohl-Dickstein, Eric A. Weiss, Niru Maheswaranathan, Surya Ganguli* \
ICML 2015. [[Paper](https://arxiv.org/abs/1503.03585)] [[Github](https://github.com/Sohl-Dickstein/Diffusion-Probabilistic-Models)] \
2 Mar 2015 \
Diffusion의 시작. Reverse diffusion process를 통해 데이터를 복구하는 방식을 제안함. 각각의 time step 별 가우시안 분포를 reparameterize 하는 방식의 시초라고 할 수 있다. 하지만 안읽어도 큰 문제는 없음. (DDPM을 이해했다는 전제하에)

**Denoising Diffusion Probabilistic Models** \
*Jonathan Ho, Ajay Jain, Pieter Abbeel* \
NeurIPS 2020. [[Paper](https://arxiv.org/abs/2006.11239)] [[Github](https://github.com/hojonathanho/diffusion)] [[Github2](https://github.com/pesser/pytorch_diffusion)] \
19 Jun 2020 \
DDPM. 읽어야 함. xt를 x0를 가지고 바로 샘플링하는 방식 제안, Loss를 simple하게 만들어도 잘 된다는 것을 보임.

**Improved Denoising Diffusion Probabilistic Models** \
*Alex Nichol<sup>1</sup>, Prafulla Dhariwal<sup>1</sup>* \
ICLR 2021. [[Paper](https://arxiv.org/abs/2102.09672)] [[Github](https://github.com/openai/improved-diffusion)] \
18 Feb 2021 \
사실 필수로 읽어야 하는 논문까지는 아님. 아키텍처의 변화와 스케줄링의 변화를 줌. 하지만 여기저기서 많이 등장하므로 읽어두면 좋음. 중요도는 그리 높지 않음.

**Diffusion Models Beat GANs on Image Synthesis** \
*Prafulla Dhariwal<sup>1</sup>, Alex Nichol<sup>1</sup>* \
arXiv 2021. [[Paper](https://arxiv.org/abs/2105.05233)] [[Github](https://github.com/openai/guided-diffusion)] \
11 May 2021 \
Classifier guidance 방식을 제안한 논문. 정말 많이 쓰이고 있으며 읽어두는 것을 추천.

**Denoising Diffusion Implicit Models**  \
*Jiaming Song, Chenlin Meng, Stefano Ermon* \
ICLR 2021. [[Paper](https://arxiv.org/abs/2010.02502)] [[Github](https://github.com/ermongroup/ddim)] \
6 Oct 2020 \
Marcov chain을 끊고 Deterministic 하게 만든 논문. 수식적으로 복잡하나, 잘 이해해둬야 하는 필수 논문 중 하나.

**Generative Modeling by Estimating Gradients of the Data Distribution** \
*Yang Song, Stefano Ermon* \
NeurIPS 2019. [[Paper](https://arxiv.org/abs/1907.05600)] [[Project](https://yang-song.github.io/blog/2021/score/)] [[Github](https://github.com/ermongroup/ncsn)] \
12 Jul 2019 \
Score-based models의 시초격인 논문. 결국 VE-SDE를 이해하기 위해선 이 논문이 선행되어야 함.

**Score-Based Generative Modeling through Stochastic Differential Equations** \
*Yang Song, Jascha Sohl-Dickstein, Diederik P. Kingma, Abhishek Kumar, Stefano Ermon, Ben Poole* \
ICLR 2021 (Oral). [[Paper](https://arxiv.org/abs/2011.13456)] [[Github](https://github.com/yang-song/score_sde)] \
26 Nov 2020 \
Score-based 와 DDPM을 SDE로 묶어낸 논문. 매우 잘 써진 논문이라 생각하며, 필수적으로 읽어봐야 한다고 생각.

**Variational Diffusion Models** \
*Diederik P. Kingma, Tim Salimans, Ben Poole, Jonathan Ho* \
NeurIPS 2021. [[Paper](https://arxiv.org/abs/2107.00630)] [[Github](https://github.com/revsic/jax-variational-diffwave)] \
1 Jul 2021 \
필수라고 적어놨지만 필자도 아직 안읽었습니다.. SNR을 정의 내린 논문. 그리고 수식적으로 잘 정리된 논문. 조만간 읽고 업데이트 하겠습니다.

**Elucidating the Design Space of Diffusion-Based Generative Models** \
*Tero Karras, Miika Aittala, Timo Aila, Samuli Laine* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2206.00364)] \
1 Jun 2022 \
실험적으로 Diffusion model을 어떻게 설계하는 것이 좋은지 잘 정리해놓은 논문.

**Classifier-Free Diffusion Guidance** \
*Jonathan Ho, Tim Salimans* \
NeurIPS Workshop 2021. [[Paper](https://arxiv.org/abs/2207.12598)] \
28 Sep 2021 \
GAN으로 치면 condition GAN. 외부에서 classifier로 guidance를 주는 대신, UNet에 바로 컨디션을 꽂아줌. 이 때 수식을 classifier guidance랑 같아지도록 전개, 잘 됨. 현재 잘 되는 대부분의 모델들은 free guidance 방식으로 학습됨.

## Personalized

따로 모을 필요가 느껴져서 목록을 새로 만들었습니다. 아직 모아놓지 않았습니다. 곧 모아볼게요.

**InstantBooth: Personalized Text-to-Image Generation without Test-Time Finetuning**\
*Jing Shi, Wei Xiong, Zhe Lin, Hyun Joon Jung*\
arXiv 2023. [[Paper](https://arxiv.org/abs/2304.03411)]\
[Submitted on 6 Apr 2023]\
1.texture inversion 처럼 토큰 하나 만들어주는 encoder를 학습시킴. 2. 이미지를 패치로 쪼개서 패치별 feature를 모아서 concat함. 3. cross att와 self att 사이에 adapter를 하나 넣어서 이미지를 생성함. 이 3단계로 personalize하는 논문. 개인적으로 패치단위로 쪼개서 feature를 뽑는 방식 덕분에 얼굴이 잘 나오는 점이 좋았고, token의 크기를 renormalization 해줘서 concept에 잡아먹히지 않게 하는 기법이 좋았음. remormalization은 다른 토큰들과 정도가 비슷해지도록 학습된 토큰을 normalization 해주는 기법.



## Stable Diffusion Freeze

**Adding Conditional Control to Text-to-Image Difusion Models**\
*Lvmin Zhang, Maneesh Agrawala*\
[[Code](https://github.com/lllyasviel/ControlNet)] \
어떤 condition 이든 학습할 수 있는 ControlNet 을 제안. Stable Diffusion encoder 의 copy 를 hypernetwork 처럼 활용하되, 학습의 안정성을 위해 zero-conv 를 도입한다. 

**Zero-shot Image-to-Image Translation**\
*Gaurav Parmar, Krishna Kumar Singh, Richard Zhang, Yijun Li, Jingwan Lu, Jun-Yan Zhu*\
arXiv 2023. [[Paper](https://arxiv.org/abs/2302.03027)] \
6 Feb 2022 \
별도의 user prompt 없이 source word(eg. dog) 와 target word(e.g. cat) 만 가지고 image translation하는 논문. 해당 단어가 포함된 여러개의 문장의 CLIP embedding 간의 차이를 editing direction으로 설정하여 inference 할때 text condition에 direction만 더하여 editing 가능, input image의 content structure 유지를 위해서 cross attention guidance를 제시(content와 background유지 굿), gaussian distribution유지를 위한 autocorrelation regularization 제안. 

**GLIGEN: Open-Set Grounded Text-to-Image Generation** \
*Yuheng Li, Haotian Liu, Qingyang Wu, Fangzhou Mu, Jianwei Yang, Jianfeng Gao, Chunyuan Li, Yong Jae Lee* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2301.07093)]\
[Submitted on 17 Jan 2023]\
Stable diffusion은 freeze 해 둔 채로 self attention과 cross attention 사이에 Gated Self attention layer를 추가하여 학습. Bounding box와 캡션, key point(스켈레톤), 이미지 가이드로 원하는 위치에 원하는 샘플을 넣을 수 있음. 잘되고, 실험 엄청 많이 해줌. 중간에 layer 넣는다는 점이 마음에 듬.

**Null-text Inversion for Editing Real Images using Guided Diffusion Models** \
*Ron Mokady, Amir Hertz, Kfir Aberman, Yael Pritch, Daniel Cohen-Or* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2211.09794)] \
17 Nov 2022 \
별도의 model fine-tuning 없이, real image 에 해당하는 null-text를 optimization 하여 prompt2prompt 방식으로 object의 semantic detail을 유지하면서 image editing을 가능하게함. 방법 좋은 결과 좋은. 괜찮은 논문.

**DiffEdit: Diffusion-based semantic image editing with mask guidance** \
*Guillaume Couairon, Jakob Verbeek, Holger Schwenk, Matthieu Cord* \
Submitted to ICLR2023. [[Paper](https://arxiv.org/abs/2210.11427)] \
20 Oct 2022 \
Reference text와 query text가 주어졌을때 두 텍스트를 적용했을때의 noise estimates 차이로 마스크를 생성 - 생성한 마스크를 통해 DDIM decoding과정에서 encoding된 것과 적절히 합쳐서 text 부분만 edit하는 간단한 방법.

**An Image is Worth One Word: Personalizing Text-to-Image Generation using Textual Inversion** \
*Rinon Gal, Yuval Alaluf, Yuval Atzmon, Or Patashnik, Amit H. Bermano, Gal Chechik, Daniel Cohen-Or* \
arXiv 2022. ICLR2023 submission [[Paper](https://arxiv.org/abs/2208.01618)] \
[Submitted on 2 Aug 2022] \
이미지 3~5장을 S* 라는 문자로 inversion한다. GAN inversion과 유사. 이미지를 생성하는 과정에서 나오는 노이즈와 given image를 inversion 하는 과정에서 나오는 노이즈간의 MSE loss를 사용하여 "A photo of S*" 라는 prompt의 S*에 해당하는 토큰을 직접 optimize한다.

**Adding Conditional Control to Text-to-Image Diffusion Models**\
*Lvmin Zhang, Maneesh Agrawala*\
arXiv 2023. [[Paper](https://arxiv.org/abs/2302.05543)] [[Code](https://github.com/lllyasviel/ControlNet)]\
[Submitted on 10 Feb 2023]\
컨디션을 처리해주는 네트워크를 하나 만들고, UNet의 feature에다가 꽂아준다. 일명 ControlNet. 엄청 잘된다. Asyrp의 상위호완 느낌이랄까
  
  
**Unsupervised Discovery of Semantic Latent Directions in Diffusion Models**\
*Yong-Hyun Park, Mingi Kwon, Junghyo Jo, Youngjung Uh*\
arXiv 2023. [[Paper](https://arxiv.org/abs/2302.12469)]\
[Submitted on 24 Feb 2023]\
UNet의 bottleneck이 local하게 linear하다는 성질을 사용하여 리만기하학을 사용한 unsupervised editing direction을 찾는 방법을 제안한 논문이다. 갓용현님의 첫번째 논문이며 나름 좋은 논문이다. 읽어주세염! 참고로, Diffusion models editing에서 보지 못했던 pose 변화 editing을 보여주고 있다. Stable diffusion에서도 editing이 된다.


## Stable Diffusion Finetuning

**Paint by Example: Exemplar-based Image Editing with Diffusion Models** \
*Binxin Yang, Shuyang Gu, Bo Zhang, Ting Zhang, Xuejin Chen, Xiaoyan Sun, Dong Chen, Fang Wen* \
CVPR2023 submission. [[Paper](https://arxiv.org/abs/2211.13227)] \
[Submitted on 23 Nov 2022] \
유저가 지정한 영역에 컨디션으로 주어진 이미지의 semantic을 생성한 논문. 1. StableDiffusion으로 init 2. 이미지의 메인 오브젝트 패치를 떼어내고, CLIP 이미지 인코더에 augmentation해서 넣어준다. 이 때 CLIP을 1024까지 임베딩을 시켜버리고, 이걸 다시 리니어레이어 몇개 통과시켜서 컨디션으로 넣어줌. 3. 2번에 따라서 학습.  결과 좋음. 방법 좋음. 논문 잘 읽힘. 괜찮은 논문.

**Multi-Concept Customization of Text-to-Image Diffusion** \
*Nupur Kumari, Bingliang Zhang, Richard Zhang, Eli Shechtman, Jun-Yan Zhu* \
arxiv Submitted on 8 Dec 2022\ preprint [[Paper](https://arxiv.org/abs/2212.04488)] 
 1)model 일부만 fine-tuning + 2) text optimization 을 통해서 Large text-to-image Diffusion model을 few-shot user images 상에서 customizing 하는 논문

  **SmartBrush: Text and Shape Guided Object Inpainting with Diffusion Model**\
  *Shaoan Xie, Zhifei Zhang, Zhe Lin, Tobias Hinz, Kun Zhang*\
  arXiv 2022. [[Paper](https://arxiv.org/abs/2212.05034)]\
  [Submitted on 9 Dec 2022]\
  마스크를 주면 거기에 텍스트에 해당하는 이미지 생성. 기본적으로 모델을 훈련을 시키는데, 마스크에 가우시안 블러커널을 통과시키고, 생성되는 이미지에서 마스크를 predict 하게 하여 predict된 마스크 영역만 대체하는 방식으로 background를 최대한 지킨다. 실제로 생성 영역을 넓게 잡아도 background가 상당히 잘 유지된다.


## Image Generation

**On the Importance of Noise Scheduling for Diffusion Models** \
*Ting Chen* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2301.10972)]\
high resolution 에서는 같은 SNR 에서도 이미지가 덜 망가지는 것으로부터, resolution 별 새로운 noise scheduling 을 제안함. \
이미지가 클수록 정보가 살아남는 것으로부터 착안하여, signal 을 낮춰주는 $xt=\sqrt{\alpha} b x_0 + \sqrt{1 - \alpha} \epsilon 을 제안.\
+) UNet backbone 이 아닙니다.

**eDiff-I: Text-to-Image Diffusion Models with an Ensemble of Expert Denoisers** \
*Yogesh Balaji, Seungjun Nah, Xun Huang, Arash Vahdat, Jiaming Song, Karsten Kreis, Miika Aittala, Timo Aila, Samuli Laine, Bryan Catanzaro, Tero Karras, Ming-Yu Liu*\
arxiv [Submitted on 2 Nov 2022 (v1), last revised 17 Nov 2022 (this version, v4)]\
 Nvidia version large text-to-image model, 한개의 diffusion model말고 각 stpe별로 여러개의 network를 학습시켜 ensemble한다.

**Score-Based Generative Modeling with Critically-Damped Langevin Diffusion** \
*Tim Dockhorn, Arash Vahdat, Karsten Kreis* \
arXiv 2021. [[Paper](https://arxiv.org/abs/2112.07068)] [[Project](https://nv-tlabs.github.io/CLD-SGM/)] \
14 Dec 2021 \
Nvidia에서 낸 논문으로 기존에 Score-based에 velocity 축을 하나 더 만들어서 수렴도 잘 되고 학습도 빠르게 만듬. 수학적으로 잘 정리되어있어서 좋은 논문.

**Cascaded Diffusion Models for High Fidelity Image Generation** \
*Jonathan Ho<sup>1</sup>, Chitwan Saharia<sup>1</sup>, William Chan, David J. Fleet, Mohammad Norouzi, Tim Salimans* \
arXiv 2021. [[Paper](https://arxiv.org/abs/2106.15282)] [[Project](https://cascaded-diffusion.github.io/)] \
30 May 2021 \
이미지 resolution을 키워가면서 생성하는 방법 소개.

**Soft Truncation: A Universal Training Technique of Score-based Diffusion Model for High Precision Score Estimation**\
*Dongjun Kim, Seungjae Shin, Kyungwoo Song, Wanmo Kang, Il-Chul Moon*\
icml 2022. [[Paper](https://arxiv.org/abs/2106.05527)] \
11 Jun 2022 \
이미지를 좀 더 잘 뽑아내는 방법 소개.

**Your ViT is Secretly a Hybrid Discriminative-Generative Diffusion Model** \
*Xiulong Yang<sup>1</sup>, Sheng-Min Shih<sup>1</sup>, Yinlin Fu, Xiaoting Zhao, Shihao Ji* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2208.07791)] [[Github](https://github.com/sndnyang/Diffusion_ViT)] \
16 Aug 2022 \
ViT를 가지고 Diffusion을 만들었지만 classification도 같이 한다는 것이 중요포인트. 그러나 이미지 생성 성능은 그리 좋지 못함. 다만 기존 하이브리드모델 중에선 제일 좋은듯.

**Progressive Deblurring of Diffusion Models for Coarse-to-Fine Image Synthesis, Sangyun Lee et al., 2022** \
*Sangyun Lee, Hyungjin Chung, Jaehyeon Kim, Jong Chul Ye* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2207.11192?context=cs)] [[Project](https://github.com/sangyun884/blur-diffusion)] \
16 Jul 2022 \
상윤좌의 논문으로, diffusion models의 generation과정이 coarse-to-fine이 아니라 holistically 생성되는것에 주목하여 이를 해결하고자 blur kernel을 삽입하여 train.
Noise에 가까울 수록 low frequency 정보만 남도록 gaussian kernel 통과시키고, 결과적으로 low freqeucny(content)정보부터 미리 생성하고, high freqeuncy(style, detail)을 나중에 생성하도록 explicit bias를 줌.

**Soft Diffusion: Score Matching for General Corruptions, Giannis Daras et al., 2022**  \
*Giannis Daras, Mauricio Delbracio, Hossein Talebi, Alexandros G. Dimakis, Peyman Milanfar* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2209.05442)] \
12 Sep 2022 \
gaussian noise말고 blur까지 씌우면 fid가 더 좋아진다 + new sampling method (momentum sampling)제안, noise(blur) scheduling 제안\

**Maximum Likelihood Training of Implicit Nonlinear Diffusion Models**\
*Dongjun Kim, Byeonghu Na, Se Jung Kwon, Dongsoo Lee, Wanmo Kang, Il-Chul Moon*\
NeurIPS22. [[Paper](https://arxiv.org/abs/2205.13699)]\
27 May 2022 \
Normalizing flow의 invertible한 성질을 적용하여, data adatible 한 nonlinear diffusion process를 implicit하게 학습. FID 성능을 올림.\

**Scalable Diffusion Models with Transformers** \
*William Peebles, Saining Xie* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2212.09748)] [[Project page](https://www.wpeebles.com/DiT)] [[Git](https://github.com/facebookresearch/DiT)]\
[Submitted on 19 Dec 2022] \
트랜스포머를 사용해서 이미지넷에서 SOTA. 기본적으로 VAE의 latent 상에서의 Diffusion이며, t랑 class를 concat 해서 mlp 하나 태우고, adaLN 을 적용시킴. 약간 LDM을 transformer로 구현한 느낌. 실험 좋고 내용 간단한데 굳이 열심히 읽어볼 필요는 없는 논문. \
  
**On the Importance of Noise Scheduling for Diffusion Models** \
*Ting Chen* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2301.10972)]\
high resolution 에서는 같은 SNR 에서도 이미지가 덜 망가지는 것으로부터, resolution 별 새로운 noise scheduling 을 제안함. \
이미지가 클수록 정보가 살아남는 것으로부터 착안하여, signal 을 낮춰주는 $xt=\sqrt{\alpha} b x_0 + \sqrt{1 - \alpha} \epsilon 을 제안.\
+) UNet backbone 이 아닙니다.
  


## Connection with other framworks

**Diffusion Autoencoders: Toward a Meaningful and Decodable Representation** \
*Konpat Preechakul, Nattanat Chatthee, Suttisak Wizadwongsa, Supasorn Suwajanakorn* \
CVPR 2022. [[Paper](https://arxiv.org/abs/2111.15640)] [[Project](https://diff-ae.github.io/)] [[Github](https://github.com/phizaz/diffae)] \
30 Dec 2021 \
Diffusion models에 semantic latent를 컨디션으로 주어서 Autoencoder 처럼 만듬. 그래서 latent가 생겼고, manipulation이 가능해짐. 성능 좋고 잘됨.

**DiffuseVAE: Efficient, Controllable and High-Fidelity Generation from Low-Dimensional Latents** \
*Kushagra Pandey, Avideep Mukherjee, Piyush Rai, Abhishek Kumar* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2201.00308)] [[Github](https://github.com/kpandey008/DiffuseVAE)] \
2 Jan 2022 \
제대로 못읽었지만, VAE의 형태를 빌려와서 합친 논문.

**High-Resolution Image Synthesis with Latent Diffusion Models** \
*Robin Rombach<sup>1</sup>, Andreas Blattmann<sup>1</sup>, Dominik Lorenz, Patrick Esser, Björn Ommer* \
arXiv 2021. [[Paper](https://arxiv.org/abs/2112.10752)] [[Github](https://github.com/CompVis/latent-diffusion)] \
20 Dec 2021 \
global AutoEncoder를 학습해서 그 latent 상에서 diffusion을 한 논문. stable-diffusion이 이 논문이다.

**Score-based Generative Modeling in Latent Space** \
*Arash Vahdat<sup>1</sup>, Karsten Kreis<sup>1</sup>, Jan Kautz* \
arXiv 2021. [[Paper](https://arxiv.org/abs/2106.05931)] \
10 Jun 2021
VAE랑 합친 논문. VAE와 Diffusion을 동시에 학습. Diffusion은 VAE의 latent space에서 학습된다.

**Tackling the Generative Learning Trilemma with Denoising Diffusion GANs** \
*Zhisheng Xiao, Karsten Kreis, Arash Vahdat* \
ICLR 2022 (Spotlight). [[Paper](https://arxiv.org/abs/2112.07804)] [[Project](https://nvlabs.github.io/denoising-diffusion-gan)] \
15 Dec 2021 \
GAN으로 특정 timestep의 이미지를 생성하는 방법으로 샘플링도 빠르게, 퀄리티도 좋게 함. GAN+Diffusion.

## Image space guidance sampling

**ILVR: Conditioning Method for Denoising Diffusion Probabilistic Models** \
*Jooyoung Choi, Sungwon Kim, Yonghyun Jeong, Youngjune Gwon, Sungroh Yoon* \
ICCV 2021 (Oral). [[Paper](https://arxiv.org/abs/2108.02938)] [[Github](https://github.com/jychoi118/ilvr_adm)] \
6 Aug 2021 \
이미지를 Low-pass filter 통과시킨 후 합쳐서 원하는 이미지랑 비슷한 이미지 생성

**RePaint: Inpainting using Denoising Diffusion Probabilistic Models** \
*Andreas Lugmayr, Martin Danelljan, Andres Romero, Fisher Yu, Radu Timofte, Luc Van Gool* \
CVPR 2022. [[Paper](https://arxiv.org/abs/2201.09865)] [[Github](https://github.com/andreas128/RePaint)] \
24 Jan 2022 \
Inpainting 논문. 마스크된 영역만 바꿔껴주는 방식을 제안. 여러번 돌리는 방법만 주목해서 보면 됨. 나머진 그닥 필요 없음.

**SDEdit: Image Synthesis and Editing with Stochastic Differential Equations**  \
*Chenlin Meng, Yang Song, Jiaming Song, Jiajun Wu, Jun-Yan Zhu, Stefano Ermon* \
ICLR 2022. [[Paper](https://arxiv.org/abs/2108.01073)] [[Project](https://sde-image-editing.github.io/)] [[Github](https://github.com/ermongroup/SDEdit)] \
2 Aug 2021 \
stroke를 노이즈를 적당히 씌웠다가 샘플링하면 비슷한 색의 real한 이미지를 얻을 수 있음.

## Classifier guidance sampling

**Blended Diffusion for Text-driven Editing of Natural Images** \
*Omri Avrahami, Dani Lischinski, Ohad Fried* \
CVPR 2022. [[Paper](https://arxiv.org/abs/2111.14818)] [[Project](https://omriavrahami.com/blended-diffusion-page/)] [[Github](https://github.com/omriav/blended-diffusion)] \
29 Nov 2021 \
특정 영역에만 CLIP을 가지고 classifier guidance로 text prompt에 맞게 이미지 생성.

**More Control for Free! Image Synthesis with Semantic Diffusion Guidance** \
*Xihui Liu, Dong Huk Park, Samaneh Azadi, Gong Zhang, Arman Chopikyan, Yuxiao Hu, Humphrey Shi, Anna Rohrbach, Trevor Darrell* \
arXiv 2021. [[Paper](https://arxiv.org/abs/2112.05744)] [[Github](https://xh-liu.github.io/sdg/)] \
10 Dec 2021 \
처음으로 text와 image guidance를 둘 다 줄 수 있다고 설명하는 논문. 그런데 둘 다 CLIP을 사용한 classifier guidance이다.

**Generating High Fidelity Data from Low-density Regions using Diffusion Models** \
*Vikash Sehwag, Caner Hazirbas, Albert Gordo, Firat Ozgenel, Cristian Canton Ferrer* \
CVPR2022, arXiv 2022. [[Paper](https://arxiv.org/abs/2203.17260)] \
31 Mar 2022 \
GAN처럼 Discriminator를 하나 사용해서 확률이 낮은 이미지를 뽑도록 유도. Low-density 이미지를 생성함.

**Self-Guided DIffusion Models** \
*Vincent Tao Hu, David W.Zhang, Yuki M.Asano, Gertjan J. Burghouts, Cees G. M. Snoek* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2210.06462)] \
12 Oct 2022 \
Off-the-shelf model들의 사용으로 feature를 뽑아내고 클러스터링을 활용한 self-guided label -> classifier, object detection, semantic segmentation 등으로 guidance를 주어 그에 따르 이미지생성 (시간이 오래 걸릴듯, high resolution 어렵다는 단점)

## Image Editing

**Unsupervised Discovery of Semantic Latent Directions in Diffusion Models**\
*Yong-Hyun Park, Mingi Kwon, Junghyo Jo, Youngjung Uh*\
arXiv 2023. [[Paper](https://arxiv.org/abs/2302.12469)]\
[Submitted on 24 Feb 2023]\
UNet의 bottleneck이 local하게 linear하다는 성질을 사용하여 리만기하학을 사용한 unsupervised editing direction을 찾는 방법을 제안한 논문이다. 갓용현님의 첫번째 논문이며 나름 좋은 논문이다. 읽어주세염! 참고로, Diffusion models editing에서 보지 못했던 pose 변화 editing을 보여주고 있다. Stable diffusion에서도 editing이 된다.


**Zero-Shot Image Restoration Using Denoising Diffusion Null-Space Model**\
*Yinhuai Wang, Jiwen Yu, Jian Zhang*\
arXiv 2022. [[Paper](https://arxiv.org/abs/2212.00490)] \
Linear Degradation $\mathbf{A}$ 를 알고 있을때, Realness restoration 을 $\mathbf{A}$ 의 null-space 에서만 진행하는 방법을 제안. 실질적인 이미지 퀄리티 향상은 Repaint 에서 제안된 time-travel 기법을 통해 이뤄졌다. 

**Zero-shot Image-to-Image Translation**\
*Gaurav Parmar, Krishna Kumar Singh, Richard Zhang, Yijun Li, Jingwan Lu, Jun-Yan Zhu*\
arXiv 2023. [[Paper](https://arxiv.org/abs/2302.03027)] \
6 Feb 2022 \
별도의 user prompt 없이 source word(eg. dog) 와 target word(e.g. cat) 만 가지고 image translation하는 논문. 해당 단어가 포함된 여러개의 문장의 CLIP embedding 간의 차이를 editing direction으로 설정하여 inference 할때 text condition에 direction만 더하여 editing 가능, input image의 content structure 유지를 위해서 cross attention guidance를 제시(content와 background유지 굿), gaussian distribution유지를 위한 autocorrelation regularization 제안. 

**Null-text Inversion for Editing Real Images using Guided Diffusion Models** \
*Ron Mokady, Amir Hertz, Kfir Aberman, Yael Pritch, Daniel Cohen-Or* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2211.09794)] \
17 Nov 2022 \
별도의 model fine-tuning 없이, real image 에 해당하는 null-text를 optimization 하여 prompt2prompt 방식으로 object의 semantic detail을 유지하면서 image editing을 가능하게함. 방법 좋은 결과 좋은. 괜찮은 논문.

**Paint by Example: Exemplar-based Image Editing with Diffusion Models** \
*Binxin Yang, Shuyang Gu, Bo Zhang, Ting Zhang, Xuejin Chen, Xiaoyan Sun, Dong Chen, Fang Wen* \
CVPR2023 submission. [[Paper](https://arxiv.org/abs/2211.13227)] \
[Submitted on 23 Nov 2022] \
유저가 지정한 영역에 컨디션으로 주어진 이미지의 semantic을 생성한 논문. 1. StableDiffusion으로 init 2. 이미지의 메인 오브젝트 패치를 떼어내고, CLIP 이미지 인코더에 augmentation해서 넣어준다. 이 때 CLIP을 1024까지 임베딩을 시켜버리고, 이걸 다시 리니어레이어 몇개 통과시켜서 컨디션으로 넣어줌. 3. 2번에 따라서 학습.  결과 좋음. 방법 좋음. 논문 잘 읽힘. 괜찮은 논문.

**Denoising Diffusion Restoration Models** \
*Bahjat Kawar, Michael Elad, Stefano Ermon, Jiaming Song* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2201.11793)] \
27 Jan 2022 \
이미지 자체가 하자가 있다고 생각하고 특정 행렬 곱으로 노이즈나.. 크롭이나.. 그런걸 나타낼 수 있다면 원본을 복구하는 방식 제안.

**Palette: Image-to-Image Diffusion Models** \
*Chitwan Saharia, William Chan, Huiwen Chang, Chris A. Lee, Jonathan Ho, Tim Salimans, David J. Fleet, Mohammad Norouzi* \
NeurlPS 2022. [[Paper](https://arxiv.org/abs/2111.05826)] \
10 Nov 2021 \
별거 안하고 그냥 튜닝해서 모델 하나로 4가지 task에서 SOTA 달성.

**DiffEdit: Diffusion-based semantic image editing with mask guidance** \
*Guillaume Couairon, Jakob Verbeek, Holger Schwenk, Matthieu Cord* \
Submitted to ICLR2023. [[Paper](https://arxiv.org/abs/2210.11427)] \
20 Oct 2022 \
Reference text와 query text가 주어졌을때 두 텍스트를 적용했을때의 noise estimates 차이로 마스크를 생성 - 생성한 마스크를 통해 DDIM decoding과정에서 encoding된 것과 적절히 합쳐서 text 부분만 edit하는 간단한 방법.

**DiffusionCLIP: Text-guided Image Manipulation Using Diffusion Models** \
*Gwanghyun Kim, Jong Chul Ye* \
CVPR 2022. [[Paper](https://arxiv.org/abs/2110.02711)] \
6 Oct 2021 \
CLIP을 가지고 model을 finetuning해서 원하는 attribute로 변환하는 논문.

**TEXT-GUIDED DIFFUSION IMAGE STYLE TRANSFER WITH CONTRASTIVE LOSS FINE-TUNING** \
*Anonymous authors* \
Submitted to ICLR2023. [[Paper](https://openreview.net/forum?id=iJ_E0ZCy8fi)] \
30 Sept 2022 \
CLIP (global + directional) & CUT loss (UNet featuremap들을 패치로 쪼개서 contrastive loss)를 사용해서 stylestransfer.

**Plug-and-Play Diffusion Features for Text-Driven Image-to-Image Translation** \
*Narek Tumanyan, Michal Geyer, Shai Bagon, Tali Dekel* \
CVPR 2023 Submission / preprint [[Paper](https://arxiv.org/abs/2211.12572)] [[Project page](https://pnp-diffusion.github.io)] \
[Submitted on 22 Nov 2022]
Stable Diffusion의 4th layer의 featuremap과 4-11th laeyr의 self attention Q,K 값을 injection 하여 real image의 structure를 유지하면서 text guided로 I2I translation을 가능하게 함. Diffusion model은 freeze, feature만 만져서 성공적으로 editing. 좋은 접근.

**Diffusion Models already have a Semantic Latent Space** \
*Mingi Kwon, Jaeseok Jeong, Youngjung Uh* \
ICLR 2023 Spotlight / preprint [[Paper](https://arxiv.org/abs/2210.10960)] [[Project page](https://kwonminki.github.io/Asyrp/)] \
[Submitted on 20 Oct 2022] \
DDIM의 샘플링 공식 중 predicted x0 부분만 바꿔주면 U-Net의 bottle-neck 부분을 semantic latent space로 쓸 수 있음을 보여준 논문. Asyrp을 제안함. 잘됩니당 좋은 논문입니당 읽어주세요.

**EDICT: Exact Diffusion Inversion via Coupled Transformations** \
*Bram Wallace, Akash Gokul, Nikhil Naik* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2211.12446)]\
DDIM inversion 과 Normalizing flow 에서 자주 사용되는 Affine coupling layer 의 수식이 동일하다는 점에서 착안하여, 완벽하게 inversion 되는 process 를 제안. \
text-conditional 일때나 guidance scale 이 클때도 reconstruction 성능이 좋습니다.

  **Boundary Guided Mixing Trajectory for Semantic Control with Diffusion Models**\
  *Ye Zhu, Yu Wu, Zhiwei Deng, Olga Russakovsky, Yan Yan*\
  arXiv 2023. [[Paper](https://arxiv.org/abs/2302.08357)]\
  [Submitted on 16 Feb 2023]\
  Asyrp을 사용하면 (Diffusion models already have a semantic latent space) 생기는 문제를 inversion 이미지와 generated 이미지의 xT 분포를 가지고 분석함. inversion한 이미지가 가우시안 분포 껍질 안쪽에 있다고 말하고, 이걸 맞춰주는 방식을 제안함. - 제대로 안읽어서 추후 업데이트 예정.
  
  
  **MasaCtrl: Tuning-Free Mutual Self-Attention Control for Consistent Image Synthesis and Editing** \
* Mingdeng Cao, Xintao Wang, Zhongang Qi, Ying Shan, Xiaohu Qie, Yinqiang Zheng *\
arXiv 2023. [[Paper](https://arxiv.org/abs/2304.08465)]\
[Submitted on 17 Apr 2023] \
Plug&play upgrade version, code 공개가 안되어있긴하지만 논문 figure보면 결과가 이상적. reference image 로 부터 가져온 self attention key, value를 editing 과정에서 사용할 때, 처음부터 쓰면 editing flexibility가 떨어지기 때문에, 일정 step 이후부터 쓰기를 제안 + cross attention 으로 부터 object와 background mask를 얻어서 self-attention guide에 사용.


## Text-focused

**Multi-Concept Customization of Text-to-Image Diffusion** \
*Nupur Kumari, Bingliang Zhang, Richard Zhang, Eli Shechtman, Jun-Yan Zhu* \
arxiv Submitted on 8 Dec 2022 \ 
preprint [[Paper](https://arxiv.org/abs/2212.04488)] \
 1)model 일부만 fine-tuning + 2) text optimization 을 통해서 Large text-to-image Diffusion model을 few-shot user images 상에서 customizing 하는 논문

**Optimizing Prompts for Text-to-Image Generation** \
*Yaru Hao, Zewen Chi, Li Dong, Furu Wei* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2212.09611)][[Demo page](https://huggingface.co/spaces/microsoft/Promptist)][[Git](https://github.com/microsoft/LMOps/tree/main/promptist)] \
[Submitted on 19 Dec 2022] \
"A white knight riding a black horse." -> "a white knight riding a black horse, intricate, elegant, highly detailed, digital painting, artstation, concept art, sharp focus, illustration, by justin gerard and artgerm, 8 k" 텍스트 뒤에 붙는 글자들을 강화학습으로 만들어낸다. GPT모델을 prompt pair로 fintuning하여 policy 모델로 사용한다. 이미지의 심미적, 텍스트 반영을 기반으로 reward를 주는 형태로 짜여져 있다.

**An Image is Worth One Word: Personalizing Text-to-Image Generation using Textual Inversion** \
*Rinon Gal, Yuval Alaluf, Yuval Atzmon, Or Patashnik, Amit H. Bermano, Gal Chechik, Daniel Cohen-Or* \
arXiv 2022. ICLR2023 submission [[Paper](https://arxiv.org/abs/2208.01618)] \
[Submitted on 2 Aug 2022] \
이미지 3~5장을 S* 라는 문자로 inversion한다. GAN inversion과 유사. 이미지를 생성하는 과정에서 나오는 노이즈와 given image를 inversion 하는 과정에서 나오는 노이즈간의 MSE loss를 사용하여 "A photo of S*" 라는 prompt의 S*에 해당하는 토큰을 직접 optimize한다.

**ReVersion: Diffusion-Based Relation Inversion from Images**\
*Ziqi Huang∗Tianxing Wu∗Yuming JiangKelvin C.K. ChanZiwei Liu*\
arXiv 2023. [[Paper](https://arxiv.org/abs/2303.13495)]\
[Submitted on 23 Mar 2023]\
Relation Inversion이라는 개념을 소개함. CLIP embedding을 살펴보니 명사, 동사, 형용사 등등 품사별로 space가 나눠져 있는것을 관측함. 이에 관계를 나타내주는 text token을 학습을 하는데, contrastive learning으로 positive는 형용사들을, negative로 나머지 정해놓은 단어들을 사용함. 이를 통해 Exemplar Images들이 지니고 있는 관계 ex) 무언가가 어디 위에 그려져 있다던지, 안에 들어가 있다던지, 옆에 나란히 위치한다던지 이런 관계를 학습할 수 있음.

**P+: Extended Textual Conditioning in Text-to-Image Generation**\
*Andrey Voynov, Qinghao Chu, Daniel Cohen-Or, Kfir Aberman*\
arXiv 2023. [[Paper](https://arxiv.org/abs/2303.09522)]\
[Submitted on 16 Mar 2023]\
토큰이 UNet의 layer별로 들어가는데, 이걸 쪼갬. StyleGAN2의 w space와 w+ space를 생각하면 되는데, 각 layer 별 prompt space를 나눠서 생각해서 P+ space라고 부름. 재밌는점은 bottleneck에 가까울수록 semantic한 의미를 지니고있고, 노이즈에 가까울수록 style이라고 해야하나.. 색깔과 관련된 그런 의미를 지님. (Asyrp과 DiffStyle과 결을 같이하는 관측) textual inversion의 확장버전으로 personalization 가능.

**StyleDiffusion: Prompt-Embedding Inversion for Text-Based Editing**\
*Senmao Li, Joost van de Weijer, Taihang Hu, Fahad Shahbaz Khan, Qibin Hou, Yaxing Wang, Jian Yang*\
arXiv 2023. [[Paper](https://arxiv.org/abs/2303.15649)]\
[Submitted on 28 Mar 2023]\
Prompt2Prompt, texture inversion 아류인데, loss로 K와 Q가 같아지도록 loss를 추가함. 뭔가 내용이 더 있는 논문이었는데 기억이 잘...


## Fast Sampling

**Progressive Distillation for Fast Sampling of Diffusion Models**\
  *Tim Salimans, Jonathan Ho*\
  arXiv 2022. [[Paper](https://arxiv.org/abs/2202.00512)] \
  Faster sampling 을 목표로, denoising 2 step 을 예측하는 student 모델을 학습시킨다. 이때, $\epsilon$-prediction 을 하게 될 경우 기존과는 달리 numerical error 에 대한 correction 이 이뤄질 수 없어서 v-prediction 이라는 새로운 parameterization 을 제안함. (v-prediction 은 생각보다 자주 쓰이니 Appendix D 는 보기를 추천)
  
**On distillation of guided diffusion models** \
*Chenlin Meng, Robin Rombach, Ruiqi Gao, Diederik P. Kingma, Stefano Ermon, Jonathan Ho, Tim Salimans* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2210.03142)]\
두번의 distillation 으로 step 을 1~4 step 으로 비약적으로 줄인다. LDM 의 경우 1 step 까지 가능하다. \
stage 1. classifier-free guidance 의 score 에 대한 student 모델 학습. \
stage 2. progressive-distillation 을 통해 step 수를 N/2 으로 계속 줄여나감.

**Minimizing Trajectory Curvature of ODE-based Generative Models** \
*Sangyun Lee, Beomsu Kim, Jong Chul Ye*\
arxiv 27 Jan 2023 [[Paper] (https://arxiv.org/abs/2301.12003)]\
sampling trajectory의 curvature를 줄여서 학습된 denoising model에 ode solver 가 fit 하도록 만들고, 적은 step에서도 generation, reconstruction이 잘 되도록 시도함

**Learning Fast Samplers for Diffusion Models by Differentiating Through Sample Quality** \
*Daniel Watson, William Chan, Jonathan Ho, Mohammad Norouzi* \
ICLR 2022. [[Paper](https://openreview.net/forum?id=VFBjuF8HEp)]  \
11 Feb 2022 \
Pre-trained을 fine-tunning 하지 않고 step#를 줄여서 빠르게 sampling 하면서도 FID/IS 를 최대한 유지할 수 있는 방법제시,
diffusion의 object function(ELBO) term을 무시하고, step과 step사이에 sampling하는 paremeter들만 KID loss 를 줘서 train.

**Pseudo Numerical Methods for Diffusion Models on Manifolds** \
*Luping Liu, Yi Ren, Zhijie Lin, Zhou Zhao* \
ICLR 2022 Poster [[Paper](https://arxiv.org/abs/2202.09778)] \
Submitted on 20 Feb 2022  \
이전 numerical ODE의 방식이 DDPM의 sampling manifold를 제대로 반영하지 못함을 지적, DDIM과 high-order numerical sampling의 장점을 결합하여 새로운 sampling 방식을 제시.
stable diffusion에서 사용된 sampling방식이고 성능이 좋다.

**gDDIM: Generalized denoising diffusion implicit models** \
*Qinsheng Zhang, Molei Tao, Yongxin Chen* \
ICLR 2023 Submission / preprint [[Paper](https://arxiv.org/abs/2206.05564)] \
[Submitted on 11 Jun 2022] \
DDPM, DDIM, 등등을 모두 SDE의 형태로 전환, Blur Diffusion이나 Critically-Damped Langevin Diffusion 까지도 SDE로 표현한 뒤, general한 form의 SDE -> DDIM을 만드는 방법을 제안한다. 이를 통해 istropic diffusion models까지 DDIM으로 fast sampling 가능하게 함. 

## Video Generation and Editing

**Video Diffusion Models** \
*Jonathan Ho, Tim Salimans, Alexey Gritsenko, William Chan, Mohammad Norouzi, David Fleet* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2204.03458)] \
7 April 2022 \
Diffusion을 이용한 Video generation을 처음으로 한 논문, Video의 길이를 늘리고, quality를 높이는 것에 대한 방법제시.

**Structure and Content-Guided Video Synthesis with Diffusion Models**\
  *Patrick Esser, Johnathan Chiu, Parmida Atighehchian, Jonathan Granskog, Anastasis Germanidis*\
  arXiv 2023. [[Paper](https://arxiv.org/abs/2302.03011)] [[Project Page](https://research.runwayml.com/gen1)]\
  [Submitted on 6 Feb 2023] \
  비디오2비디오 translation을 할 때, 이미 또는 텍스트로 가이드를 주는 논문. 비디오의 time에 따른 Spatio-temporal을 위해 temporal convolution/attention 네트워크를 삽입하였고, structure를 유지시키기 위해 depth estimation 을 사용하였음. 또한 훈련때 사용한 비디오를 CLIP image encoder에 태워, 기존 텍스트 대신 image로 condition을 줄 수 있도록 훈련함. 
  
**MagicVideo: Efficient Video Generation With Latent Diffusion Models**\
  *Daquan Zhou, Weimin Wang, Hanshu Yan, Weiwei Lv, Yizhe Zhu, Jiashi Feng*\
  arXiv 2023. [[Paper](https://arxiv.org/abs/2211.11018)] [[Project Page](https://magicvideo.github.io/#)]\
  [Submitted on 20 Nov 2022]\
  비디오를 가지고 훈련시키는 데, adaptor 라는 개념을 추가하여, frame 간의 관계 정보를 공유하도록 한다. 이 때 Directed Temporal Attention 을 사용해서 - Masked Self attention과 거의 동일한 개념.- 뒤쪽 frame에게만 영향을 끼치도록 만듬. 나쁘지 않은 논문.
  
**Latent-Shift: Latent Diffusion with Temporal Shift for Efficient Text-to-Video Generation **
*Jie An1;2* Songyang Zhang1;2* Harry Yang2 Sonal Gupta2 Jia-Bin Huang2;3 Jiebo Luo1;2 Xi Yin2*
arXiv 2023. [Paper](https://arxiv.org/pdf/2304.08477.pdf)] [[Project Page(https://latent-shift.github.io/)]
[Submitted on 17 Apr 2023]
T2I model로 T2V model을 학습. 4d tensor(frame x channel x width x height)를 denoising하여 video 생성, Frame 간 정보 교환을 위해 attention 대신 temporal axis로 latent feature (특정 channel만) 를 shift 하는 temporal shift block을 U-Net안에 추가.

**Tune-A-Video: One-Shot Tuning of Image Diffusion Models for Text-to-Video Generation **
*Jay Zhangjie Wu1 Yixiao Ge2 Xintao Wang2 Stan Weixian Lei1 Yuchao Gu1 Yufei Shi1 Wynne Hsu4 Ying Shan2 Xiaohu Qie3 Mike Zheng Shou1*
arXiv 2023. [[Paper](https://arxiv.org/abs/2212.11565)][[Project page](https://tuneavideo.github.io/)]
[Submitted on 17 Mar 2023]
한개의 reference video 에 대하여, T2I diffusion model을 T2V diffusion model로 fine-tunning함. T2I -> T2V 만들때 self-attention을 직전 프레임과 처음 프레임을 key와 value만드는데 쓰도록 바꿈 (Spatial temporal attention) + Temporal attention block 추가(inflation).

**Video-P2P: Video Editing with Cross-attention Control**
*Shaoteng Liu1 Yuechen Zhang1 Wenbo Li1 Zhe Lin3 Jiaya Jia1;2*
arXiv 2023. [[Paper](https://video-p2p.github.io/)] [[Project page](https://video-p2p.github.io/)]
[Submitted on 8 Mar 2023]
Input video 한개에 T2I->T2V fine-tunning(Tune-A-Video와 비슷한 방식), T2I -> T2V 만들때 self-attention을 처음 프레임만을 key와 value만드는데 쓰도록 바꿈 (Frame attention), decoupled-guidance attention으로 background 안바뀌고 foreground object만 editing되도록함(Mask생성)

**Pix2Video: Video Editing using Image Diffusion**
*Duygu Ceylan1* Chun-Hao P. Huang1* Niloy J. Mitra1,2*
arXiv 2023. [[Paper](https://arxiv.org/abs/2303.12688)] [[Project page](https://duyguceylan.github.io/pix2video.github.io/)]
첫 frame 부터 시작하여 이후 frame 으로 점차 propagate하는 방식으로 editing, 이전 프레임과 첫 프레임을 attend하도록 feature를 injection. flickering을 방지하기 위해 이전프레임과 현재프레임의 predicted x0 간의 l2 distance 를 비교하여 denoising할때 classifier guidance를 줌.


**VideoFusion: Decomposed Diffusion Models for High-Quality Video Generation**
*Zhengxiong Luo1,2,4,5 Dayou Chen2 Yingya Zhang2 †Yan Huang4,5 Liang Wang4,5 Yujun Shen3 Deli Zhao2 Jingren Zhou2 Tieniu Tan4,5,6*
arXiv 2023. [[Paper](https://arxiv.org/pdf/2303.08320.pdf)]
[Submitted on 22 Mar 2023]
alibaba에서 release한 text2video diffusion model의 논문, forward 할 때 frame별로 independent noise + shared noise를 섞는 것을 제안.




## 3D

**DiffRF: Rendering-Guided 3D Radiance Field Diffusion** \
*Norman Müller, Yawar Siddiqui, Lorenzo Porzi, Samuel Rota Bulò, Peter Kontschieder, Matthias Nießner*\
arXiv 2022. [[Paper](https://arxiv.org/abs/2212.01206)] \
2 Dec 2022 \
Diffusion 으로 3d radiacne field generation한 논문. 이전에 DreamFusion이나 GAUDI 와 같이 diffusion으로 3D generation하는 works이 있었지만, 3d unet 을 활용하여 3d Radiance field를 직접 denoise하는 것은 이 연구가 처음. 모든 sample을 voxel grid로 만들어야하는 precomputation이 필요하다. quality를 높이기 위해 3d radiance field의 denoising network 학습이외에 render 된 2d image 상에서의 RGB loss와 마찬가지로 rendered image를 처리하는 CNN network를 추가하였다.\

## 수학기반향상

**On Calibrating Diffusion Probabilistic Models**\
*Tianyu Pang, Cheng Lu, Chao Du, Min Lin, Shuicheng Yan, Zhijie Deng*\
  arXiv 2023. [[Paper](https://arxiv.org/abs/2302.10688)] [[Code](https://github.com/thudzj/Calibrated-DPMs)]\
  [Submitted on 21 Feb 2023]\
  각 스텝에서 예측된 스코어의 합이 0이 되어야 한다고 주장. 이를 위해서 Theorem 1을 제안하는데, ∀0≤s<t≤T 일때 s에서 구한 스코어와 t에서 구한 스코어가 같다는 말을 한다. -(xs|xt)일때- 용현님의 생각은 이 Theorem 1이 DDIM이 왜 잘 동작하는지 보여주고 있으며, gDDIM에서 주장하는 바와도 연관된다고 평가하심. 이를 확장하여 x0의 스코어의 평균이 0이니 xt의 스코어의 평균이 0이어야 한다는 주장을 한다. (Eq.13) 이건 공감 못하셨다. 이를 만족시킬 수 있는 예타t를 스코어에 넣는 방법을 제안했고, 이를 통해 DPM-Solver의 성능을 모든 NFE에서 올렸다.

**Improving Score-based Diffusion Models by Enforcing the Underlying Score Fokker-Planck Equation**\
*Chieh-Hsin Lai, Yuhta Takida, Naoki Murata, Toshimitsu Uesaka, Yuki Mitsufuji, Stefano Ermon*\
NeurIPS 2022 Workshop. [[Paper](https://arxiv.org/abs/2210.04296)]\
Submitted on 9 Oct 2022 (v1)\
Fokker-Planck Equations은 브라운운동에서 한 샘플의 움직임이 아니라 전체 distribution이 어떻게 움직이지는지에 관련된 수식이다. 이를 Eq.6에서 보여주고 있는데, t~=0 일 때 Fokker-Planck Equations에 위반되는 모습이 보여진다고 주장한다. 이를 감마FP 텀을 가지고 조절해줘서 맞춰주는데, 실험이 많지는 않다. 워크샵 페이퍼이다.

**On Calibrating Diffusion Probabilistic Models**\
*Tianyu Pang, Cheng Lu, Chao Du, Min Lin, Shuicheng Yan, Zhijie Deng*\
arXiv 2023. [[Paper](https://arxiv.org/abs/2302.10688)] [[Code](https://github.com/thudzj/Calibrated-DPMs)]\
[Submitted on 21 Feb 2023]\
모델의 아웃풋인 스코어도 마팅게일을 만족해야 한다고 주장한다. SDE식을 만족한다는 것은 xt가 마팅게일이라는 의미이기도 한데, 직접적으로 스코어가 마팅게일이어야 한다고 말한 논문은 이게 처음인 듯 하다. 조금 어렵다.

**Score-based generative model learnmanifold-like structures with constrained mixing**\
*Li Kevin Wenliang, Ben Moran*\
NeurIPS 2022 Workshop. [[Paper](https://openreview.net/forum?id=eSZqaIrDLZR)]\
score를 svd 해서 분석해본 결과 재밌게도 eigenvalue가 낮은 친구들이 semantic한 의미를 지니고 있음을 보임. 직관적으로 생각해보면 각 score들은 timestep에 맞는 distribution으로 향하는 방향이어야 하고, 이에 맞춰서 eigenvalue가 높은 방향들은 각 distirbution 밖으로 향하는 방향이라고 이해할 수 있음.

**Score-based Diffusion Models in Function Space**\
*Jae Hyun Lim*, Nikola B. Kovachki*, Ricardo Baptista*, Christopher Beckham, Kamyar Azizzadenesheli, Jean Kossaifi, Vikram Voleti, Jiaming Song, Karsten Kreis, Jan Kautz, Christopher Pal, Arash Vahdat, Anima Anandkumar*\
arXiv 2023. [[Paper](https://arxiv.org/abs/2302.07400)]\
[Submitted on 14 Feb 2023]\
꽤나 어려운 논문. 일단 간단히 말하자면 어떤 function을 생성하는 논문임. infinite dimension에서 Lebesgue measure가 불가능 하기 때문에 Radon–Nikodym Theorem을 통해 probability measure를 구함. (정확하지 않은 표현인데.. 요약이 힘드네요. 4번식 밑에 줄이 정확한 표현) 이 때 u(뮤)는 Cameron-Martin space라고 여기고 Feldman–Hájek Theorem을 적용해서 8번 식을 구함. 적다보니 요약이 불가한 논문이란 것을 깨달았고, 본인도 읽은지 몇 주 됐다고 기억이 가물가물함. 추후 업데이트 해보겠음.


## 기타

**Human Motion Diffusion Model** \
*Guy Tevet, Sigal Raab, Brian Gordon, Yonatan Shafir, Daniel Cohen-Or, Amit H. Bermano* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2209.14916)][[Project page](https://guytevet.github.io/mdm-page/)] \
[Submitted on 29 Sep 2022] \
사람의 Motion을 생성하는데 Diffusion을 사용. spatial한 정보가 필요없기에 Transformer를 사용하였다. 이 때 모든 xt에 대하여 모델은 바로 x0를 예측한다. classifier-free guidance를 10%로 사용하였으며 이를 통해 text-to-motion 생성이 가능하다.

**PhysDiff: Physics-Guided Human Motion Diffusion Model** \
*Ye Yuan, Jiaming Song, Umar Iqbal, Arash Vahdat, Jan Kautz* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2212.02500)] \
[Submitted on 5 Dec 2022] \
Motion Diffusion Model에서 발이 떨어지는 문제를 해결하기 위해 강화학습을 사용함. 자세한건 패스..

**Score-based Diffusion Models in Function Space**\
*Jae Hyun Lim*, Nikola B. Kovachki*, Ricardo Baptista*, Christopher Beckham, Kamyar Azizzadenesheli, Jean Kossaifi, Vikram Voleti, Jiaming Song, Karsten Kreis, Jan Kautz, Christopher Pal, Arash Vahdat, Anima Anandkumar*\
arXiv 2023. [[Paper](https://arxiv.org/abs/2302.07400)]\
[Submitted on 14 Feb 2023]\
꽤나 어려운 논문. 일단 간단히 말하자면 어떤 function을 생성하는 논문임. infinite dimension에서 Lebesgue measure가 불가능 하기 때문에 Radon–Nikodym Theorem을 통해 probability measure를 구함. (정확하지 않은 표현인데.. 요약이 힘드네요. 4번식 밑에 줄이 정확한 표현) 이 때 u(뮤)는 Cameron-Martin space라고 여기고 Feldman–Hájek Theorem을 적용해서 8번 식을 구함. 적다보니 요약이 불가한 논문이란 것을 깨달았고, 본인도 읽은지 몇 주 됐다고 기억이 가물가물함. 추후 업데이트 해보겠음.




## 읽을것들

**Soft Diffusion: Score Matching for General Corruptions** \
*Giannis Daras, Mauricio Delbracio, Hossein Talebi, Alexandros G. Dimakis, Peyman Milanfar* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2209.05442)] \
12 Sep 2022 \
blur써 Sota (안읽음)

  
## Active Learning
- [Towards Reducing Labeling Cost in Deep Object Detection(2021)](https://deep-learning-study.tistory.com/732), paper [[pdf](https://arxiv.org/abs/2106.11921)]

## Pose estimation
- [Hourglass(2016)](https://deep-learning-study.tistory.com/617)

## long tail
- [Class-Balanced Loss(2019)](https://deep-learning-study.tistory.com/671), paper [[pdf](https://arxiv.org/pdf/1901.05555.pdf)]

- [Seesaw Loss for Long-Tailed Instance Segmentation(2020)](https://deep-learning-study.tistory.com/902), paper [[pdf](https://arxiv.org/abs/2008.10032)]


## Face Recognition
- Pytorch 구현 코드로 살펴보는 [FaceNet(2015)](https://deep-learning-study.tistory.com/681), paper [[pdf](https://arxiv.org/pdf/1503.03832.pdf)]

## Model Compression
- [Deep Compression(2016)](https://deep-learning-study.tistory.com/683), paper [[pdf](https://arxiv.org/abs/1510.00149)]

## Activation Function
- [Mish(2019)](https://deep-learning-study.tistory.com/636), paper [[pdf](https://arxiv.org/abs/1908.08681)]

## Augmentation
- [CutMix(2019)](https://deep-learning-study.tistory.com/633), paper [[pdf](https://arxiv.org/abs/1905.04899)]

- [Learning Data Augmentation Strategies for Object Detection(2019](https://deep-learning-study.tistory.com/705), paper [[pdf](https://arxiv.org/pdf/1906.11172.pdf)]

- [Simple Copy-Paste is a Strong Data Augmentation Method for Instance Segmentation(2020)](https://deep-learning-study.tistory.com/708), paper [[pdf](https://arxiv.org/abs/2012.07177)]

## Style Transfer
- PyTorch 구현 코드로 살펴보는 [A Neural Algorithm of Artistic Style(2016)](https://deep-learning-study.tistory.com/679), PyTorch Code [[Google Colab](https://github.com/Seonghoon-Yu/Paper_Review_and_Implementation_in_PyTorch/blob/master/style_transfer/style_transfer(2015).ipynb) / [Blog Posting](https://deep-learning-study.tistory.com/680)], paper [[pdf](https://arxiv.org/abs/1508.06576)]

## Regularization
- [DropBlock(2018)](https://deep-learning-study.tistory.com/631), paper [[pdf](https://arxiv.org/abs/1810.12890)]

## Normalization

- [Batch Normalization(2015)](https://deep-learning-study.tistory.com/421)

- [Group Normalization(2018)](https://deep-learning-study.tistory.com/726), paper [[pdf](https://arxiv.org/abs/1803.08494)]

- [Cross iteration BN(2020)](https://deep-learning-study.tistory.com/635), paper [[pdf](https://arxiv.org/abs/2002.05712)]

## Optimization

- [An overview of gradient descent optimization algorithm(2017)](https://deep-learning-study.tistory.com/415)

- [AdamW(2017)](https://deep-learning-study.tistory.com/750), paper [[pdf](https://arxiv.org/abs/1711.05101)]

