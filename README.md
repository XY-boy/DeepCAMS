# DeepCAMS
[STOTEN 2022] Generating long-term (2003-2020) hourly 0.25° global PM2.5 dataset via spatiotemporal downscaling of CAMS with deep learning (DeepCAMS)
# Generating long-term (2003-2020) hourly 0.25° global PM2.5 dataset via spatiotemporal downscaling of CAMS with deep learning (DeepCAMS)
## Introuction
This is the official implementation of our paper [Generating long-term (2003-2020) hourly 0.25° global PM2.5 dataset via spatiotemporal downscaling of CAMS with deep learning (DeepCAMS)](https://doi.org/10.1016/j.scitotenv.2022.157747) published on Science of The Total Environment (**STOTEN**).  

### The overall two-stage flowchart
 ![image](/img/network.png)
 
 ### Quantitive results
 ![image](/img/res1png.png)
 
 ### Qualitive results
 ![image](/img/res2.png)
 #### More details can be found in our paper!
 ## Environment
 * CUDA 10.0
 * pytorch 1.x
 * build [DCNv2](https://github.com/CharlesShang/DCNv2)
 
 ## Dataset Preparation
 Please download our dataset in Baidu Baidu Netdisk [Jilin-189](https://pan.baidu.com/s/1Y1-mS5gf7m8xSTJQPn4WZw) Code:31ct
 or Zenodo: <a href="https://doi.org/10.5281/zenodo.6969604"><img src="https://zenodo.org/badge/DOI/10.5281/zenodo.6969604.svg" alt="DOI"></a>
 
 You can also train your dataset following the directory sturture below!
 
### Data directory structure
trainset--  
&emsp;|&ensp;train--  
&emsp;&emsp;|&ensp;LR4x---  
&emsp;&emsp;&emsp;| 000.png  
&emsp;&emsp;&emsp;| ···.png  
&emsp;&emsp;&emsp;| 099.png  
&emsp;&emsp;|&ensp;GT---   
&emsp;&emsp;|&ensp;Bicubic4x--- 

testset--  
&emsp;|&ensp;eval--  
&emsp;&emsp;|&ensp;LR4x---  
&emsp;&emsp;&emsp;| 000.png  
&emsp;&emsp;&emsp;| ···.png  
&emsp;&emsp;&emsp;| 099.png  
&emsp;&emsp;|&ensp;GT---   
&emsp;&emsp;|&ensp;Bicubic4x--- 
 
 ## Training
```
python main.py
```

## Test
```
python eval.py
```

## Citation
If you find our work helpful, please cite:  
```
@ARTICLE{9530280,  
author={Xiao, Yi and Su, Xin and Yuan, Qiangqiang and Liu, Denghong and Shen, Huanfeng and Zhang, Liangpei},  
journal={IEEE Transactions on Geoscience and Remote Sensing},  
title={Satellite Video Super-Resolution via Multiscale Deformable Convolution Alignment and Temporal Grouping Projection},   
year={2022},  
volume={60},  
number={},  
pages={1-19},  
doi={10.1109/TGRS.2021.3107352}
}
```

## Acknowledgement
Our work is built upon [RBPN](https://github.com/alterzero/RBPN-PyTorch) and [TDAN](https://github.com/YapengTian/TDAN-VSR-CVPR-2020).  
Thanks to the author for the source code !



 


