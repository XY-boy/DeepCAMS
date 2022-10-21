# DeepCAMS
# Generating long-term (2003-2020) hourly 0.25° global PM2.5 dataset via spatiotemporal downscaling of CAMS with deep learning (DeepCAMS)
## Introuction
This is the official implementation of our paper [Generating long-term (2003-2020) hourly 0.25° global PM2.5 dataset via spatiotemporal downscaling of CAMS with deep learning (DeepCAMS)](https://doi.org/10.1016/j.scitotenv.2022.157747) published on Science of The Total Environment (**STOTEN**).  

### The overall two-stage flowchart
 ![image](/img/flowchart.png)
 
 ### Temporal downscaling results
 ![image](/img/td.png)
 
 ### Spatial Downscaling results
 ![image](/img/sd.png)
 
 ### In-situ Validation
 ![image](/img/eval.png)
 
 #### More details can be found in our paper!
 ## Environment
 * CUDA 10.0
 * pytorch 1.x
 * build [DCNv2](https://github.com/CharlesShang/DCNv2)
 
 ## Dataset Preparation
 Please download our dataset from Zenodo: <a href="https://doi.org/10.5281/zenodo.6967082"><img src="https://zenodo.org/badge/DOI/10.5281/zenodo.6967082.svg" alt="DOI"></a>
 
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
 
 ## Two Stage Training
```
python main.py
```

## Test
```
python eval.py
```

## Citation
If you find our work helpful, please consider to cite:  
```
@article{xiao2022generating,
  title={Generating a long-term (2003- 2020) hourly 0.25° global PM2. 5 dataset via spatiotemporal downscaling of CAMS with deep learning (DeepCAMS)},
  author={Xiao, Yi and Wang, Yuan and Yuan, Qiangqiang and He, Jiang and Zhang, Liangpei},
  journal={Science of The Total Environment},
  volume={848},
  pages={157747},
  year={2022},
  publisher={Elsevier}
}
```

## Acknowledgement
Our work is built upon [XVFI](https://github.com/JihyongOh/XVFI) and [ABPN](https://github.com/Holmes-Alan/ABPN).  
Thanks to the author for the awesome works!



 


