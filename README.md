# DeepCAMS
# Generating long-term (2003-2020) hourly 0.25° global PM2.5 dataset via spatiotemporal downscaling of CAMS with deep learning (DeepCAMS)
## Introuction
This is the official implementation of our paper [Generating long-term (2003-2020) hourly 0.25° global PM2.5 dataset via spatiotemporal downscaling of CAMS with deep learning (DeepCAMS)](https://doi.org/10.1016/j.scitotenv.2022.157747) published on <u>Science of The Total Environment</u> (**STOTEN**).


## Dataset Preparation
 Please download our DeepCAMS from Zenodo: <a href="https://doi.org/10.5281/zenodo.6967082"><img src="https://zenodo.org/badge/DOI/10.5281/zenodo.6967082.svg" alt="DOI"></a>
 
### The overall two-stage flowchart
<img src="img/flowchart.png" alt="Flowchart" width="600"/>
 
 ### Temporal downscaling results
<img src="img/td.png" alt="Temporal Downscaling" width="600"/>
 
 ### Spatial Downscaling results
<img src="img/sd.png" alt="Spatial Downscaling" width="600"/>
 
 ### In-situ Validation
<img src="img/eval.png" alt="OpenAQ in-situ Validation" width="600"/>
 
 #### More details can be found in our paper!
 ## Environment
 * CUDA 10.0
 * pytorch 1.x
 
 ## Model Training
 ### 1) For spatial downscaling
 Download the LR-HR paired Geos-CF from [Google Drive](https://drive.google.com/drive/folders/1DjccGiyZHeBrivw-Xg-FN8MW3noOrGhF?usp=sharing)
 ### 2) For temporal downscaling
 Download the hourly Geos-CF from [Google Drive](https://drive.google.com/drive/folders/1Wr13Q_eQSkRCYZj8741pJ49A655QRQDd?usp=sharing)
 ### 3) Two Stage Training
```
python /T-SR/my_train.py
python /S-SR/main_3x.py
```

## Test
```
python T-SR/test.py
python S-SR/demo_3x.py
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



 


