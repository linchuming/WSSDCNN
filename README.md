# WSSDCNN
The unofficial implementation of Weakly- and Self-Supervised Learning for Content-Aware Deep Image Retargeting



### Preparation

- TensorFlow >= 1.6
- Download the param (577M) from BaiduYun: https://pan.baidu.com/s/15UvYzwriYCOiqqb6ZvhqsA (passwd: jlq5)
- Move the param files to directory `model_ckpt`



### Inference

- ` python test.py`



### Results

The approach has limits and their results are not always good.  Some good results show below (aspect ratio = 0.5) :

| Original                   | WSSDCNN                        | Bicubic                        |
| -------------------------- | ------------------------------ | ------------------------------ |
| ![](results/butterfly.png) | ![](results/butterfly_0.5.png) | ![](results/butterfly_bic.png) |
| ![](results/eagle.png)     | ![](results/eagle_0.5.png)     | ![](results/eagle_bic.png)     |
| ![](results/surfer.png)    | ![](results/surfer_0.5.png)    | ![](results/surfer_bic.png)    |



