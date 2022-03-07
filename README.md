# BMVOS
This is the official implementation of "Pixel-Level Bijective Matching for Video Object Segmentation" (WACV 2022).

PDF: [[official version]](https://openaccess.thecvf.com/content/WACV2022/papers/Cho_Pixel-Level_Bijective_Matching_for_Video_Object_Segmentation_WACV_2022_paper.pdf) 
[[arXiv version]](https://arxiv.org/pdf/2110.01644.pdf)

```
@inproceedings{cho2022pixel,
  title={Pixel-Level Bijective Matching for Video Object Segmentation},
  author={Cho, Suhwan and Lee, Heansung and Kim, Minjung and Jang, Sungjun and Lee, Sangyoun},
  booktitle={Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision},
  pages={129--138},
  year={2022}
}
```

## Benchmark Results
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/pixel-level-bijective-matching-for-video/semi-supervised-video-object-segmentation-on-7)](https://paperswithcode.com/sota/semi-supervised-video-object-segmentation-on-7?p=pixel-level-bijective-matching-for-video)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/pixel-level-bijective-matching-for-video/semi-supervised-video-object-segmentation-on-8)](https://paperswithcode.com/sota/semi-supervised-video-object-segmentation-on-8?p=pixel-level-bijective-matching-for-video)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/pixel-level-bijective-matching-for-video/semi-supervised-video-object-segmentation-on-9)](https://paperswithcode.com/sota/semi-supervised-video-object-segmentation-on-9?p=pixel-level-bijective-matching-for-video)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/pixel-level-bijective-matching-for-video/semi-supervised-video-object-segmentation-on-10)](https://paperswithcode.com/sota/semi-supervised-video-object-segmentation-on-10?p=pixel-level-bijective-matching-for-video)



## Architecture
![image](https://user-images.githubusercontent.com/54178929/142851144-0a2a83cb-e7ef-422b-a20f-14041e328983.png)



## Download
[[pre-computed results]](https://drive.google.com/file/d/1IkUGkIH86ERpAPQeJNckgLdwQ8yBCNgk/view?usp=sharing)


## Usage
1. Define the paths in 'local_config.py'.

2. Select the pre-trained model and testing dataset by modifying 'main_runfile.py'.

3. Run BMVOS!
```
python main_runfile.py
```



## Note
Code and models are only available for non-commercial research purposes.

If you have any questions, please feel free to contact me :)
```
E-mail: chosuhwan@yonsei.ac.kr
```
