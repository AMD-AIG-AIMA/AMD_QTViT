# QT-ViT: Improving Linear Attention in ViT with Quadratic Taylor Expansion ([paper](https://openreview.net/pdf?id=V2e0A2XIPF)）



## Getting Started

### Installation

```bash
conda create -n qtvit python=3.10
conda activate qtvit
conda install mpi4py openmpi
pip install einops opencv-python timm==0.6.13 tqdm torchprofile matplotlib transformers onnx onnxsim onnxruntime pycocotools
pip install git+https://github.com/zhijian-liu/torchpack.git@3a5a9f7ac665444e1eb45942ee3f8fc7ffbd84e5
```

### Dataset

<details>
  <summary>ImageNet: https://www.image-net.org/</summary>
</details>

  ```python
  Our code expects the ImageNet dataset directory to follow the following structure:

  imagenet
  ├── train
  ├── val
  ```

## Results

### ImageNet


| Model    | Top1 Acc | ImageNet Top5 Acc | Params |  MACs  |
| -------- | :------: | :---------------: | :----: | :----: |
| QT-ViT-1 |   79.6   |       94.7        |  9.4M  | 0.52G  |
| QT-ViT-2 |   82.5   |       95.9        | 24.9M  | 1.60G  |
| QT-ViT-3 |   83.9   |       96.7        | 49.7M  | 3.97G  |
| QT-ViT-4 |   84.7   |       96.7        | 53.0M  | 5.26G  |
| QT-ViT-5 |   85.2   |       97.0        | 64.1M  | 6.96G  |
| QT-ViT-6 |   86.0   |       97.3        | 246.8M | 27.60G |



## Training

``` 
torchpack dist-run -np 8 \
python train_cls_model.py configs/cls/imagenet/b1.yaml \ 
	--data_provider.image_size "[128,160,192,224]" \
	--run_config.eval_image_size "[224]" \
	--path ./exp/cls/imagenet/b1_224/
```



## Citation

```
@inproceedings{xu2024qtvit,
  title={QT-ViT: Improving Linear Attention in ViT with Quadratic Taylor Expansion},
  author={Xu, Yixing and Li, Chao and Li, Dong and Sheng, Xiao and Jiang, Fan and Tian, Lu and Barsoum, Emad},
  booktitle={The Thirty-eighth Annual Conference on Neural Information Processing Systems}
}
```
