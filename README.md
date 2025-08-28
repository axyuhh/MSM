# MSM

Official package for this paper (NIR-II Fluorescence Image Enhancement via Multi-Step Modulation).

### Preparations

Dataset: NIR-II fluorescence image data is sourced from [Project](https://github.com/zhuoranzma/Deep-learning-for-in-vivo-near-infrared-imaging).

Model: Our [pre-trained models](https://drive.google.com/drive/folders/182zT3aOv8MMXOB94OjLt_0UT8c2N3_S5?usp=sharing) can be obtained.

### Train

python train.py --dataroot /dataset --checkpoints_dir ./checkpoints --exp_name project --model_name season_transfer --gpu 0 --niter 100 --niter_decay 0 --n_attribute 2 --n_style 8 --batch_size 1 --is_flip --use_dropout

### Test

python test.py --dataroot /dataset --checkpoints_dir ./checkpoints --exp_name project --model_name season_transfer --gpu 0 --n_attribute 2 --n_style 8 --batch_size 1 --use_dropout

### Reference

The code is based on project [DMIT](https://github.com/Xiaoming-Yu/DMIT).

### Note

If this code is useful to you, please cite our paper.

```
@article{yu2025nir,
  title={NIR-II Fluorescence Image Enhancement via Multi-Step Modulation},
  author={Yu, Xiaoming and Tian, Jie and Hu, Zhenhua},
  journal={Expert Systems with Applications},
  pages={128728},
  year={2025},
  publisher={Elsevier}
}
```

If you want to use NIR-II fluorescence data and the synthetic data, please cite their [paper](https://github.com/zhuoranzma/Deep-learning-for-in-vivo-near-infrared-imaging).
