
# Accelerated Multi-Modal MR Imaging with Transformers

## Dependencies
* numpy==1.18.5
* scikit_image==0.16.2
* torchvision==0.8.1
* torch==1.7.0
* runstats==1.8.0
* pytorch_lightning==1.0.6
* h5py==2.10.0
* PyYAML==5.4



**multi gpu train**
```bash
python -m torch.distributed.launch --nproc_per_node=8   train.py --experiment sr_multi_cross
```

**single gpu train**
```bash
python train.py --experiment sr_multi_cross
```

**multi gpu test**
```bash
python -m torch.distributed.launch --nproc_per_node=8   test.py --experiment sr_multi_cross
```

**single gpu test**
```bash
python test.py --experiment sr_multi_cross
```

```--experiment``` is the experiment you running. And you can change the config file to set the parameter when training or testing.


Citation


```
@misc{feng2021accelerated,
      title={Accelerated Multi-Modal MR Imaging with Transformers}, 
      author={Chun-Mei Feng and Yunlu Yan and Geng Chen and Huazhu Fu and Yong Xu and Ling Shao},
      year={2021},
      eprint={2106.14248},
      archivePrefix={arXiv},
      primaryClass={eess.IV}
}

```
