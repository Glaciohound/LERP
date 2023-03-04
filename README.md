# LERP (Logical Entity RePresentation)
Official Repository for
[ICLR 2023 Paper: Logical Entity Representation in Knowledge-Graphs for Differentiable Rule Learning](https://arxiv.org/abs/2305.12738)
by [Chi Han](https://glaciohound.github.io),
[Qizheng He](https://sites.google.com/site/qizhenghe96/home),
Charles Yu,
[Xinya Du](https://xinyadu.github.io),
[Hanghang Tong](http://tonghanghang.org),
[Heng Ji](http://blender.cs.illinois.edu/hengji.html).



## Requirements
- PyTorch 1.12.1
- NumPy


## Running the Code

Scripts for running LERP on graph completion dataset are stored in `./scripts/` directory.
Before running the scripts, please specify a general directory `$LOG` for storing all checkpoints and logs, and each trial will create a sub-directory inside it for storage purposes. Please be aware that the checkpoint can be large depending on the task and configuration, and can occupy up to 4GB of space, so we advise carefully selecting the position in your disk.
You can also specify the gpu you want to use into `$CUDA_VISIBLE_DEVICES` argument. If none is specified, the system will default to 0.

Besides running the commands in the following way, you can also directly pass in the arguments like `bash scripts/run_wn18rr.sh ./log 0`.

To train and evaluate LERP on Family dataset:
```
bash scripts/run_family.sh $LOG $CUDA_VISIBLE_DEVICES
```

On UMLS dataset:
```
bash scripts/run_umls.sh $LOG $CUDA_VISIBLE_DEVICES
```

On Kinship dataset:
```
bash scripts/run_kinship.sh $LOG $CUDA_VISIBLE_DEVICES
```

On WN18RR dataset:
```
bash scripts/run_wn18rr.sh $LOG $CUDA_VISIBLE_DEVICES
```

On WN-18 dataset:
```
bash scripts/run_wn18.sh $LOG $CUDA_VISIBLE_DEVICES
```



## Acknowledgement
This code partially is borrowed from [Neural LP](https://github.com/fanyangxyz/Neural-LP) and [DRUM](https://github.com/alisadeghian/DRUM).

## Citation

To cite our paper, please use the following BibTeX:
```
@inproceedings{han2023logical,
  title={Logical Entity Representation in Knowledge-Graphs for Differentiable Rule Learning},
  author={Han, Chi and He, Qizheng and Yu, Charles and Du, Xinya and Tong, Hanghang and Ji, Heng},
  booktitle={The Eleventh International Conference on Learning Representations},
  year={2023}
}
```
