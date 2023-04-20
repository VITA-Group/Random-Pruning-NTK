# On the Neural Tangent Kernel Analysis of Randomly Pruned Neural Networks

### Environment

We recommend to create a conda environment and install the following:
* Python 3.7.11
* Pytorch 1.8.0 (GPU-enabled version)

In some cases, the package ```request``` might be missing in the environment. To install request,
```
python -m pip install requests
```


### Test the environment is installed successfully
Simple run
```
python open_lth.py
```
In response, you will see the following message.

```
==================================================================================
OpenLTH: A Framework for Research on Lottery Tickets and Beyond
----------------------------------------------------------------------------------
Choose a command to run:
    * open_lth.py train [...] => Train a model.
    * open_lth.py lottery [...] => Run a lottery ticket hypothesis experiment.
    * open_lth.py lottery_branch [...] => Run a lottery branch.
==================================================================================
```


### Welcome

This repo includes codes for the experiment implementation of the paper [*On the Neural Tangent Kernel Analysis of Randomly Pruned Neural Networks*](https://proceedings.mlr.press/v206/yang23b/yang23b.pdf), by Hongru Yang and Zhangyang Wang.

The implementation is heavily based on [Jonathan Frankle's implemenation](https://github.com/facebookresearch/open_lth) for experiments on the lottery ticket hypothesis, and further developed in Xiaohan Chen's [work](https://github.com/VITA-Group/ElasticLTH).

### New Command
To run random pruning, use the following command:
```
python open_lth.py lottery --default_hparams=cifar_resnet_50 --levels=16 --pruning_strategy=random_global --replicate=1
```

### Citation

If you find this useful, please cite the following paper:
```

@InProceedings{pmlr-v206-yang23b,
  title = 	 {On the Neural Tangent Kernel Analysis of Randomly Pruned Neural Networks},
  author =       {Yang, Hongru and Wang, Zhangyang},
  booktitle = 	 {Proceedings of The 26th International Conference on Artificial Intelligence and Statistics},
  pages = 	 {1513--1553},
  year = 	 {2023},
  editor = 	 {Ruiz, Francisco and Dy, Jennifer and van de Meent, Jan-Willem},
  volume = 	 {206},
  series = 	 {Proceedings of Machine Learning Research},
  month = 	 {25--27 Apr},
  publisher =    {PMLR},
  pdf = 	 {https://proceedings.mlr.press/v206/yang23b/yang23b.pdf},
  url = 	 {https://proceedings.mlr.press/v206/yang23b.html},
}

```

### License

This repository is licensed under the MIT license, as found in the LICENSE file.