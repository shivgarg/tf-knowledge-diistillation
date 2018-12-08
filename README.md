# Introduction
This repo contains code to train a model for imagenet classification( can be extended to other classification tasks) by using [Knowledge Distillation](https://arxiv.org/abs/1503.02531). Usage:
```
python train.py --dataset_dir "path to imagenet tfrecords" --teacher_network "teacher network name" --teacher_ckpt "teacher pretrained ckpt" --student_network "student network name"  -student_scope "student network scope" --checkpoint_dir "ckpt dir" --Temperature "temperature for knowledge distillation" --optimiser "optimiser type"
```
1. Available networks are listed in this [file](nets/nets_factory.py) as a network_map.
2. The pretrained models on imagenet for teacher can be downloaded from [here](https://github.com/tensorflow/models/tree/master/research/slim#Pretrained).
3. To fine a pretrained student network, pass the pretrained ckpt as "--student_ckpt "student ckpt path" " flag.

This code was used to develop light weight models for [LPIRC](https://rebootingcomputing.ieee.org/lpirc).

More documentation will follow soon.
