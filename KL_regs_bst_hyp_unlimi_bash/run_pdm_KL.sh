#!/bin/bash

#CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 /home/irving/federated_matching/FedMA/Semantic_Segmentation/save_weights.py --n_=2 --init_same=True --model=unet --bilinear=True --dataset=carvana --n_channels=3 --n_classes=1 --batch_size=8 --lr=0.0001 --epochs=2 --trials=10
python /Users/irving/Github_projects/probabilistic_federated_matching_KL/KL_reg_unlimi.py --layers=0 --n=2 --dataset='mnist' --net_config='784, 100, 10' --experiment='pdm_KL' --epochs=1 --trials=10 --batch_size=32 --num_pool_workers=1 --device=0