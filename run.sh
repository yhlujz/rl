export CUDA_VISIBLE_DEVICES=2,3
torchrun --nproc_per_node=2 train_map_ddp.py 