
CUDA_VISIBLE_DEVICES=0 python -m pdb training/exp_runner.py --conf ./confs/ShapeNet-triplets.conf --batch_size 4 --is_continue

#CUDA_VISIBLE_DEVICES=0 python training/exp_runner.py --conf ./confs/ShapeNet-triplets.conf --batch_size 1