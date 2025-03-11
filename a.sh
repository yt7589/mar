torchrun --nproc_per_node=1 --nnodes=1 --node_rank=0 --master_addr=192.168.1.115 --master_port=8002 \
main_mar.py \
--img_size 256 --vae_path work/pretrained_models/vae/kl16.ckpt --vae_embed_dim 16 --vae_stride 16 --patch_size 1 \
--model mar_base --diffloss_d 6 --diffloss_w 1024 \
--epochs 400 --warmup_epochs 100 --batch_size 4 --blr 1.0e-4 --diffusion_batch_mul 4 \
--output_dir ./work/outputs  \
--data_path ./work/datasets/ImageNet-Mini
