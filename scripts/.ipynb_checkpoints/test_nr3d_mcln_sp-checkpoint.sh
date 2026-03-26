TORCH_DISTRIBUTED_DEBUG=INFO CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  python -m torch.distributed.launch \
    --nproc_per_node 8 --master_port 2222 \
    train_dist_mod.py --num_decoder_layers 6 \
    --use_color \
    --weight_decay 0.0005 \
    --data_root /dsjxytest/can_not_remove/data/MCLN_data \
    --val_freq 3 --batch_size 12 --save_freq 3 --print_freq 500 \
    --lr_backbone=1e-3 --lr=1e-4 \
    --dataset nr3d --test_dataset nr3d \
    --detect_intermediate --joint_det \
    --use_soft_token_loss --use_contrastive_align \
    --log_dir /dsjxytest/can_not_remove/model/vg3d/all_logs/ \  
    --lr_decay_epochs 150 \
    --butd_cls --self_attend \
    --checkpoint_path /twb/twb/model/vg3d/MCLN-main/output/checkpoints0/ckpt_epoch_54.pth \
    --eval \
    --model MCLN \
    --exp MCLN \