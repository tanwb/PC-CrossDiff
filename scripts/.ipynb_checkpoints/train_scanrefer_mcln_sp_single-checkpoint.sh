TORCH_DISTRIBUTED_DEBUG=INFO CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch \
    --nproc_per_node 8 --master_port 4444 \
    train_dist_mod.py --num_decoder_layers 6 \
    --use_color \
    --weight_decay 0.0005 \
    --data_root /dsjxytest/can_not_remove/data/MCLN_data \
    --val_freq 1 --batch_size 4 --save_freq 1 --print_freq 500 \
    --lr_backbone=2e-3 --lr=2e-4 \
    --dataset scanrefer --test_dataset scanrefer \
    --detect_intermediate --joint_det \
    --use_soft_token_loss --use_contrastive_align \
    --log_dir /dsjxytest/can_not_remove/model/vg3d/all_logs/ \
    --lr_decay_epochs 50 75 \
    --pp_checkpoint /dsjxytest/can_not_remove/data/MCLN_data/gf_detector_l6o256.pth \
    -self_attend --augment_det \
    --max_epoch 100 \
    --model MCLN \
    # --small_lr \
    --exp MCLN \