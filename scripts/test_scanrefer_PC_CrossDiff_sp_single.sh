#!/bin/bash
set -euo pipefail
set -m

kill_existing_training() {
    echo "[cleanup] stop existing training processes"
    pkill -9 -f "train_dist_mod.py" 2>/dev/null || true
    pkill -9 -f "torchrun" 2>/dev/null || true
    if [ "$(jobs -r | wc -l)" -gt 0 ]; then
        kill -9 $(jobs -p) 2>/dev/null || true
    fi
    sleep 2
    echo "[cleanup] done"
}

kill_existing_training


LOG_ROOT=/trxydsjtwb/data/model/vg3d/all_logs

TORCH_DISTRIBUTED_DEBUG=INFO CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch \
    --nproc_per_node 8 --master_port 1111 \
    train_dist_mod.py --num_decoder_layers 6 \
    --use_color \
    --weight_decay 0.0005 \
    --data_root /trxydsjtwb/data/data_set/3DVG_Data \
    --val_freq 3 --batch_size 8 --save_freq 3 --print_freq 500 \
    --lr_backbone=2e-3 --lr=2e-4 \
    --dataset scanrefer --test_dataset scanrefer \
    --detect_intermediate --joint_det \
    --use_soft_token_loss --use_contrastive_align \
    --log_dir "${LOG_ROOT}/" \
    --lr_decay_epochs 50 75 \
    --self_attend --augment_det \
    --checkpoint_path /trxydsjtwb/data/model/vg3d/all_logs/PC_CrossDiff/scanrefer/XXX/ckpt_epoch_1.pth \
    --model PC_CrossDiff \
    --small_lr \
    --exp "PC_CrossDiff" \
    --eval
