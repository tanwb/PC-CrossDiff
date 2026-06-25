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

TORCH_DISTRIBUTED_DEBUG=INFO CUDA_VISIBLE_DEVICES=1,3,4,5,6 python -m torch.distributed.launch \
    --nproc_per_node 5 --master_port 4444 \
    train_dist_mod.py --num_decoder_layers 6 \
    --use_color \
    --weight_decay 0.0005 \
    --data_root /trxydsjtwb/data/data_set/3DVG_Data \
    --val_freq 1 --batch_size 6 --save_freq 1 --print_freq 500 \
    --lr_backbone=1e-3 --lr=1e-4 \
    --dataset nr3d --test_dataset nr3d \
    --detect_intermediate --joint_det \
    --use_soft_token_loss --use_contrastive_align \
    --log_dir "${LOG_ROOT}/" \
    --lr_decay_epochs 150 \
    --pp_checkpoint /trxydsjtwb/data/data_set/3DVG_Data/gf_detector_l6o256.pth \
    --checkpoint_path none \
    --butd_cls --self_attend \
    --max_epoch 240 \
    --dgt_start_epoch 100 --dgt_stop_epoch 120 \
    --dgt_ratio_diff_threshold 0.020 \
    --dgt_clamp_min 0.998 --dgt_clamp_max 1.002 --dgt_main_max_step 0.0005 \
    --dgt_geom_weight 0.003 --dgt_geom_cap_ratio 0.003 \
    --dgt_core_residual_scale 0.020 --gs_residual_scale 0.015 --dgt_geom_residual_scale 0.500 \
    --main_score_mode det50 \
    --gs_harmonize_weight 0.0020 --gs_harmonize_cap_ratio 0.006 --gs_tau -0.05 \
    --gs_det_good_thr 0.55 --gs_seg_good_thr 0.55 --gs_agree_good_thr 0.50 \
    --gs_det_bad_thr 0.50 --gs_seg_bad_thr 0.50 --gs_agree_bad_thr 0.45 \
    --dgt_val_patience 1 --dgt_val_min_delta 0.0 --dgt_val_drop_tolerance 0.0015 --dgt_val_severe_drop 0.0025 \
    --gs_quality_blend 0.70 \
    --gs_late_start_epoch 140 --gs_activation_drop 0.0035 --gs_det_activation_drop 0.0045 \
    --geom_late_start_epoch 160 --geom_gap_threshold 0.052 --geom_gap_worsen_threshold 0.004 --geom_quality_floor 0.40 --geom_seg_stable_drop 0.006 \
    --late_stage_confirm_epochs 2 --late_stage_cooldown_epochs 2 --late_stage_kill_drop 0.0025 \
    --enable_grad_consistency \
    --model PC_CrossDiff \
    --exp "PC_CrossDiff"
