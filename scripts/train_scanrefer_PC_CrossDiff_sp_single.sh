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


LOG_ROOT=/twb/data/model/vg3d/all_logs

TORCH_DISTRIBUTED_DEBUG=INFO CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch \
    --nproc_per_node 8 --master_port 4589 \
    train_dist_mod.py --num_decoder_layers 6 \
    --use_color \
    --weight_decay 0.0005 \
    --data_root /twb/data/data_set/3DVG_Data \
    --val_freq 1 --batch_size 8 --save_freq 1 --print_freq 500 \
    --lr_backbone=2e-3 --lr=2e-4 \
    --dataset scanrefer --test_dataset scanrefer \
    --detect_intermediate --joint_det \
    --use_soft_token_loss --use_contrastive_align \
    --log_dir "${LOG_ROOT}/" \
    --lr_decay_epochs 50 75 \
    --self_attend --augment_det \
    --max_epoch 150 \
    --dgt_start_epoch 30 --dgt_stop_epoch 45 \
    --dgt_ratio_diff_threshold 0.008 \
    --dgt_clamp_min 0.995 --dgt_clamp_max 1.005 --dgt_main_max_step 0.0010 \
    --dgt_aux_clamp_min 0.995 --dgt_aux_clamp_max 1.005 --dgt_aux_max_step 0.0010 --dgt_aux_momentum 0.970 \
    --dgt_geom_weight 0.0 --dgt_geom_cap_ratio 0.008 \
    --dgt_core_residual_scale 0.040 --dgt_aux_residual_scale 0.020 --gs_residual_scale 0.020 --dgt_geom_residual_scale 0.012 \
    --dgt_core_residual_floor 0.0 --dgt_aux_residual_floor 0.0 --gs_residual_floor 0.0 --dgt_geom_residual_floor 0.0 \
    --dgt_floor_quality_thr 0.55 \
    --main_score_mode det50 \
    --gs_harmonize_weight 0.0015 --gs_harmonize_cap_ratio 0.006 --gs_tau -0.10 \
    --dgt_ce_late_start_epoch -1 --dgt_ce_late_warmup_epochs 6 --dgt_ce_boost_max 1.05 --dgt_sem_boost_max 1.08 \
    --gs_det_good_thr 0.55 --gs_seg_good_thr 0.55 --gs_agree_good_thr 0.50 \
    --gs_det_bad_thr 0.50 --gs_seg_bad_thr 0.50 --gs_agree_bad_thr 0.45 \
    --dgt_val_patience 2 --dgt_val_min_delta 0.0 --dgt_val_drop_tolerance 0.002 --dgt_val_severe_drop 0.003 \
    --gs_quality_blend 0.70 \
    --gs_late_start_epoch 60 --gs_activation_drop 0.0035 --gs_det_activation_drop 0.0045 \
    --geom_late_start_epoch 72 --geom_gap_threshold 0.060 --geom_gap_worsen_threshold 0.012 --geom_quality_floor 0.40 --geom_seg_stable_drop 0.004 \
    --late_stage_confirm_epochs 2 --late_stage_cooldown_epochs 2 --late_stage_kill_drop 0.0035 \
    --enable_grad_consistency \
    --model PC_CrossDiff \
    --exp "PC_CrossDiff"
