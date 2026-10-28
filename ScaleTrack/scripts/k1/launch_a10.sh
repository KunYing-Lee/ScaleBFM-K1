#!/usr/bin/env bash
set -euo pipefail
cd /localssd/lky/ScaleBFM-K1/ScaleTrack
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export TORCHINDUCTOR_COMPILE_THREADS=2
export SCALETRACK_COMPILE_CACHE_DIR=/localssd/lky/scalebfm-k1-cache/compiled
export PYTHONUNBUFFERED=1
exec /localssd/lky/scalebfm-k1-venv/bin/python -m torch.distributed.run --standalone --nproc_per_node=8 \
 scripts/pretrain/rsl_rl/train.py --distributed --headless \
 --task K1-BFM-Transformer-Tracking-SuperMimic \
 --motion_file /localssd/lky/scalebfm-k1-data/combined_exact/train \
 --test_motion_file /localssd/lky/scalebfm-k1-data/combined_exact/eval \
 --num_envs 512 --seed 42 --max_iterations 320000 --run_name k1_bones_lafan_a10_full \
 agent.supermimic.wm_batch_size=256 \
 agent.supermimic.imagination_batch_size=256 \
 agent.supermimic.actor_micro_batch_size=256 \
 agent.supermimic.actor_checkpointing=true \
 agent.supermimic.lr_scale_with_world_size=false "$@"
