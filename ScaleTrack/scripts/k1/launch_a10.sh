#!/usr/bin/env bash
set -euo pipefail
# Run from any directory with the active IsaacLab Python environment.
: "${K1_DATA_DIR:?Set K1_DATA_DIR to the packed library containing train/ and eval/}"
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export TORCHINDUCTOR_COMPILE_THREADS=2
export SCALETRACK_COMPILE_CACHE_DIR="${SCALETRACK_COMPILE_CACHE_DIR:-${XDG_CACHE_HOME:-$HOME/.cache}/scalebfm-k1/compiled}"
export PYTHONUNBUFFERED=1
exec "${PYTHON:-python}" -m torch.distributed.run --standalone --nproc_per_node="${NPROC_PER_NODE:-8}" \
 scripts/pretrain/rsl_rl/train.py --distributed --headless \
 --task K1-BFM-Transformer-Tracking-SuperMimic \
 --motion_file "$K1_DATA_DIR/train" \
 --test_motion_file "$K1_DATA_DIR/eval" \
 --num_envs 512 --seed 42 --max_iterations 320000 --run_name k1_bones_lafan_a10_full \
 agent.supermimic.wm_batch_size=256 \
 agent.supermimic.imagination_batch_size=256 \
 agent.supermimic.actor_micro_batch_size=256 \
 agent.supermimic.actor_checkpointing=true \
 agent.supermimic.lr_scale_with_world_size=false "$@"
